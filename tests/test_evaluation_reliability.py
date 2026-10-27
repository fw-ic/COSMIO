# =============================================================================
# COSMIO: A Benchmark for Cross-Survey Modality Imputation
#
# Copyright (c) 2026 Futurewei Technologies, Inc.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# =============================================================================

from __future__ import annotations

import copy
import importlib.util
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "asyn-diffusion" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_imputation_methods as evaluator
import evaluate_lensing_arc_metrics as arc
from asyndiff import config as config_module
from asyndiff.data.protocol import center_pad_tensor, restore_to_native
from asyndiff.models.diffusion import GaussianDiffusion
from asyndiff.ops.metrics import fair_crps_ensemble
from asyndiff.ops.random import SEED_SCHEME, sample_seed, sample_generators
from asyndiff.ops.visualization import append_history
from asyndiff.training import pipeline
from asyndiff.training.common import create_dataset

spec = importlib.util.spec_from_file_location("eval_test_helpers", ROOT / "asyn-diffusion/tests/helpers.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.mark.parametrize("asinh", [False, True])
def test_evaluation_of_train_split_has_no_augmentation(tmp_path, asinh):
    config = helpers.make_test_config(tmp_path, normalized_with_asinh=asinh)
    config["data"].update(augment_flip_prob=1.0, augment_shift_prob=1.0, augment_scale_prob=1.0)
    evaluation = create_dataset(config, "train", training=False)
    training = create_dataset(config, "train", training=True)
    assert not evaluation.train and training.train
    one, two = evaluation[0], evaluation[0]
    augmented = training[0]
    for modality in one["inputs"]:
        torch.testing.assert_close(one["inputs"][modality], two["inputs"][modality], rtol=0, atol=0)
        torch.testing.assert_close(one["native_targets"][modality], two["native_targets"][modality], rtol=0, atol=0)
        assert not torch.equal(augmented["inputs"][modality], one["inputs"][modality])
    assert not create_dataset(config, "train").train
    assert not create_dataset(config, "train", visualization_subset=True, training=True).train


def test_geometry_restores_resize_and_padding_by_provenance():
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 31), torch.linspace(-1, 1, 29), indexing="ij")
    native = torch.exp(-8 * (xx.square() + yy.square()))[None]
    resized = F.interpolate(native[None], (64, 64), mode="bilinear", align_corners=False)[0]
    restored = restore_to_native(resized, native, spatial_mode="resize")
    incorrectly_cropped = restore_to_native(resized, native, spatial_mode="padding")
    assert (restored - native).square().mean() < (incorrectly_cropped - native).square().mean() / 100
    padded, _ = center_pad_tensor(native, (64, 64))
    torch.testing.assert_close(restore_to_native(padded, native, spatial_mode="padding"), native, rtol=0, atol=0)
    torch.testing.assert_close(evaluator._resize_to_native(resized, native, spatial_mode="resize"), restored)
    with pytest.raises(ValueError, match="native output"):
        evaluator._resize_to_native(resized, native)
    with pytest.raises(ValueError, match="smaller"):
        restore_to_native(native, padded, spatial_mode="padding")


@pytest.mark.parametrize("asinh", [False, True])
def test_clipping_matches_normalization(tmp_path, asinh):
    config = helpers.make_test_config(tmp_path, normalized_with_asinh=asinh)
    diffusion = pipeline._build_diffusion(config, torch.device("cpu"))
    assert diffusion.clip_x0 is asinh
    config["diffusion"]["clip_x0"] = False
    assert not pipeline._build_diffusion(config, torch.device("cpu")).clip_x0


class PointwiseModel(torch.nn.Module):
    num_heads = 1

    def forward(self, x, t, head_index=None):
        return x * 0.17


@pytest.mark.parametrize("sampler", ["ddpm", "ddim"])
@pytest.mark.parametrize("fix_mode", ["clean", "noised"])
def test_all_stochastic_steps_are_record_stable(sampler, fix_mode):
    diffusion = GaussianDiffusion(PointwiseModel(), timesteps=7, clip_x0=False)
    source = torch.linspace(-0.5, 0.5, 3 * 2 * 8 * 8).reshape(3, 2, 8, 8)
    ids = ["shard:a", "shard:b", "shard:c"]
    def sample(indices, draw=0):
        x0 = source[indices]
        mask = torch.zeros_like(x0)
        mask[:, :1] = 1
        generators = sample_generators([sample_seed(123, ids[i], draw) for i in indices], "cpu")
        kwargs = dict(fix_mode=fix_mode, generators=generators)
        if sampler == "ddpm":
            return diffusion.sample_with_mask(x0, mask, **kwargs)
        return diffusion.sample_ddim_with_mask(x0, mask, steps=[6, 4, 2, 0], eta=1.0, **kwargs)
    together = sample([0, 1, 2])
    independent = torch.cat([sample([i]) for i in range(3)])
    reordered = sample([2, 0, 1])[[1, 2, 0]]
    torch.testing.assert_close(together, independent, rtol=0, atol=0)
    torch.testing.assert_close(together, reordered, rtol=0, atol=0)
    torch.testing.assert_close(together[:, :1], source[:, :1], rtol=0, atol=0)
    assert not torch.equal(together[:, 1:], sample([0, 1, 2], draw=1)[:, 1:])


@pytest.mark.parametrize("asinh", [False, True])
def test_real_adapter_batch_order_and_saved_seed_invariance(tmp_path, monkeypatch, asinh):
    config = helpers.make_test_config(tmp_path / "fixture", normalized_with_asinh=asinh)
    config["model"]["head_mode"] = "single"
    config["diffusion"]["timesteps"] = 6
    config["evaluation"]["inference_steps"] = 4
    config["data"].update(augment_flip_prob=1.0, augment_shift_prob=1.0, augment_scale_prob=1.0)
    checkpoint = tmp_path / "checkpoint.pt"
    diffusion = pipeline._build_diffusion(config, torch.device("cpu"))
    torch.save({"config": config, "model": diffusion.model.state_dict()}, checkpoint)
    monkeypatch.setattr(config_module, "load_with_overrides", lambda path: copy.deepcopy(config))
    runs = []
    for run, (indices, batch_size) in enumerate([([0, 1], 2), ([1, 0], 1)]):
        adapter = evaluator.AsynDiffusionAdapter(checkpoint_path=checkpoint)
        result = evaluator._evaluate_method(
            adapter, "train", indices, ["lsst", "euclid"], ["roman"], 2, 1234, batch_size, 0,
            pred_sample_normalized_root=tmp_path / str(run),
        )
        assert not adapter._dataset("train").train
        assert result["seed_scheme"] == SEED_SCHEME
        assert result["clip_x0"] is asinh
        assert result["geometry_policy"].startswith("padding" if asinh else "resize")
        runs.append({row["record_id"]: row for row in result["examples"]})
        adapter.close()
    for rid, row in runs[0].items():
        other = runs[1][rid]
        for key in ("rmse", "psnr", "crps"):
            assert row["metrics"]["roman"][key] == pytest.approx(other["metrics"]["roman"][key], rel=1e-5, abs=1e-6)
        saved = [torch.load(item["pred_sample_normalized_paths"]["roman"], weights_only=True) for item in (row, other)]
        assert saved[0]["seed"] == saved[1]["seed"] == sample_seed(1234, rid, 0)
        assert saved[0]["seed_scheme"] == SEED_SCHEME
        torch.testing.assert_close(saved[0]["pred_normalized"], saved[1]["pred_normalized"], rtol=1e-5, atol=2e-6)


class SyntheticAdapter(evaluator.MethodAdapter):
    def __init__(self, bad_prediction=False, bad_target=False):
        super().__init__("synthetic")
        self.bad_prediction, self.bad_target = bad_prediction, bad_target

    def native_targets(self, split, sample_index):
        return "record:bad-or-good", "source", {"roman": torch.full((1, 4, 4), float("nan") if self.bad_target else 1.0)}

    def infer_example(self, split, sample_index, known_modalities, output_stem):
        return {"roman": torch.full((1, 4, 4), float("inf")) if self.bad_prediction else torch.rand(1, 4, 4)}

    def split_size(self, split):
        return 1


@pytest.mark.parametrize("bad_prediction,bad_target", [(True, False), (False, True)])
def test_evaluator_fails_with_record_context(bad_prediction, bad_target):
    with pytest.raises(ValueError, match="synthetic.*record_id=record:bad-or-good"):
        evaluator._evaluate_method(SyntheticAdapter(bad_prediction, bad_target), "val", [0], ["lsst", "euclid"], ["roman"], 2, 123, 1, 0)


def test_evaluator_cli_writes_strict_json_and_seed_protocol(tmp_path, monkeypatch):
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(evaluator, "_build_adapter", lambda **kwargs: SyntheticAdapter())
    monkeypatch.setattr(sys, "argv", ["evaluate", "--known", "lsst,euclid", "--crps-samples", "2", "--output", str(output)])
    evaluator.main()
    result = json.loads(output.read_text(), parse_constant=lambda token: pytest.fail(token))
    assert result["seed"] == 1234 and result["seed_scheme"] == SEED_SCHEME


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_crps_and_arc_reject_nonfinite_inputs(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        fair_crps_ensemble(torch.tensor([0.0, bad]).reshape(2, 1, 1, 1), torch.zeros(1, 1, 1))
    with pytest.raises(ValueError, match="nonfinite"):
        arc.compute_lensing_arc_metrics(np.full((9, 9), bad), np.zeros((9, 9)))
    with pytest.raises(ValueError, match="nonfinite"):
        arc._denormalize_roman(np.full((3, 9, 9), bad), {})


def test_arc_undefined_values_have_reasons_counts_and_strict_null(tmp_path):
    blank = np.zeros((9, 9))
    row = arc.compute_lensing_arc_metrics(blank, blank, gt_arc_mask=blank.astype(bool), pred_arc_mask=blank.astype(bool))
    assert row["undefined_reason"] == "no_gt_arc+no_pred_arc"
    aggregate = arc._aggregate_rows([row])
    assert aggregate["valid_pair_count"] == 0
    assert aggregate["undefined_pair_count"] == 1
    assert aggregate["undefined_reason_counts"] == {"no_gt_arc+no_pred_arc": 1}
    payload = {"methods": {"synthetic": {"aggregate": aggregate, "examples": [row]}}}
    encoded = json.dumps(arc._finite_or_none(payload), allow_nan=False)
    assert json.loads(encoded)["methods"]["synthetic"]["examples"][0]["angular_extent_abs_error"] is None
    path = tmp_path / "arc.json"
    arc._write_summary_tsv(path, payload)
    assert "null" in path.with_suffix(".tsv").read_text()
    unexplained = dict(row, undefined_reason=None)
    with pytest.raises(ValueError, match="Unexpected nonfinite"):
        arc._aggregate_rows([unexplained])


def test_arc_cli_reports_pair_coverage_without_changing_conditional_aee(tmp_path, monkeypatch, capsys):
    complete = [dict(angular_extent_gt=2.0, angular_extent_pred=2.0 + error,
                     angular_extent_abs_error=error, valid_gt_arc=True,
                     valid_pred_arc=True, undefined_reason=None)
                for error in (0.10, 1.20, 1.30, 0.05)]
    partial = [dict(row) for row in complete]
    partial[1].update(angular_extent_pred=float("nan"), angular_extent_abs_error=float("nan"),
                      valid_pred_arc=False, undefined_reason="no_pred_arc")
    partial[2].update(angular_extent_gt=float("nan"), angular_extent_abs_error=float("nan"),
                      valid_gt_arc=False, undefined_reason="no_gt_arc")
    undetected = [dict(row, angular_extent_pred=float("nan"), angular_extent_abs_error=float("nan"),
                       valid_pred_arc=False, undefined_reason="no_pred_arc") for row in complete]
    methods = {name: {"aggregate": arc._aggregate_rows(rows), "examples": rows}
               for name, rows in (("complete", complete), ("partial", partial), ("undetected", undetected))}
    monkeypatch.setattr(arc, "evaluate_method", lambda method_name, **kwargs: methods[method_name])
    monkeypatch.setattr(arc, "_load_normalization_stats", lambda root: {})
    output = tmp_path / "arc.json"
    argv = ["evaluate_lensing_arc_metrics", "--lens-data-root", str(tmp_path), "--output", str(output)]
    for method in methods:
        argv.extend(["--eval-json", f"{method}={tmp_path / (method + '.json')}"])
    monkeypatch.setattr(sys, "argv", argv)
    arc.main()
    payload = json.loads(output.read_text(), parse_constant=lambda token: pytest.fail(token))
    aggregates = {name: result["aggregate"] for name, result in payload["methods"].items()}
    assert aggregates["complete"]["angular_extent_abs_error"]["median"] == pytest.approx(0.65)
    assert aggregates["partial"]["angular_extent_abs_error"]["median"] == pytest.approx(0.075)
    # Prediction-only detection is not the denominator population of conditional AEE.
    assert aggregates["partial"]["valid_pred_arc_rate"] == 0.75
    assert aggregates["partial"]["valid_pair_rate"] == 0.5
    with output.with_suffix(".tsv").open() as handle:
        table = {row["method"]: row for row in csv.DictReader(handle, delimiter="\t")}
    assert table["partial"]["valid_pair_count"] == "2"
    assert table["partial"]["count"] == "4"
    assert float(table["partial"]["valid_pair_rate"]) == 0.5
    assert float(table["partial"]["angular_extent_abs_error_median"]) == pytest.approx(0.075)
    assert table["undetected"]["angular_extent_abs_error_median"] == "null"
    assert float(table["undetected"]["valid_pair_rate"]) == 0.0
    console = capsys.readouterr().out.splitlines()
    header = next(line for line in console if line.startswith("method\t")).split("\t")
    row = dict(zip(header, next(line for line in console if line.startswith("partial\t")).split("\t")))
    assert row["valid_pair_count"] == "2" and row["count"] == "4"
    assert float(row["valid_pair_rate"]) == 0.5


@pytest.mark.parametrize("reader", ["arc", "visualizer"])
@pytest.mark.parametrize("path_kind", ["parent", "absolute", "symlink"])
def test_record_id_readers_reject_paths_outside_data_root(tmp_path, reader, path_kind):
    from vit_lens_classification.visualize_imputed_roman_samples import _record_to_shard_path

    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.zarr"
    outside.mkdir()
    if path_kind == "parent":
        record_id = "../outside.zarr:0"
    elif path_kind == "absolute":
        record_id = f"{outside}:0"
    else:
        link = root / "linked.zarr"
        link.symlink_to(outside, target_is_directory=True)
        record_id = "linked.zarr:0"
    cache = arc.LensZarrCache(root, "roman", "source_lensed_clean_psf", "deflector_clean_psf")
    resolve = cache._parse_record_id if reader == "arc" else lambda value: _record_to_shard_path(root, value)
    assert resolve("run_000/shards/lens/lens_000000.zarr:2") == (root / "run_000/shards/lens/lens_000000.zarr", 2)
    with pytest.raises(ValueError, match="outside the data root"):
        resolve(record_id)


def test_prediction_loader_refuses_executable_pickle_payload(tmp_path):
    marker = tmp_path / "executed.txt"

    class ExecutablePayload:
        def __reduce__(self):
            expression = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')"
            return eval, (expression,)

    prediction = tmp_path / "prediction.pt"
    torch.save(ExecutablePayload(), prediction)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        arc._prediction_tensor_from_payload(prediction, "roman")
    assert not marker.exists()


def test_arc_method_typo_does_not_fall_back_to_the_only_method(tmp_path):
    results = tmp_path / "evaluation.json"
    examples = [{"record_id": "run_000/shards/lens/lens_000000.zarr:0"}]
    results.write_text(json.dumps({"methods": {"asyn-diffusion": {"examples": examples}}}))
    assert arc._method_examples(results, "asyn-diffusion") == examples
    with pytest.raises(KeyError, match="Available.*asyn-diffusion"):
        arc._method_examples(results, "asyn-difusion")


def test_index_build_rejects_full_dataset_counts_on_small_input(tmp_path):
    from asyndiff.data.index import build_sample_index

    config = helpers.make_test_config(tmp_path)
    config["data"].update(train_count=90000, val_count=9000, test_count=0)
    with pytest.raises(ValueError, match="data.val_count=9000.*only 4 records"):
        build_sample_index(config)


@pytest.mark.parametrize("pred_second_yx, expected_pred, expected_error", [
    ((40, 29), np.pi / 2, 0.0),
    ((40, 40), np.pi / 4, np.pi / 4),
])
def test_arc_evaluation_entry_has_analytic_angular_span(tmp_path, pred_second_yx, expected_pred, expected_error):
    from argparse import Namespace

    source = np.zeros((59, 59))
    source[29, 40] = source[40, 29] = 1.0
    prediction = np.zeros_like(source)
    prediction[29, 40] = prediction[pred_second_yx] = 1.0
    deflector = np.zeros_like(source)
    deflector[29, 29] = 1.0
    saved = tmp_path / "prediction.pt"
    torch.save({"pred_normalized": torch.asinh(torch.from_numpy(prediction))[None].repeat(3, 1, 1)}, saved)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({"methods": {"synthetic": {"examples": [{
        "record_id": "run_000/shards/lens/lens_000000.zarr:0", "sample_index": 0,
        "sample_id": "synthetic", "pred_sample_normalized_paths": {"roman": str(saved)},
    }]}}}))

    class Cache:
        def source_and_deflector(self, record_id):
            return source, deflector

        def pixel_scale(self, record_id):
            return None

    stats = {band: dict(center=0.0, sigma=1.0, a_neg=1.0, a_pos=1.0) for band in arc.ROMAN_BANDS}
    args = Namespace(limit=None, target_modality="roman", prediction_field="sample",
                     mode="separate_masks", eps=1e-8, progress_every=0, no_examples=False)
    # Known-angle components isolate geometry while retaining both production extractors.
    result = arc.evaluate_method("synthetic", evaluation, Cache(), stats, args,
                                 arc.ArcMaskConfig(smooth_sigma=0.0, min_area=1))
    row = result["examples"][0]
    assert row["angular_extent_gt"] == pytest.approx(np.pi / 2, abs=1e-12)
    assert row["angular_extent_pred"] == pytest.approx(expected_pred, abs=1e-12)
    assert row["angular_extent_abs_error"] == pytest.approx(expected_error, abs=1e-12)
    assert result["aggregate"]["valid_pair_rate"] == 1.0


def test_history_rejects_unexplained_nonfinite_metrics(tmp_path):
    with pytest.raises(ValueError):
        append_history(tmp_path / "history.jsonl", {"crps": float("nan")})
    append_history(tmp_path / "valid.jsonl", {"crps": None, "undefined_metrics": {"crps": "no_evaluated_samples"}})
    json.loads((tmp_path / "valid.jsonl").read_text(), parse_constant=lambda token: pytest.fail(token))


def test_model_nonfinite_cannot_be_hidden_by_clipping():
    class InvalidModel(PointwiseModel):
        def forward(self, x, t, head_index=None):
            return torch.full_like(x, float("inf"))
    diffusion = GaussianDiffusion(InvalidModel(), timesteps=4, objective="x0", clip_x0=True)
    with pytest.raises(ValueError, match="model output.*nonfinite"):
        diffusion.p_sample(torch.zeros(1, 2, 4, 4), torch.zeros(1, dtype=torch.long))


def test_psnr_uses_each_gt_image_peak_across_channels():
    from asyndiff.ops.metrics import batch_reconstruction_metrics
    target = torch.full((2, 3, 2, 2), 2.0)
    target[1, 2] = 10.0
    # Image 0: peak=2, MSE=1 => 6.0206 dB.
    # Image 1: peak=10, MSE=4 => 13.9794 dB. Mean is 10 dB.
    prediction = target - torch.tensor([1.0, 2.0]).view(2, 1, 1, 1)
    score = batch_reconstruction_metrics(prediction, target)
    assert score['psnr'] == pytest.approx(10.0)
    assert evaluator._reconstruction_metrics(prediction[0], target[0])['psnr'] == pytest.approx(6.020599913279624)
    assert evaluator._reconstruction_metrics(prediction[1], target[1])['psnr'] == pytest.approx(13.979400086720377)


def test_psnr_is_invariant_to_positive_flux_unit_scaling():
    target = torch.tensor([2.0, 3.0, 4.0, 8.0]).reshape(1, 2, 2)
    prediction = target - 0.5
    a = evaluator._reconstruction_metrics(prediction, target)
    b = evaluator._reconstruction_metrics(prediction * 7, target * 7)
    assert a['psnr'] == pytest.approx(b['psnr'])
    assert b['rmse'] == pytest.approx(7 * a['rmse'])


@pytest.mark.parametrize('value', [0.0, -2.0, 3.0])
def test_psnr_preserves_independent_evaluator_numerical_floors(value):
    target = torch.full((1, 2, 2), value)
    score = evaluator._reconstruction_metrics(target, target)
    # Exact reconstruction retains the old 1e-12 MSE floor and peak floor.
    expected = 20 * np.log10(max(value, 1e-12)) + 120
    assert score['psnr'] == pytest.approx(expected)


@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_training_psnr_is_image_weighted_for_uneven_batches(monkeypatch, batch_size):
    from types import SimpleNamespace
    config = {'experiment': {'seed': 1}, 'evaluation': {
        'num_crps_samples': 2, 'known_combinations': [['lsst', 'euclid']],
    }}
    targets = torch.tensor([1.0, 2.0, 10.0]).view(3, 1, 1, 1).expand(3, 3, 2, 2)
    predictions = torch.tensor([0.0, 0.0, 9.0]).view(3, 1, 1, 1).expand_as(targets)
    loader = []
    for start in range(0, 3, batch_size):
        stop = min(3, start + batch_size)
        n = stop - start
        modalities = {'lsst': torch.zeros(n, 5, 2, 2), 'euclid': torch.zeros(n, 1, 2, 2), 'roman': predictions[start:stop]}
        loader.append({'record_id': [f'r:{i}' for i in range(start, stop)],
            'sample_id': [str(i) for i in range(start, stop)], 'label': torch.zeros(n),
            'inputs': modalities,
            'native_targets': {**modalities, 'roman': targets[start:stop]},
            'valid_masks': {name: torch.ones(n, 1, 2, 2) for name in modalities},
            'native_sizes': {name: (2, 2) for name in modalities}})
    monkeypatch.setattr(pipeline, '_sample_with_mask', lambda config, diffusion, x_full, *args, **kwargs: x_full)
    monkeypatch.setattr(pipeline, 'denormalize_tensor', lambda modality, tensor, stats: tensor)
    result = pipeline._evaluate(config, SimpleNamespace(stats={}, spatial_mode='native'), loader,
        SimpleNamespace(eval=lambda: None), torch.device('cpu'))
    # Individual PSNRs are 0, 0, 20 dB. Equal batch averaging would give 10 for B=2.
    assert result['roman_both_psnr'] == pytest.approx(20 / 3)
    assert result['mean_psnr'] == pytest.approx(20 / 3)


@pytest.mark.parametrize("provided_masks", [False, True])
def test_deleted_arc_mode_is_rejected_even_with_masks(provided_masks):
    image = np.ones((9, 9))
    masks = {"gt_arc_mask": image.astype(bool), "pred_arc_mask": image.astype(bool)} if provided_masks else {}
    with pytest.raises(ValueError, match="Unsupported arc metric mode"):
        arc.compute_lensing_arc_metrics(image, image, mode="gt_mask_diagnostic", **masks)


@pytest.mark.parametrize("asinh", [False, True])
def test_training_flips_are_shared_and_native_targets_remain_aligned(tmp_path, monkeypatch, asinh):
    import asyndiff.data.dataset as dataset_module
    from asyndiff.data.protocol import fit_tensor_to_size
    config = helpers.make_test_config(tmp_path, normalized_with_asinh=asinh)
    config["data"]["augment_flip_prob"] = 0.5
    dataset = create_dataset(config, "train", training=False)
    original = dataset[0]
    # Horizontal only; subsequent calls are disabled photometric decisions.
    draws = iter([0.1, 0.9] + [0.9] * 6)
    monkeypatch.setattr(dataset_module.random, "random", lambda: next(draws))
    augmented = create_dataset(config, "train", training=True)[0]
    for name in original["inputs"]:
        torch.testing.assert_close(augmented["native_targets"][name], torch.flip(original["native_targets"][name], [-1]))
        native = dataset_module._to_channel_first(dataset._get_group(dataset.records[0]["shard_path"])[name][0])
        expected, mask = fit_tensor_to_size(torch.flip(native, [-1]), (64, 64), use_padding=asinh)
        torch.testing.assert_close(augmented["inputs"][name], dataset_module._normalize(name, expected, dataset.stats))
        torch.testing.assert_close(augmented["valid_masks"][name], mask)


def test_cache_compares_requested_counts_not_resolved_remainder(tmp_path):
    from asyndiff.data.index import build_and_store_artifacts, artifacts_need_rebuild
    config = helpers.make_test_config(tmp_path)
    config["data"]["test_count"] = 0
    build_and_store_artifacts(config)
    index = json.loads(Path(config["data"]["index_path"]).read_text())
    assert index["summary"]["test"] == 1 and index["build_settings"]["counts"]["test"] == 0
    assert not artifacts_need_rebuild(config)
    config["data"]["selected_runs"] = ["run_000"]
    assert artifacts_need_rebuild(config)


@pytest.mark.parametrize("empty_split", ["train", "val"])
def test_diffusion_training_rejects_empty_required_splits_before_model(tmp_path, monkeypatch, empty_split):
    config = helpers.make_test_config(tmp_path)
    monkeypatch.setattr(pipeline, "artifacts_need_rebuild", lambda config: False)
    real_create = pipeline.create_dataset
    def create(config, split, **kwargs):
        dataset = real_create(config, split, **kwargs)
        if split == empty_split:
            dataset.records = []
        return dataset
    monkeypatch.setattr(pipeline, "create_dataset", create)
    monkeypatch.setattr(pipeline, "_build_diffusion", lambda *a, **k: pytest.fail("model must not be allocated"))
    with pytest.raises(ValueError, match=f"nonempty {empty_split}"):
        pipeline.train_asyn_diffusion(config)


def test_legacy_head_counts_are_validated_without_changing_head_mode():
    assert pipeline._configured_num_heads({"head_mode": "single", "num_heads": 3}) == 1
    for count in (1, 4):
        assert pipeline._configured_num_heads({"num_heads": count}) == count
    with pytest.raises(ValueError, match="num_heads=3"):
        pipeline._configured_num_heads({"num_heads": 3})


def test_arc_method_passes_actual_source_image(tmp_path, monkeypatch):
    import argparse
    source, prediction = np.ones((9, 9)), np.full((9, 9), 2.0)
    class Cache:
        def source_and_deflector(self, record_id):
            return source, source
        def pixel_scale(self, record_id):
            return None
    example = {"record_id": "record", "sample_index": 0, "sample_id": "sample"}
    monkeypatch.setattr(arc, "_method_examples", lambda *a: [example])
    monkeypatch.setattr(arc, "_resolve_prediction_path", lambda *a: tmp_path / "prediction.pt")
    monkeypatch.setattr(arc, "_prediction_tensor_from_payload", lambda *a: prediction)
    monkeypatch.setattr(arc, "_denormalize_roman", lambda image, stats: image)
    original = arc.compute_lensing_arc_metrics
    def capture(pred, gt, **kwargs):
        np.testing.assert_array_equal(gt, source)
        np.testing.assert_array_equal(pred, prediction)
        return original(pred, gt, **kwargs)
    monkeypatch.setattr(arc, "compute_lensing_arc_metrics", capture)
    args = argparse.Namespace(limit=None, target_modality="roman", prediction_field="sample",
                              mode="separate_masks", eps=1e-8, progress_every=0, no_examples=False)
    result = arc.evaluate_method("model", tmp_path / "eval.json", Cache(), {}, args, arc.ArcMaskConfig())
    assert len(result["examples"]) == 1


@pytest.mark.parametrize("new_noise", [None, 0.0, 1.0])
def test_noise_augmentation_name_and_legacy_alias(tmp_path, new_noise):
    config = helpers.make_test_config(tmp_path)
    config["data"]["augment_shift_prob"] = 1.0
    if new_noise is not None:
        config["data"]["augment_noise_prob"] = new_noise
    training = create_dataset(config, "train", training=True)
    assert training.augment_noise_prob == (1.0 if new_noise is None else new_noise)
    image = torch.zeros(1, 16, 16)
    transformed = training._augment(image)
    if new_noise == 0.0:
        torch.testing.assert_close(transformed, image, rtol=0, atol=0)
    else:
        assert not torch.equal(transformed, image)


class DistinctDrawsAdapter(SyntheticAdapter):
    def __init__(self):
        super().__init__()
        self.draw_calls = 0

    def native_targets(self, split, sample_index):
        return "record:distinct", "sample", {"roman": torch.full((1, 4, 4), 2.0)}

    def infer_example(self, split, sample_index, known_modalities, output_stem):
        self.draw_calls += 1
        return {"roman": torch.full((1, 4, 4), float(2 * self.draw_calls - 1))}

    def normalize_prediction_tensor(self, split, modality, tensor):
        return tensor / 10.0


@pytest.mark.parametrize("mean_root_given", [False, True])
def test_e5_sample_saves_scored_mean_and_preserves_downstream_sample(tmp_path, mean_root_given):
    adapter = DistinctDrawsAdapter()
    mean_root = tmp_path / "means" if mean_root_given else None
    result = evaluator._evaluate_method(
        adapter, "val", [0], ["lsst", "euclid"], ["roman"], 2, 123, 1, 0,
        pred_normalized_root=mean_root, pred_sample_normalized_root=tmp_path / "samples",
    )
    assert adapter.draw_calls == 2
    row = result["examples"][0]
    mean = torch.load(row["pred_normalized_path"], weights_only=True)
    sample = torch.load(row["pred_sample_normalized_paths"]["roman"], weights_only=True)
    torch.testing.assert_close(mean["pred_normalized"]["roman"], torch.full((1, 4, 4), 0.2))
    torch.testing.assert_close(sample["pred_normalized"], torch.full((1, 4, 4), 0.1))
    assert mean["prediction_kind"] == "ensemble_mean" and mean["ensemble_size"] == 2
    assert mean["mean_domain"] == "native_flux" and mean["scored_metrics"] == ["rmse", "psnr"]
    assert sample["prediction_kind"] == "single_sample" and sample["draw_index"] == 0
    target = adapter.native_targets("val", 0)[2]["roman"]
    rescored = evaluator._reconstruction_metrics(mean["pred_normalized"]["roman"] * 10, target)
    assert row["metrics"]["roman"]["rmse"] == rescored["rmse"] == 0.0
    assert row["metrics"]["roman"]["psnr"] == rescored["psnr"]
    assert result["metric_prediction_kind"] == {"rmse": "ensemble_mean", "psnr": "ensemble_mean", "crps": "ensemble"}
    assert arc._resolve_prediction_path(row, "roman", "sample") == Path(row["pred_sample_normalized_paths"]["roman"])
    from vit_lens_classification.manifest import build_prediction_entries
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({"methods": {"synthetic": result}}))
    entries = build_prediction_entries(evaluation, split="train", label=1, method_name="synthetic")
    assert entries[0]["prediction_path"] == row["pred_sample_normalized_paths"]["roman"]
    expected_root = mean_root if mean_root_given else tmp_path / "samples" / "ensemble_mean"
    assert Path(result["pred_normalized_root"]) == expected_root


def test_e5_metric_values_do_not_change_when_enabling_artifact_saving(tmp_path):
    args = ("val", [0], ["lsst", "euclid"], ["roman"], 2, 123, 1, 0)
    score_only = evaluator._evaluate_method(DistinctDrawsAdapter(), *args)
    with_artifacts = evaluator._evaluate_method(DistinctDrawsAdapter(), *args, pred_sample_normalized_root=tmp_path)
    assert score_only["aggregate"] == with_artifacts["aggregate"]
    assert not score_only["pred_normalized_paths"] and not score_only["pred_sample_normalized_paths"]


def test_e5_cli_sample_option_also_saves_mean_with_matching_json_metadata(tmp_path, monkeypatch):
    adapter = DistinctDrawsAdapter()
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(evaluator, "_build_adapter", lambda **kwargs: adapter)
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--known", "lsst,euclid", "--crps-samples", "2", "--output", str(output),
        "--save-pred-sample-normalized", "--pred-sample-normalized-root", str(tmp_path / "samples"),
        "--pred-normalized-root", str(tmp_path / "means"),
    ])
    evaluator.main()
    result = json.loads(output.read_text())
    row = result["methods"]["asyn-diffusion"]["examples"][0]
    assert adapter.draw_calls == 2
    assert Path(row["pred_normalized_path"]).is_file()
    assert Path(row["pred_normalized_path"]).is_relative_to(Path(result["pred_normalized_root"]))
    assert result["metric_prediction_kind"]["rmse"] == "ensemble_mean"
    assert result["downstream_prediction_kind"] == "single_sample"
