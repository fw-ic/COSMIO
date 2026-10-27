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

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vit_lens_classification.data import RomanManifestDataset
from vit_lens_classification.manifest import select_window, split_selection_report, validate_classifier_splits, _resolve_manifest_shard_path
from vit_lens_classification.runtime import checkpoint_metric_key, run_evaluation


@pytest.mark.parametrize("total,start,count", [(0, 0, None), (10, -1, 2), (10, 0, 0), (10, 0, -1), (10, 8, 8), (10, 10, None)])
def test_windows_reject_empty_negative_and_partial_requests(total, start, count):
    with pytest.raises(ValueError, match="available="):
        select_window(list(range(total)), start, count, "test run")


def test_windows_allow_exact_end_and_report_excluded_splits():
    assert select_window(list(range(10)), 8, 2, "run") == [8, 9]
    assert select_window(list(range(10)), 8, None, "run") == [8, 9]
    report = split_selection_report({"records": [{"split": s} for s in ("train", "val", "test", "test")]}, {"train": "train", "val": "eval"})
    assert report["test"] == {"available": 2, "mapped_to": None, "retained": 0, "excluded": 2}
    assert report["val"]["mapped_to"] == "eval"


def test_relative_paths_ignore_cwd_collisions(tmp_path, monkeypatch):
    root, cwd = tmp_path / "root", tmp_path / "cwd"
    root.mkdir(); cwd.mkdir()
    for directory in (root, cwd):
        (directory / "shard.zarr").mkdir()
    monkeypatch.chdir(cwd)
    assert _resolve_manifest_shard_path(root, "shard.zarr") == (root / "shard.zarr").resolve()
    assert _resolve_manifest_shard_path(root, cwd / "shard.zarr") == (cwd / "shard.zarr").resolve()


def test_unused_normalization_metadata_is_not_loaded(tmp_path):
    prediction = tmp_path / "prediction.pt"
    torch.save({"pred_normalized": torch.ones(3, 4, 4)}, prediction)
    manifest = tmp_path / "manifest.json"
    entry = {"split": "eval", "source_kind": "pred_sample_normalized", "prediction_path": str(prediction),
             "label": 1, "record_id": "x", "sample_id": "x"}
    manifest.write_text(json.dumps({"normalization_metadata_path": str(tmp_path / "missing.json"), "entries": [entry]}))
    dataset = RomanManifestDataset(manifest, "eval", 4)
    assert dataset[0]["image"].shape == (3, 4, 4)
    assert dataset.normalization_metadata is None
    with pytest.raises(FileNotFoundError):
        dataset._load_image({"source_kind": "lens_raw"})


class OrderedClasses(Dataset):
    def __init__(self, labels):
        self.entries = [{"label": label, "record_id": f"r{i:03d}"} for i, label in enumerate(labels)]
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, i):
        entry = self.entries[i]
        return {**entry, "sample_id": entry["record_id"], "image": torch.tensor([1.0, 0.0])}


def test_limited_eval_is_stratified_fixed_and_records_actual_ids():
    dataset = OrderedClasses([0] * 18 + [1] * 2)
    loader = DataLoader(dataset, batch_size=4)
    args = dict(model=torch.nn.Identity(), loader=loader, device=torch.device("cpu"), threshold=0.5, max_batches=1)
    metrics, rows = run_evaluation(**args)
    again, rows_again = run_evaluation(**args)
    assert rows == rows_again
    assert {row["label"] for row in rows} == {0, 1} and metrics["count"] == 4
    assert metrics["auroc"] == 0.5
    assert metrics["evaluation_subset"]["record_ids"] == [row["record_id"] for row in rows]
    assert checkpoint_metric_key({"auroc": 0.5, "accuracy": 1.0}) == 0.5


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1]])
def test_eval_rejects_empty_and_single_class_scoring(labels):
    with pytest.raises(ValueError):
        run_evaluation(torch.nn.Identity(), DataLoader(OrderedClasses(labels), batch_size=2), torch.device("cpu"), 0.5)


@pytest.mark.parametrize("auroc", [None, float("nan"), float("inf")])
def test_invalid_auroc_never_falls_back_to_accuracy(auroc):
    with pytest.raises(ValueError, match="accuracy is not a substitute"):
        checkpoint_metric_key({"auroc": auroc, "accuracy": 1.0})


def test_manifest_checks_record_overlap_and_missing_classes():
    entries = [{"record_id": f"{split}_{label}", "split": split, "label": label} for split in ("train", "eval") for label in (0, 1)]
    validate_classifier_splits(entries)
    with pytest.raises(ValueError, match="both labels"):
        validate_classifier_splits(entries[:-1])
    entries[-1]["record_id"] = entries[0]["record_id"]
    with pytest.raises(ValueError, match="both the train and eval"):
        validate_classifier_splits(entries)


def test_imputed_manifest_cli_requires_explicit_output(monkeypatch):
    import sys
    from vit_lens_classification.build_imputed_roman_manifest import parse_args
    args = ["build_manifest", "--train-results-json", "train_results.json"]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, "argv", args + ["--output", "chosen_manifest.json"])
    assert parse_args().output == "chosen_manifest.json"
