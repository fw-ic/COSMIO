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
import torch
import zarr

from helpers import make_test_config, write_manifest_tree, write_prebuilt_artifacts
from asyndiff.data.dataset import NonlensJointDataset, collate_samples
from asyndiff.data.index import artifacts_need_rebuild, build_sample_index, build_visualization_subset, compute_channel_stats
from asyndiff.data.normalization import SIGNED_CAPPED_ASINH_SCHEMA
from asyndiff.data.protocol import fit_tensor_to_size, stack_valid_masks
from asyndiff.training.common import resize_batch_to_native


def test_build_sample_index_and_stats(tmp_path: Path) -> None:
    data_root = write_manifest_tree(tmp_path)
    config = {
        "data": {
            "data_root": str(data_root),
            "target_group": "images",
            "train_count": 2,
            "val_count": 1,
            "test_count": 1,
            "resize": 64,
        },
        "evaluation": {
            "visualization_splits": ["train", "val", "test"],
            "visualization_size": 1,
        },
    }
    manifest = build_sample_index(config)
    assert manifest["summary"] == {"total": 4, "train": 2, "val": 1, "test": 1}
    subset = build_visualization_subset(manifest, "train", 1)
    assert subset["size"] <= 1
    stats = compute_channel_stats(manifest, tmp_path / "stats.json")
    assert set(stats.keys()) == {"lsst", "euclid", "roman"}


def test_rejects_non_images_target_group(tmp_path: Path) -> None:
    data_root = write_manifest_tree(tmp_path)
    config = {
        "data": {
            "data_root": str(data_root),
            "target_group": "full_clean_nopsf",
            "train_count": 2,
            "val_count": 1,
            "test_count": 1,
            "resize": 64,
        },
        "evaluation": {
            "visualization_splits": ["train", "val"],
            "visualization_size": 1,
        },
    }
    try:
        build_sample_index(config)
    except ValueError as exc:
        assert "only supports `data.target_group: images`" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-images target group")


def test_dataset_and_collate(tmp_path: Path) -> None:
    artifacts = write_prebuilt_artifacts(tmp_path)
    dataset = NonlensJointDataset(
        split="train",
        index_path=artifacts["index_path"],
        stats_path=artifacts["stats_path"],
        target_group="images",
        resize=64,
        train=False,
    )
    sample = dataset[0]
    assert "record_id" in sample
    assert sample["inputs"]["lsst"].shape == (5, 64, 64)
    assert sample["inputs"]["euclid"].shape == (1, 64, 64)
    assert sample["inputs"]["roman"].shape == (3, 64, 64)
    assert sample["valid_masks"]["lsst"].shape == (1, 64, 64)
    assert sample["native_targets"]["roman"].shape == (3, 59, 59)

    batch = collate_samples([sample])
    assert batch["inputs"]["roman"].shape == (1, 3, 64, 64)
    assert batch["valid_masks"]["roman"].shape == (1, 1, 64, 64)
    assert batch["native_targets"]["lsst"].shape == (1, 5, 32, 32)


def test_compute_channel_stats_uses_normalization_metadata(tmp_path: Path) -> None:
    data_root = write_manifest_tree(tmp_path, normalized_with_asinh=True)
    config = {
        "data": {
            "data_root": str(data_root),
            "target_group": "images",
            "train_count": 2,
            "val_count": 1,
            "test_count": 1,
            "resize": 64,
        },
        "evaluation": {
            "visualization_splits": ["train", "val", "test"],
            "visualization_size": 1,
        },
    }
    manifest = build_sample_index(config)
    stats = compute_channel_stats(manifest, tmp_path / "stats.json")

    assert stats["schema"] == SIGNED_CAPPED_ASINH_SCHEMA
    assert stats["normalized_data_root"] == str(data_root)
    assert set(stats["modalities"].keys()) == {"lsst", "euclid", "roman"}


def test_dataset_uses_asinh_normalized_inputs_without_re_normalizing(tmp_path: Path) -> None:
    artifacts = write_prebuilt_artifacts(tmp_path, normalized_with_asinh=True)
    dataset = NonlensJointDataset(
        split="train",
        index_path=artifacts["index_path"],
        stats_path=artifacts["stats_path"],
        target_group="images",
        resize=64,
        train=False,
    )

    sample = dataset[0]
    group = zarr.open_group(str(artifacts["index_manifest"]["records"][0]["shard_path"]), mode="r")["images"]
    roman = torch.from_numpy(np.moveaxis(np.asarray(group["roman"][0], dtype=np.float32), -1, 0)).float()
    expected_input, valid_mask = fit_tensor_to_size(roman, (64, 64), use_padding=True)
    expected_target = torch.sinh(roman)

    assert torch.allclose(sample["inputs"]["roman"], expected_input, atol=1.0e-6)
    assert torch.allclose(sample["valid_masks"]["roman"], valid_mask, atol=1.0e-6)
    assert torch.allclose(sample["native_targets"]["roman"], expected_target, atol=1.0e-6)


def test_resize_batch_to_native_crops_center_padded_prediction(tmp_path: Path) -> None:
    artifacts = write_prebuilt_artifacts(tmp_path, normalized_with_asinh=True)
    dataset = NonlensJointDataset(
        split="train",
        index_path=artifacts["index_path"],
        stats_path=artifacts["stats_path"],
        target_group="images",
        resize=64,
        train=False,
    )

    sample = dataset[0]
    prediction = sample["inputs"]["roman"].unsqueeze(0)
    restored = resize_batch_to_native(prediction, sample["native_targets"]["roman"].unsqueeze(0))
    assert torch.allclose(restored[0], sample["inputs"]["roman"][:, 2:61, 2:61], atol=1.0e-6)


def test_stack_valid_masks_repeats_single_channel_masks_per_modality(tmp_path: Path) -> None:
    artifacts = write_prebuilt_artifacts(tmp_path, normalized_with_asinh=True)
    dataset = NonlensJointDataset(
        split="train",
        index_path=artifacts["index_path"],
        stats_path=artifacts["stats_path"],
        target_group="images",
        resize=64,
        train=False,
    )

    sample = dataset[0]
    stacked = stack_valid_masks(sample["valid_masks"])
    assert stacked.shape == (9, 64, 64)
    assert torch.allclose(stacked[:5], sample["valid_masks"]["lsst"].expand(5, -1, -1), atol=1.0e-6)
    assert torch.allclose(stacked[5:6], sample["valid_masks"]["euclid"], atol=1.0e-6)
    assert torch.allclose(stacked[6:9], sample["valid_masks"]["roman"].expand(3, -1, -1), atol=1.0e-6)


def test_artifacts_need_rebuild_when_target_group_changes(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    assert not artifacts_need_rebuild(config)

    config["data"]["target_group"] = "full_clean_nopsf"
    assert artifacts_need_rebuild(config)


def test_artifacts_need_rebuild_when_signed_asinh_stats_are_stale(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, normalized_with_asinh=True)
    assert not artifacts_need_rebuild(config)

    Path(config["data"]["stats_path"]).write_text(
        json.dumps(
            {
                "lsst": {"mean": [0.0] * 5, "std": [1.0] * 5},
                "euclid": {"mean": [0.0], "std": [1.0]},
                "roman": {"mean": [0.0] * 3, "std": [1.0] * 3},
            }
        ),
        encoding="utf-8",
    )

    assert artifacts_need_rebuild(config)


def test_visualization_subset_uses_unique_record_ids(tmp_path: Path) -> None:
    data_root = write_manifest_tree(tmp_path, run_name="run_000", sample_ids=["dup", "dup"])
    write_manifest_tree(tmp_path, run_name="run_001", sample_ids=["dup", "dup"])
    config = {
        "data": {
            "data_root": str(data_root),
            "target_group": "images",
            "train_count": 4,
            "val_count": 0,
            "test_count": 0,
            "resize": 64,
        },
        "evaluation": {
            "visualization_splits": ["train", "val"],
            "visualization_size": 1,
        },
    }
    manifest = build_sample_index(config)
    assert len(manifest["records"]) == 4
    assert len({record["record_id"] for record in manifest["records"]}) == 4

    subset = build_visualization_subset(manifest, "train", 1)
    subset_path = tmp_path / "subset.json"
    subset_path.write_text(json.dumps(subset), encoding="utf-8")
    stats_path = tmp_path / "stats.json"
    compute_channel_stats(manifest, stats_path)
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = NonlensJointDataset(
        split="train",
        index_path=index_path,
        stats_path=stats_path,
        target_group="images",
        resize=64,
        visualization_subset=subset_path,
    )
    assert len(dataset) == 1

