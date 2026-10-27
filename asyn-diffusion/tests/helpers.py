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
from typing import Any

import numpy as np
import torch
import zarr

from asyndiff.data.index import build_record_id, compute_channel_stats


def _signed_asinh_test_metadata(output_root: Path) -> dict[str, Any]:
    def _band_params() -> dict[str, float]:
        return {
            "center": 0.0,
            "sigma": 1.0,
            "cap_low": -1.0,
            "cap_high": 1.0,
            "a_neg": 1.0,
            "a_pos": 1.0,
            "p16": -0.25,
            "p84": 0.25,
            "sample_count": 16,
        }

    return {
        "input_root": str(output_root.parent / "raw"),
        "output_root": str(output_root),
        "processed_shards": 1,
        "method": {
            "name": "signed_capped_asinh",
            "range": [-1.0, 1.0],
        },
        "modalities": {
            "lsst": {band: _band_params() for band in ("g", "r", "i", "z", "y")},
            "euclid": {"VIS": _band_params()},
            "roman": {band: _band_params() for band in ("F106", "F129", "F158")},
        },
    }


def write_manifest_tree(
    tmp_path: Path,
    sample_count: int = 4,
    run_name: str = "run_000",
    sample_ids: list[str] | None = None,
    normalized_with_asinh: bool = False,
) -> Path:
    data_root = tmp_path / "data"
    run_dir = data_root / run_name / "shards" / "nonlens"
    run_dir.mkdir(parents=True)
    shard_path = run_dir / "nonlens_000000.zarr"
    root = zarr.open_group(str(shard_path), mode="w")

    ids = sample_ids or [f"sample_{idx:03d}" for idx in range(sample_count)]
    sample_count = len(ids)

    root.create_dataset("id", data=np.array(ids, dtype="<U16"), dtype="<U16", shape=(sample_count,))
    root.create_dataset("label", data=np.arange(sample_count, dtype=np.int64) % 2, dtype="i8", shape=(sample_count,))
    root.create_dataset("index", data=np.arange(sample_count, dtype=np.int64), dtype="i8", shape=(sample_count,))

    group = root.create_group("images")
    if normalized_with_asinh:
        lsst = torch.linspace(-0.7, 0.7, steps=sample_count * 32 * 32 * 5, dtype=torch.float32).view(
            sample_count, 32, 32, 5
        ).numpy()
        euclid = torch.linspace(-0.6, 0.6, steps=sample_count * 64 * 64, dtype=torch.float32).view(
            sample_count, 64, 64, 1
        ).numpy()
        roman = torch.linspace(-0.5, 0.5, steps=sample_count * 59 * 59 * 3, dtype=torch.float32).view(
            sample_count, 59, 59, 3
        ).numpy()
    else:
        lsst = torch.linspace(0.0, 1.0, steps=sample_count * 32 * 32 * 5, dtype=torch.float32).view(
            sample_count, 32, 32, 5
        ).numpy()
        euclid = torch.linspace(0.2, 1.2, steps=sample_count * 64 * 64, dtype=torch.float32).view(
            sample_count, 64, 64, 1
        ).numpy()
        roman = torch.linspace(-0.5, 0.5, steps=sample_count * 59 * 59 * 3, dtype=torch.float32).view(
            sample_count, 59, 59, 3
        ).numpy()
    group.create_dataset("lsst", data=lsst, chunks=(1, 32, 32, 5), dtype="f4")
    group.create_dataset("euclid", data=euclid, chunks=(1, 64, 64, 1), dtype="f4")
    group.create_dataset("roman", data=roman, chunks=(1, 59, 59, 3), dtype="f4")

    manifest_path = data_root / run_name / "shards" / "manifest_lre_nonlens.json"
    manifest_path.write_text(json.dumps({"results": [{"status": "ok", "path": str(shard_path)}]}), encoding="utf-8")
    if normalized_with_asinh:
        (data_root / "normalization_metadata.json").write_text(
            json.dumps(_signed_asinh_test_metadata(data_root)),
            encoding="utf-8",
        )
    return data_root


def write_prebuilt_artifacts(tmp_path: Path, normalized_with_asinh: bool = False) -> dict[str, Any]:
    data_root = write_manifest_tree(tmp_path, sample_count=4, normalized_with_asinh=normalized_with_asinh)
    shard_path = data_root / "run_000" / "shards" / "nonlens" / "nonlens_000000.zarr"
    splits = ["train", "train", "val", "test"]
    index_manifest = {
        "schema_version": 2,
        "build_settings": {"counts": {"train": 2, "val": 1, "test": 1},
                           "selected_runs": [], "manifest_name": "manifest_lre_nonlens.json"},
        "records": [
            {
                "record_id": build_record_id(data_root, shard_path, idx),
                "sample_id": f"sample_{idx:03d}",
                "label": idx % 2,
                "global_index": idx,
                "local_index": idx,
                "run_id": "run_000",
                "shard_path": str(shard_path),
                "target_group": "images",
                "split": splits[idx],
            }
            for idx in range(4)
        ],
        "summary": {"total": 4, "train": 2, "val": 1, "test": 1},
        "target_group": "images",
        "data_root": str(data_root),
        "resize": 64,
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_manifest), encoding="utf-8")

    stats_path = tmp_path / "stats.json"
    stats = compute_channel_stats(index_manifest, stats_path)

    subset_path = tmp_path / "viz.json"
    subset_path.write_text(
        json.dumps(
            {
                "subsets": {
                    "train": {
                        "split": "train",
                        "record_ids": [build_record_id(data_root, shard_path, 0)],
                        "sample_ids": ["sample_000"],
                        "size": 1,
                    },
                    "val": {
                        "split": "val",
                        "record_ids": [build_record_id(data_root, shard_path, 2)],
                        "sample_ids": ["sample_002"],
                        "size": 1,
                    },
                    "test": {
                        "split": "test",
                        "record_ids": [build_record_id(data_root, shard_path, 3)],
                        "sample_ids": ["sample_003"],
                        "size": 1,
                    },
                },
                "schema_version": 2,
                "size": 1,
            }
        ),
        encoding="utf-8",
    )

    return {
        "data_root": data_root,
        "index_manifest": index_manifest,
        "index_path": index_path,
        "stats_path": stats_path,
        "stats": stats,
        "subset_path": subset_path,
    }


def make_test_config(tmp_path: Path, normalized_with_asinh: bool = False) -> dict[str, Any]:
    artifacts = write_prebuilt_artifacts(tmp_path, normalized_with_asinh=normalized_with_asinh)
    return {
        "experiment": {
            "name": "test",
            "output_root": str(tmp_path / "runs"),
            "seed": 123,
        },
        "data": {
            "data_root": str(artifacts["data_root"]),
            "target_group": "images",
            "resize": 64,
            "num_workers": 0,
            "pin_memory": False,
            "train_count": 2,
            "val_count": 1,
            "test_count": 1,
            "index_path": str(artifacts["index_path"]),
            "stats_path": str(artifacts["stats_path"]),
            "visualization_subset_path": str(artifacts["subset_path"]),
            "augment_flip_prob": 0.0,
            "augment_shift_prob": 0.0,
            "augment_scale_prob": 0.0,
        },
        "model": {
            "head_mode": "multi",
            "in_channels": 9,
            "out_channels": 9,
            "base_channels": 16,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
            "time_dim": 64,
            "num_modalities": 3,
            "num_heads": 4,
        },
        "diffusion": {
            "timesteps": 20,
            "beta_schedule": "linear",
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
            "objective": "eps",
            "clip_x0": True,
        },
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 1.0e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "log_every": 1,
            "val_every": 1,
            "max_train_batches": 1,
        },
        "evaluation": {
            "visualization_splits": ["train", "val", "test"],
            "visualization_split": "train",
            "visualization_size": 1,
            "max_val_batches": 1,
            "visualization_known_modalities": ["lsst", "euclid"],
            "known_combinations": [["lsst", "euclid"], ["lsst"], ["euclid"]],
            "num_crps_samples": 5,
            "inference_steps": 8,
            "fix_mode": "noised",
            "ddim_eta": 1.0,
        },
    }

