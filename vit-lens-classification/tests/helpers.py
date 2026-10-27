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


def band_params() -> dict[str, float]:
    return {
        "center": 0.0,
        "sigma": 1.0,
        "cap_low": -1.0,
        "cap_high": 1.0,
        "a_neg": 1.0,
        "a_pos": 1.0,
        "p16": -0.2,
        "p84": 0.2,
        "sample_count": 16,
    }


def write_normalization_metadata(path: Path, *, input_root: Path | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_root": str(input_root or path.parent / "raw"),
        "output_root": str(path.parent),
        "processed_shards": 1,
        "method": {"name": "signed_capped_asinh", "range": [-1.0, 1.0]},
        "modalities": {
            "lsst": {band: band_params() for band in ("g", "r", "i", "z", "y")},
            "euclid": {"VIS": band_params()},
            "roman": {band: band_params() for band in ("F106", "F129", "F158")},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_shard(shard_path: Path, sample_ids: list[str], roman_offset: float, normalized: bool) -> Path:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = len(sample_ids)
    root = zarr.open_group(str(shard_path), mode="w")
    root.create_dataset("id", data=np.asarray(sample_ids, dtype="<U32"), shape=(sample_count,), dtype="<U32")
    root.create_dataset("label", data=np.zeros(sample_count, dtype=np.uint8), shape=(sample_count,), dtype="u1")
    root.create_dataset("index", data=np.arange(sample_count, dtype=np.int64), shape=(sample_count,), dtype="i8")
    images = root.create_group("images")

    lsst = np.linspace(0.0, 1.0, sample_count * 32 * 32 * 5, dtype=np.float32).reshape(sample_count, 32, 32, 5)
    euclid = np.linspace(0.1, 1.1, sample_count * 64 * 64, dtype=np.float32).reshape(sample_count, 64, 64, 1)
    roman = np.linspace(roman_offset, roman_offset + 0.5, sample_count * 59 * 59 * 3, dtype=np.float32).reshape(
        sample_count, 59, 59, 3
    )
    if normalized:
        roman = np.clip(roman, -1.0, 1.0)
        lsst = np.clip(lsst, -1.0, 1.0)
        euclid = np.clip(euclid, -1.0, 1.0)

    images.create_dataset("lsst", data=lsst, chunks=(1, 32, 32, 5), dtype="f4")
    images.create_dataset("euclid", data=euclid, chunks=(1, 64, 64, 1), dtype="f4")
    images.create_dataset("roman", data=roman, chunks=(1, 59, 59, 3), dtype="f4")
    return shard_path


def write_nonlens_fixture(tmp_path: Path) -> tuple[Path, Path]:
    nonlens_root = tmp_path / "nonlens_norm"
    shard_path = nonlens_root / "run_000" / "shards" / "nonlens" / "nonlens_000000.zarr"
    _write_shard(shard_path, ["nonlens_train_0", "nonlens_train_1", "nonlens_eval_0", "nonlens_eval_1"], 0.0, True)
    index_payload = {
        "records": [
            {
                "record_id": f"run_000/shards/nonlens/nonlens_000000.zarr:{idx}",
                "sample_id": sample_id,
                "label": 0,
                "global_index": idx,
                "local_index": idx,
                "run_id": "run_000",
                "shard_path": str(shard_path),
                "target_group": "images",
                "split": "train" if idx < 2 else "val",
            }
            for idx, sample_id in enumerate(["nonlens_train_0", "nonlens_train_1", "nonlens_eval_0", "nonlens_eval_1"])
        ],
        "summary": {"total": 4, "train": 2, "val": 2, "test": 0},
        "target_group": "images",
        "data_root": str(nonlens_root),
        "resize": 64,
    }
    index_path = tmp_path / "nonlens_index.json"
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")
    manifest_path = nonlens_root / "run_000" / "shards" / "manifest_lre_nonlens.json"
    manifest_path.write_text(
        json.dumps({"results": [{"status": "ok", "path": str(shard_path)}], "summary": {"ok": 1}}),
        encoding="utf-8",
    )
    metadata_path = write_normalization_metadata(nonlens_root / "normalization_metadata.json", input_root=tmp_path / "lens_raw")
    return index_path, metadata_path


def write_lens_fixture(tmp_path: Path, *, normalized: bool = True) -> Path:
    lens_root = tmp_path / ("lens_norm" if normalized else "lens_raw")
    for run_name, count, offset in (("run_000", 3, -0.3), ("run_001", 2, -0.1)):
        shard_path = lens_root / run_name / "shards" / "lens" / f"lens_{run_name[-3:]}000.zarr"
        sample_ids = [f"{run_name}_sample_{idx}" for idx in range(count)]
        _write_shard(shard_path, sample_ids, offset, normalized)
        manifest_path = lens_root / run_name / "shards" / "manifest_lens.json"
        manifest_path.write_text(
            json.dumps({"results": [{"status": "ok", "path": str(shard_path)}], "summary": {"ok": 1}}),
            encoding="utf-8",
        )
    return lens_root


def write_saved_prediction(path: Path, sample_index: int, record_id: str, sample_id: str, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "asyn-diffusion",
        "split": "lens",
        "sample_index": sample_index,
        "record_id": record_id,
        "sample_id": sample_id,
        "known_modalities": ["lsst", "euclid"],
        "target_modality": "roman",
        "seed": 1234,
        "pred_normalized": torch.full((3, 59, 59), float(value), dtype=torch.float32),
    }
    torch.save(payload, path)
    return path


def write_imputation_results_json(path: Path, prediction_paths: list[Path], split: str) -> Path:
    examples = []
    for idx, pred_path in enumerate(prediction_paths):
        examples.append(
            {
                "sample_index": idx,
                "record_id": f"{split}_record_{idx}",
                "sample_id": f"{split}_sample_{idx}",
                "metrics": {"roman": {"rmse": 0.1}},
                "pred_sample_normalized_paths": {"roman": str(pred_path)},
            }
        )
    payload = {
        "split": "lens",
        "input_modalities": ["lsst", "euclid"],
        "target_modalities": ["roman"],
        "methods": {
            "asyn-diffusion": {
                "method": "asyn-diffusion",
                "input_modalities": ["lsst", "euclid"],
                "target_modalities": ["roman"],
                "examples": examples,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_train_config(tmp_path: Path, manifest_path: Path, name: str) -> dict[str, Any]:
    return {
        "experiment": {
            "name": name,
            "output_root": str(tmp_path / "runs"),
        },
        "data": {
            "manifest_path": str(manifest_path),
            "image_size": 32,
            "num_workers": 0,
        },
        "model": {
            "in_channels": 3,
            "image_size": 32,
            "patch_size": 8,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 64,
            "mlp_dim": 128,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        "train": {
            "seed": 123,
            "batch_size": 2,
            "epochs": 1,
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
            "balanced_sampling": True,
            "max_train_steps": 1,
            "max_eval_batches": 1,
            "checkpoint_every_epochs": 1,
            "device": "cpu",
        },
        "evaluation": {
            "split": "eval",
            "threshold": 0.5,
        },
    }

