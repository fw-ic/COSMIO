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
import torch.nn.functional as F
import zarr
from asyndiff.config import ensure_parent, load_json
from asyndiff.constants import MODALITY_CHANNELS, MODALITY_ORDER
from asyndiff.data.normalization import (
    SIGNED_CAPPED_ASINH_SCHEMA,
    load_normalization_metadata,
    stats_from_normalization_metadata,
)
_ALLOWED_TARGET_GROUP = "images"
def _relative_shard_path(data_root: Path, shard_path: Path) -> Path:
    try:
        return shard_path.relative_to(data_root)
    except ValueError:
        pass
    parts = shard_path.parts
    try:
        shards_index = parts.index("shards")
    except ValueError as exc:
        raise ValueError(
            f"{shard_path!r} is not under {data_root!r} and does not contain a recoverable run-relative shards path."
        ) from exc
    if shards_index <= 0:
        raise ValueError(
            f"{shard_path!r} is not under {data_root!r} and is missing the run id segment before `shards`."
        )
    run_segment = parts[shards_index - 1]
    if not str(run_segment).startswith("run_"):
        raise ValueError(
            f"{shard_path!r} is not under {data_root!r} and its inferred run segment {run_segment!r} is invalid."
        )
    return Path(run_segment, *parts[shards_index:])
def build_record_id(data_root: Path, shard_path: Path, local_index: int) -> str:
    relative_shard = _relative_shard_path(data_root, shard_path)
    return f"{relative_shard.as_posix()}:{local_index}"
def _resolve_manifest_shard_path(data_root: Path, shard_path: str | Path) -> Path:
    path = Path(shard_path)
    if path.is_absolute():
        return path.resolve()
    candidates = [data_root / path]
    for offset, part in enumerate(path.parts):
        if str(part).startswith("run_"):
            candidates.append(data_root / Path(*path.parts[offset:]))
            break
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()
def backfill_record_ids(index_manifest: dict[str, Any]) -> dict[str, Any]:
    data_root_value = index_manifest.get("data_root")
    if data_root_value is None:
        return index_manifest
    data_root = Path(data_root_value)
    changed = False
    for record in index_manifest.get("records", []):
        if "record_id" not in record:
            record["record_id"] = build_record_id(data_root, Path(record["shard_path"]), int(record["local_index"]))
            changed = True
    if changed:
        index_manifest["records"].sort(key=_record_sort_key)
    return index_manifest
def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record["sample_id"]),
        str(record.get("record_id", "")),
        int(record["global_index"]),
    )
def _apply_configured_splits(records: list[dict[str, Any]], data_cfg: dict[str, Any]) -> None:
    train_count = data_cfg.get("train_count")
    val_count = data_cfg.get("val_count")
    test_count = data_cfg.get("test_count")
    if train_count is None and val_count is None and test_count is None:
        raise ValueError("data.train_count, data.val_count, or data.test_count must be set.")
    total = len(records)
    requested = {
        "train_count": int(train_count or 0),
        "val_count": int(val_count or 0),
        "test_count": int(test_count or 0),
    }
    train_count = max(0, min(int(train_count or 0), total))
    test_count = max(0, min(int(test_count or 0), max(0, total - train_count)))
    val_count = max(0, min(int(val_count or 0), max(0, total - train_count - test_count)))
    # train_count is clamped to the record total before val_count is derived, so a
    # dataset smaller than train_count silently leaves val (and test) empty. That
    # produces all-NaN validation metrics and no best checkpoint, with no error.
    for name, resolved in (("train_count", train_count), ("val_count", val_count), ("test_count", test_count)):
        if requested[name] > 0 and resolved == 0:
            raise ValueError(
                f"data.{name}={requested[name]} leaves the {name.split('_')[0]} split empty: "
                f"only {total} records are available and the configured counts are {requested}. "
                "Lower the counts to fit the dataset."
            )
    val_start = max(train_count, total - val_count - test_count)
    val_end = total - test_count
    for index, record in enumerate(records):
        if index < train_count:
            record["split"] = "train"
        elif val_start <= index < val_end:
            record["split"] = "val"
        else:
            record["split"] = "test"
def _load_group_array(group: zarr.Group, name: str, local_index: int) -> np.ndarray:
    array = np.asarray(group[name][local_index], dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    return np.moveaxis(array, -1, 0)
def _resize_channels(channels_first: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(channels_first).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy()
def _validate_target_group(target_group: str) -> None:
    if target_group != _ALLOWED_TARGET_GROUP:
        raise ValueError(
            f"This baseline only supports `data.target_group: {_ALLOWED_TARGET_GROUP}`; got `{target_group}`."
        )
def _requested_build_settings(data_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "counts": {split: int(data_cfg.get(f"{split}_count") or 0) for split in ("train", "val", "test")},
        "selected_runs": sorted({str(run).strip() for run in data_cfg.get("selected_runs", []) if str(run).strip()}),
        "manifest_name": str(data_cfg.get("manifest_name", "manifest_lre_nonlens.json")),
    }


def build_sample_index(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    data_root = Path(data_cfg["data_root"])
    manifest_name = str(data_cfg.get("manifest_name", "manifest_lre_nonlens.json"))
    manifests = sorted(data_root.glob(f"run_*/shards/{manifest_name}"))
    selected_runs = {str(run).strip() for run in data_cfg.get("selected_runs", []) if str(run).strip()}
    if selected_runs:
        manifests = [manifest_path for manifest_path in manifests if manifest_path.parent.parent.name in selected_runs]
    if not manifests:
        detail = f" matching runs={sorted(selected_runs)}" if selected_runs else ""
        raise FileNotFoundError(f"No manifests named {manifest_name!r} found under {data_root}{detail}")
    target_group = str(data_cfg["target_group"])
    _validate_target_group(target_group)
    records: list[dict[str, Any]] = []
    for manifest_path in manifests:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for result in manifest.get("results", []):
            if result.get("status") != "ok":
                continue
            shard_path = _resolve_manifest_shard_path(data_root, result["path"])
            root = zarr.open_group(str(shard_path), mode="r")
            if target_group not in root:
                continue
            ids = np.asarray(root["id"][:]).astype(str)
            labels = np.asarray(root["label"][:]).astype(np.int64)
            indices = np.asarray(root["index"][:]).astype(np.int64)
            run_id = shard_path.parents[2].name
            for local_index, sample_id in enumerate(ids.tolist()):
                records.append(
                    {
                        "record_id": build_record_id(data_root, shard_path, local_index),
                        "sample_id": sample_id,
                        "label": int(labels[local_index]),
                        "global_index": int(indices[local_index]),
                        "local_index": local_index,
                        "run_id": run_id,
                        "shard_path": str(shard_path),
                        "target_group": target_group,
                    }
                )
    records.sort(key=_record_sort_key)
    _apply_configured_splits(records, data_cfg)
    summary = {
        "total": len(records),
        "train": sum(1 for item in records if item["split"] == "train"),
        "val": sum(1 for item in records if item["split"] == "val"),
        "test": sum(1 for item in records if item["split"] == "test"),
    }
    return {
        "schema_version": 2,
        "build_settings": _requested_build_settings(data_cfg),
        "records": records,
        "summary": summary,
        "target_group": target_group,
        "data_root": str(data_root),
        "resize": int(data_cfg["resize"]),
    }
def build_visualization_subset(index_manifest: dict[str, Any], split: str, size: int) -> dict[str, Any]:
    index_manifest = backfill_record_ids(index_manifest)
    split_records = [item for item in index_manifest["records"] if item["split"] == split]
    subset = split_records[:size]
    return {
        "split": split,
        "record_ids": [item["record_id"] for item in subset],
        "sample_ids": [item["sample_id"] for item in subset],
        "size": len(subset),
    }
def build_visualization_subsets(index_manifest: dict[str, Any], splits: list[str], size: int) -> dict[str, Any]:
    return {
        "subsets": {split: build_visualization_subset(index_manifest, split=split, size=size) for split in splits},
        "schema_version": 2,
        "size": int(size),
    }
def compute_channel_stats(index_manifest: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    metadata = load_normalization_metadata(index_manifest["data_root"])
    if metadata is not None:
        stats = stats_from_normalization_metadata(metadata, index_manifest["data_root"])
        path = ensure_parent(output_path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        return stats
    resize = int(index_manifest["resize"])
    train_records = [item for item in index_manifest["records"] if item["split"] == "train"]
    sums = {name: np.zeros((MODALITY_CHANNELS[name],), dtype=np.float64) for name in MODALITY_ORDER}
    sq_sums = {name: np.zeros((MODALITY_CHANNELS[name],), dtype=np.float64) for name in MODALITY_ORDER}
    counts = {name: 0 for name in MODALITY_ORDER}
    current_shard_path: str | None = None
    current_group: zarr.Group | None = None
    for record in train_records:
        shard_path = record["shard_path"]
        if shard_path != current_shard_path:
            current_shard_path = shard_path
            current_group = zarr.open_group(shard_path, mode="r")[record["target_group"]]
        assert current_group is not None
        local_index = int(record["local_index"])
        for name in MODALITY_ORDER:
            array = _load_group_array(current_group, name, local_index)
            resized = _resize_channels(array, (resize, resize))
            sums[name] += resized.sum(axis=(1, 2), dtype=np.float64)
            sq_sums[name] += np.square(resized, dtype=np.float64).sum(axis=(1, 2), dtype=np.float64)
            counts[name] += int(resized.shape[1] * resized.shape[2])
    stats: dict[str, Any] = {}
    for name in MODALITY_ORDER:
        mean = sums[name] / max(1, counts[name])
        variance = np.maximum(sq_sums[name] / max(1, counts[name]) - np.square(mean), 1.0e-12)
        stats[name] = {
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
        }
    path = ensure_parent(output_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats
def build_and_store_artifacts(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    index_manifest = build_sample_index(config)
    index_path = ensure_parent(data_cfg["index_path"])
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_manifest, handle, indent=2)
    stats = compute_channel_stats(index_manifest, data_cfg["stats_path"])
    subset = build_visualization_subsets(
        index_manifest,
        splits=[
            str(split)
            for split in config["evaluation"].get(
                "visualization_splits", [config["evaluation"].get("visualization_split", "train")]
            )
        ],
        size=int(config["evaluation"]["visualization_size"]),
    )
    subset_path = ensure_parent(data_cfg["visualization_subset_path"])
    with subset_path.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle, indent=2)
    print(
        json.dumps(
            {
                "index_path": str(index_path),
                "stats_path": str(data_cfg["stats_path"]),
                "visualization_subset_path": str(subset_path),
                "summary": index_manifest["summary"],
                "modalities": stats,
            },
            indent=2,
        )
    )
def _index_schema_valid(index_manifest: dict[str, Any]) -> bool:
    required_top = {"records", "summary", "target_group", "data_root", "resize"}
    if not required_top.issubset(index_manifest.keys()):
        return False
    records = index_manifest.get("records", [])
    if not isinstance(records, list):
        return False
    if not records:
        return True
    required_record = {
        "record_id",
        "sample_id",
        "global_index",
        "local_index",
        "shard_path",
        "split",
        "target_group",
    }
    return required_record.issubset(records[0].keys())
def _subset_schema_valid(subset_manifest: dict[str, Any], expected_splits: list[str]) -> bool:
    if int(subset_manifest.get("schema_version", -1)) < 2:
        return False
    if "subsets" not in subset_manifest:
        return False
    if sorted(subset_manifest["subsets"].keys()) != sorted(expected_splits):
        return False
    for split in expected_splits:
        payload = subset_manifest["subsets"].get(split, {})
        if "record_ids" not in payload:
            return False
    return True
def artifacts_need_rebuild(config: dict[str, Any]) -> bool:
    data_cfg = config["data"]
    index_path = Path(data_cfg["index_path"])
    stats_path = Path(data_cfg["stats_path"])
    subset_path = Path(data_cfg["visualization_subset_path"])
    if not index_path.exists() or not stats_path.exists() or not subset_path.exists():
        return True
    try:
        index_manifest = backfill_record_ids(load_json(index_path))
        stats_manifest = load_json(stats_path)
        subset_manifest = load_json(subset_path)
    except (OSError, json.JSONDecodeError):
        return True
    if not _index_schema_valid(index_manifest):
        return True
    target_group = str(data_cfg["target_group"])
    if target_group != _ALLOWED_TARGET_GROUP:
        return True
    if str(index_manifest.get("target_group")) != target_group:
        return True
    if str(index_manifest.get("data_root")) != str(data_cfg["data_root"]):
        return True
    if int(index_manifest.get("resize", -1)) != int(data_cfg["resize"]):
        return True
    expected_splits = [
        str(split)
        for split in config["evaluation"].get("visualization_splits", [config["evaluation"].get("visualization_split", "train")])
    ]
    if not _subset_schema_valid(subset_manifest, expected_splits):
        return True
    if int(subset_manifest.get("size", -1)) != int(config["evaluation"]["visualization_size"]):
        return True
    if index_manifest.get("schema_version") != 2 or index_manifest.get("build_settings") != _requested_build_settings(data_cfg):
        return True
    normalization_metadata = load_normalization_metadata(data_cfg["data_root"])
    if normalization_metadata is not None:
        if str(stats_manifest.get("schema")) != SIGNED_CAPPED_ASINH_SCHEMA:
            return True
        if str(stats_manifest.get("normalized_data_root")) != str(data_cfg["data_root"]):
            return True
        if "modalities" not in stats_manifest:
            return True
    elif str(stats_manifest.get("schema", "")) == SIGNED_CAPPED_ASINH_SCHEMA:
        return True
    return False
