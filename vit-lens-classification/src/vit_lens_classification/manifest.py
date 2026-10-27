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
import zarr


def load_json(path: str | Path) -> dict[str, Any]:
    payload_path = Path(path)
    with payload_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_record_id(data_root: Path, shard_path: Path, local_index: int) -> str:
    root = Path(data_root)
    shard = Path(shard_path)
    for candidate_root, candidate_shard in ((root, shard), (root.resolve(), shard.resolve())):
        try:
            return f"{candidate_shard.relative_to(candidate_root).as_posix()}:{local_index}"
        except ValueError:
            continue
    run_parts = list(shard.parts)
    for offset, part in enumerate(run_parts):
        if part.startswith("run_"):
            return f"{Path(*run_parts[offset:]).as_posix()}:{local_index}"
    return f"{shard.name}:{local_index}"


def _resolve_manifest_shard_path(root: Path, shard_path: str | Path) -> Path:
    path = Path(shard_path)
    if path.is_absolute():
        return path.resolve()
    candidates = [root / path]
    parts = list(path.parts)
    for offset, part in enumerate(parts):
        if part.startswith("run_"):
            candidates.append(root / Path(*parts[offset:]))
            break
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def build_nonlens_entries(nonlens_index_path: str | Path, reports: list | None = None) -> list[dict[str, Any]]:
    index_manifest = load_json(nonlens_index_path)
    if reports is not None:
        reports.append(split_selection_report(index_manifest, {"train": "train", "val": "eval"}))
    entries: list[dict[str, Any]] = []
    for record in index_manifest.get("records", []):
        split = str(record.get("split", ""))
        if split not in {"train", "val"}:
            continue
        entries.append(
            {
                "split": "train" if split == "train" else "eval",
                "label": 0,
                "source_kind": "nonlens_normalized",
                "record_id": str(record["record_id"]),
                "sample_id": str(record["sample_id"]),
                "run_id": str(record.get("run_id", "")),
                "shard_path": str(record["shard_path"]),
                "local_index": int(record["local_index"]),
                "target_group": str(record.get("target_group", "images")),
            }
        )
    return entries


def build_nonlens_entries_for_splits(
    nonlens_index_path: str | Path,
    split_map: dict[str, str],
    reports: list | None = None,
) -> list[dict[str, Any]]:
    index_manifest = load_json(nonlens_index_path)
    if reports is not None:
        reports.append(split_selection_report(index_manifest, split_map))
    entries: list[dict[str, Any]] = []
    for record in index_manifest.get("records", []):
        source_split = str(record.get("split", ""))
        output_split = split_map.get(source_split)
        if output_split is None:
            continue
        entries.append(
            {
                "split": str(output_split),
                "label": 0,
                "source_kind": "nonlens_normalized",
                "record_id": str(record["record_id"]),
                "sample_id": str(record["sample_id"]),
                "run_id": str(record.get("run_id", "")),
                "shard_path": str(record["shard_path"]),
                "local_index": int(record["local_index"]),
                "target_group": str(record.get("target_group", "images")),
            }
        )
    return entries


def _iter_lens_manifest_shards(lens_root: Path, run_name: str) -> list[Path]:
    manifest_path = lens_root / run_name / "shards" / "manifest_lens.json"
    manifest = load_json(manifest_path)
    shard_paths = [
        _resolve_manifest_shard_path(lens_root, item["path"])
        for item in manifest.get("results", [])
        if item.get("status") == "ok"
    ]
    if not shard_paths:
        raise FileNotFoundError(f"No lens shards found for {run_name} in {manifest_path}.")
    return sorted(shard_paths)


def build_lens_entries(
    lens_root: str | Path,
    run_name: str,
    split: str,
    target_group: str = "images",
    source_kind: str = "lens_normalized",
) -> list[dict[str, Any]]:
    root = Path(lens_root)
    entries: list[dict[str, Any]] = []
    for shard_path in _iter_lens_manifest_shards(root, run_name):
        group = zarr.open_group(str(shard_path), mode="r")
        if target_group not in group:
            raise KeyError(f"Missing target group `{target_group}` in {shard_path}.")
        sample_ids = np.asarray(group["id"][:]).astype(str)
        for local_index, sample_id in enumerate(sample_ids.tolist()):
            entries.append(
                {
                    "split": str(split),
                    "label": 1,
                    "source_kind": str(source_kind),
                    "record_id": build_record_id(root, shard_path, local_index),
                    "sample_id": str(sample_id),
                    "run_id": str(run_name),
                    "shard_path": str(shard_path),
                    "local_index": int(local_index),
                    "target_group": str(target_group),
                }
            )
    return entries


def build_nonlens_run_entries(
    nonlens_root: str | Path,
    run_names: list[str] | tuple[str, ...],
    split: str,
    target_group: str = "images",
    start: int = 0,
    count: int | None = None,
) -> list[dict[str, Any]]:
    root = Path(nonlens_root)
    entries: list[dict[str, Any]] = []
    for run_name in [str(item) for item in run_names]:
        manifest_path = root / run_name / "shards" / "manifest_lre_nonlens.json"
        manifest = load_json(manifest_path)
        shard_paths = [
            _resolve_manifest_shard_path(root, item["path"])
            for item in manifest.get("results", [])
            if item.get("status") == "ok"
        ]
        if not shard_paths:
            raise FileNotFoundError(f"No nonlens shards found for {run_name} in {manifest_path}.")
        for shard_path in sorted(shard_paths):
            group = zarr.open_group(str(shard_path), mode="r")
            if target_group not in group:
                raise KeyError(f"Missing target group `{target_group}` in {shard_path}.")
            sample_ids = np.asarray(group["id"][:]).astype(str)
            for local_index, sample_id in enumerate(sample_ids.tolist()):
                entries.append(
                    {
                        "split": str(split),
                        "label": 0,
                        "source_kind": "nonlens_normalized",
                        "record_id": build_record_id(root, shard_path, local_index),
                        "sample_id": str(sample_id),
                        "run_id": str(run_name),
                        "shard_path": str(shard_path),
                        "local_index": int(local_index),
                        "target_group": str(target_group),
                    }
                )
    return select_window(entries, start, count, f"nonlens runs {list(run_names)}")


def select_window(entries: list, start: int, count: int | None, context: str) -> list:
    start = int(start)
    count = len(entries) - start if count is None else int(count)
    if start < 0 or count <= 0 or start >= len(entries) or start + count > len(entries):
        raise ValueError(
            f"{context}: requested window start={start}, count={count}, end={start + count}; "
            f"available={len(entries)}. Require start >= 0, count > 0 and end <= available."
        )
    return entries[start:start + count]


def split_selection_report(index_manifest: dict[str, Any], split_map: dict[str, str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in index_manifest.get("records", []):
        split = str(record.get("split", ""))
        counts[split] = counts.get(split, 0) + 1
    return {split: {"available": count, "mapped_to": split_map.get(split),
                    "retained": count if split in split_map else 0,
                    "excluded": 0 if split in split_map else count}
            for split, count in sorted(counts.items())}


def validate_classifier_splits(entries: list[dict[str, Any]]) -> None:
    for split in ("train", "eval"):
        labels = {int(entry["label"]) for entry in entries if entry.get("split") == split}
        if labels != {0, 1}:
            raise ValueError(f"Classifier {split} split requires both labels 0 and 1; got {sorted(labels)}.")
    train_ids = {entry["record_id"] for entry in entries if entry.get("split") == "train"}
    eval_ids = {entry["record_id"] for entry in entries if entry.get("split") == "eval"}
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(f"{len(overlap)} record_id(s) appear in both the train and eval splits: {sorted(overlap)[:3]}")


def _resolve_method(results: dict[str, Any], method_name: str | None) -> str:
    methods = results.get("methods", {})
    if method_name is not None:
        if method_name not in methods:
            raise KeyError(f"Method `{method_name}` not present in evaluation results. Available: {sorted(methods)}")
        return method_name
    if len(methods) != 1:
        raise ValueError(f"Expected exactly one method in evaluation results, found {sorted(methods)}. Pass --method.")
    return next(iter(methods))


def build_prediction_entries(
    evaluation_results_path: str | Path,
    split: str,
    label: int,
    method_name: str | None = None,
    target_modality: str = "roman",
    start: int = 0,
    count: int | None = None,
) -> list[dict[str, Any]]:
    results = load_json(evaluation_results_path)
    resolved_method = _resolve_method(results, method_name)
    method_results = results["methods"][resolved_method]
    entries: list[dict[str, Any]] = []
    examples = list(method_results.get("examples", []))
    selected_examples = select_window(examples, start, count, f"prediction JSON {evaluation_results_path}")
    for example in selected_examples:
        pred_paths = example.get("pred_sample_normalized_paths", {})
        if target_modality not in pred_paths:
            raise KeyError(
                f"Missing `{target_modality}` saved sample path in {evaluation_results_path} for "
                f"record_id={example.get('record_id')}. Re-run evaluator with --save-pred-sample-normalized."
            )
        entries.append(
            {
                "split": str(split),
                "label": int(label),
                "source_kind": "pred_sample_normalized",
                "record_id": str(example["record_id"]),
                "sample_id": str(example["sample_id"]),
                "prediction_path": str(pred_paths[target_modality]),
                "method": str(resolved_method),
                "target_modality": str(target_modality),
            }
        )
    return entries


def build_imputed_entries(
    evaluation_results_path: str | Path,
    split: str,
    method_name: str | None = None,
    target_modality: str = "roman",
    start: int = 0,
    count: int | None = None,
) -> list[dict[str, Any]]:
    return build_prediction_entries(
        evaluation_results_path=evaluation_results_path,
        split=split,
        label=1,
        method_name=method_name,
        target_modality=target_modality,
        start=start,
        count=count,
    )


def summarize_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_label: dict[str, int] = {"0": 0, "1": 0}
    by_source_kind: dict[str, int] = {}
    for entry in entries:
        split = str(entry["split"])
        by_split[split] = by_split.get(split, 0) + 1
        label_key = str(int(entry["label"]))
        by_label[label_key] = by_label.get(label_key, 0) + 1
        source_kind = str(entry["source_kind"])
        by_source_kind[source_kind] = by_source_kind.get(source_kind, 0) + 1
    return {
        "total": len(entries),
        "by_split": by_split,
        "by_label": by_label,
        "by_source_kind": by_source_kind,
    }


def build_manifest_payload(
    track: str,
    normalization_metadata_path: str | Path,
    entries: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "track": str(track),
        "labels": {"nonlens": 0, "lens": 1},
        "normalization_metadata_path": str(normalization_metadata_path),
        "summary": summarize_entries(entries),
        "entries": entries,
    }
    if extra:
        payload.update(extra)
    return payload

