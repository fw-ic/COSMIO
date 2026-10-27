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

import argparse
import json

from vit_lens_classification.build_imputed_roman_manifest import build_manifest as build_imputed_manifest
from vit_lens_classification.build_real_roman_manifest import build_manifest as build_real_manifest

from .helpers import (
    write_imputation_results_json,
    write_lens_fixture,
    write_nonlens_fixture,
    _write_shard,
    write_saved_prediction,
)


def test_build_real_manifest(tmp_path) -> None:
    nonlens_index_path, metadata_path = write_nonlens_fixture(tmp_path)
    nonlens_root = metadata_path.parent
    lens_root = write_lens_fixture(tmp_path, normalized=True)
    output_path = build_real_manifest(
        argparse.Namespace(
            nonlens_index_path=str(nonlens_index_path),
            nonlens_source="run",
            nonlens_root=str(nonlens_root),
            nonlens_run="run_000",
            train_negative_start=0,
            train_negative_count=2,
            eval_negative_start=2,
            eval_negative_count=2,
            lens_root=str(lens_root),
            normalization_metadata=str(metadata_path),
            train_run="run_000",
            eval_run="run_001",
            target_group="images",
            output=str(tmp_path / "real_manifest.json"),
        )
    )
    payload = json.loads(output_path.read_text())
    assert payload["track"] == "real_roman"
    assert payload["summary"]["by_split"]["train"] == 5
    assert payload["summary"]["by_split"]["eval"] == 4
    assert payload["summary"]["by_source_kind"]["lens_normalized"] == 5
    assert payload["summary"]["by_source_kind"]["nonlens_normalized"] == 4


def test_build_imputed_manifest(tmp_path) -> None:
    nonlens_index_path, metadata_path = write_nonlens_fixture(tmp_path)
    lens_root = write_lens_fixture(tmp_path, normalized=True)
    train_pred_paths = [
        write_saved_prediction(tmp_path / "preds" / "train_0.pt", 0, "train_record_0", "train_sample_0", 0.25),
        write_saved_prediction(tmp_path / "preds" / "train_1.pt", 1, "train_record_1", "train_sample_1", 0.5),
    ]
    train_results = write_imputation_results_json(tmp_path / "train_results.json", train_pred_paths, split="train")
    output_path = build_imputed_manifest(
        argparse.Namespace(
            nonlens_index_path=str(nonlens_index_path),
            train_negative_source="real",
            train_negative_results_json=None,
            train_results_json=str(train_results),
            eval_results_json=None,
            normalization_metadata=str(metadata_path),
            eval_source="real",
            eval_lens_root=str(lens_root),
            eval_run="run_001",
            method=None,
            target_modality="roman",
            output=str(tmp_path / "imputed_manifest.json"),
        )
    )
    payload = json.loads(output_path.read_text())
    assert payload["track"] == "imputed_roman"
    assert payload["summary"]["by_split"]["train"] == 4
    assert payload["summary"]["by_split"]["eval"] == 4
    assert payload["summary"]["by_source_kind"]["pred_sample_normalized"] == 2
    assert payload["summary"]["by_source_kind"]["lens_normalized"] == 2


def test_build_imputed_manifest_with_imputed_train_negatives(tmp_path) -> None:
    nonlens_index_path, metadata_path = write_nonlens_fixture(tmp_path)
    lens_root = write_lens_fixture(tmp_path, normalized=True)
    train_pos_pred_paths = [
        write_saved_prediction(tmp_path / "preds" / "lens_train_0.pt", 0, "lens_train_record_0", "lens_train_sample_0", 0.25),
        write_saved_prediction(tmp_path / "preds" / "lens_train_1.pt", 1, "lens_train_record_1", "lens_train_sample_1", 0.5),
    ]
    train_neg_pred_paths = [
        write_saved_prediction(tmp_path / "preds" / "nonlens_train_0.pt", 0, "nonlens_train_record_0", "nonlens_train_sample_0", -0.2),
        write_saved_prediction(tmp_path / "preds" / "nonlens_train_1.pt", 1, "nonlens_train_record_1", "nonlens_train_sample_1", -0.1),
    ]
    train_results = write_imputation_results_json(tmp_path / "train_results.json", train_pos_pred_paths, split="train")
    train_negative_results = write_imputation_results_json(
        tmp_path / "train_negative_results.json",
        train_neg_pred_paths,
        split="train",
    )
    output_path = build_imputed_manifest(
        argparse.Namespace(
            nonlens_index_path=str(nonlens_index_path),
            train_negative_source="imputed",
            train_negative_results_json=str(train_negative_results),
            train_results_json=str(train_results),
            eval_results_json=None,
            normalization_metadata=str(metadata_path),
            eval_source="real",
            eval_lens_root=str(lens_root),
            eval_run="run_001",
            method=None,
            target_modality="roman",
            output=str(tmp_path / "imputed_negative_manifest.json"),
        )
    )
    payload = json.loads(output_path.read_text())
    assert payload["track"] == "imputed_roman"
    assert payload["summary"]["by_split"]["train"] == 4
    assert payload["summary"]["by_split"]["eval"] == 4
    assert payload["summary"]["by_source_kind"]["pred_sample_normalized"] == 4
    assert payload["summary"]["by_source_kind"]["lens_normalized"] == 2
    assert payload["summary"]["by_source_kind"]["nonlens_normalized"] == 2


def test_build_imputed_manifest_with_imputed_eval_negatives(tmp_path) -> None:
    nonlens_index_path, metadata_path = write_nonlens_fixture(tmp_path)
    lens_root = write_lens_fixture(tmp_path, normalized=True)
    train_pos_pred_paths = [
        write_saved_prediction(tmp_path / "preds" / "lens_train_0.pt", 0, "lens_train_record_0", "lens_train_sample_0", 0.25),
        write_saved_prediction(tmp_path / "preds" / "lens_train_1.pt", 1, "lens_train_record_1", "lens_train_sample_1", 0.5),
    ]
    train_neg_pred_paths = [
        write_saved_prediction(tmp_path / "preds" / "nonlens_train_0.pt", 0, "nonlens_train_record_0", "nonlens_train_sample_0", -0.2),
        write_saved_prediction(tmp_path / "preds" / "nonlens_train_1.pt", 1, "nonlens_train_record_1", "nonlens_train_sample_1", -0.1),
    ]
    eval_neg_pred_paths = [
        write_saved_prediction(tmp_path / "preds" / "nonlens_eval_0.pt", 0, "nonlens_eval_record_0", "nonlens_eval_sample_0", -0.4),
        write_saved_prediction(tmp_path / "preds" / "nonlens_eval_1.pt", 1, "nonlens_eval_record_1", "nonlens_eval_sample_1", -0.3),
    ]
    train_results = write_imputation_results_json(tmp_path / "train_results.json", train_pos_pred_paths, split="train")
    train_negative_results = write_imputation_results_json(
        tmp_path / "train_negative_results.json",
        train_neg_pred_paths,
        split="train",
    )
    eval_negative_results = write_imputation_results_json(
        tmp_path / "eval_negative_results.json",
        eval_neg_pred_paths,
        split="eval",
    )
    output_path = build_imputed_manifest(
        argparse.Namespace(
            nonlens_index_path=str(nonlens_index_path),
            train_negative_source="imputed",
            train_negative_results_json=str(train_negative_results),
            train_results_json=str(train_results),
            eval_results_json=None,
            eval_negative_source="imputed",
            eval_negative_results_json=str(eval_negative_results),
            normalization_metadata=str(metadata_path),
            eval_source="real",
            eval_lens_root=str(lens_root),
            eval_run="run_001",
            method=None,
            target_modality="roman",
            output=str(tmp_path / "imputed_eval_negative_manifest.json"),
        )
    )
    payload = json.loads(output_path.read_text())
    assert payload["summary"]["by_split"]["train"] == 4
    assert payload["summary"]["by_split"]["eval"] == 4
    assert payload["summary"]["by_source_kind"]["pred_sample_normalized"] == 6
    assert payload["summary"]["by_source_kind"]["lens_normalized"] == 2
    assert "nonlens_normalized" not in payload["summary"]["by_source_kind"]


def test_build_imputed_manifest_with_mixed_train_negatives(tmp_path) -> None:
    nonlens_index_path, metadata_path = write_nonlens_fixture(tmp_path)
    lens_root = write_lens_fixture(tmp_path, normalized=True)
    run_007_root = tmp_path / "nonlens_run_007_norm"
    shard_path = run_007_root / "run_007" / "shards" / "nonlens" / "nonlens_000007.zarr"
    _write_shard(shard_path, ["run_007_real_0", "run_007_real_1"], 0.15, True)
    manifest_path = run_007_root / "run_007" / "shards" / "manifest_lre_nonlens.json"
    manifest_path.write_text(
        json.dumps({"results": [{"status": "ok", "path": str(shard_path)}], "summary": {"ok": 1}}),
        encoding="utf-8",
    )

    train_pos_pred_paths = [
        write_saved_prediction(tmp_path / "preds" / "lens_train_0.pt", 0, "lens_train_record_0", "lens_train_sample_0", 0.25),
        write_saved_prediction(tmp_path / "preds" / "lens_train_1.pt", 1, "lens_train_record_1", "lens_train_sample_1", 0.5),
    ]
    train_neg_pred_paths = [
        write_saved_prediction(tmp_path / "preds" / "nonlens_train_0.pt", 0, "nonlens_train_record_0", "nonlens_train_sample_0", -0.2),
        write_saved_prediction(tmp_path / "preds" / "nonlens_train_1.pt", 1, "nonlens_train_record_1", "nonlens_train_sample_1", -0.1),
    ]
    train_results = write_imputation_results_json(tmp_path / "train_results.json", train_pos_pred_paths, split="train")
    train_negative_results = write_imputation_results_json(
        tmp_path / "train_negative_results.json",
        train_neg_pred_paths,
        split="train",
    )
    output_path = build_imputed_manifest(
        argparse.Namespace(
            nonlens_index_path=str(nonlens_index_path),
            train_negative_source="mixed",
            train_negative_results_json=str(train_negative_results),
            train_negative_real_root=str(run_007_root),
            train_negative_real_runs="run_007",
            train_results_json=str(train_results),
            eval_results_json=None,
            normalization_metadata=str(metadata_path),
            eval_source="real",
            eval_lens_root=str(lens_root),
            eval_run="run_001",
            method=None,
            target_modality="roman",
            output=str(tmp_path / "imputed_mixed_negative_manifest.json"),
        )
    )
    payload = json.loads(output_path.read_text())
    assert payload["track"] == "imputed_roman"
    assert payload["summary"]["by_split"]["train"] == 6
    assert payload["summary"]["by_split"]["eval"] == 4
    assert payload["summary"]["by_source_kind"]["pred_sample_normalized"] == 4
    assert payload["summary"]["by_source_kind"]["lens_normalized"] == 2
    assert payload["summary"]["by_source_kind"]["nonlens_normalized"] == 4

