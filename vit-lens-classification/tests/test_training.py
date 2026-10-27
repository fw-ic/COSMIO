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
from pathlib import Path

from vit_lens_classification.build_imputed_roman_manifest import build_manifest as build_imputed_manifest
from vit_lens_classification.build_real_roman_manifest import build_manifest as build_real_manifest
from vit_lens_classification.evaluate_vit import evaluate
from vit_lens_classification.train_vit import train

from .helpers import (
    make_train_config,
    write_imputation_results_json,
    write_lens_fixture,
    write_nonlens_fixture,
    write_saved_prediction,
)


def _build_real_manifest(tmp_path: Path) -> Path:
    nonlens_index_path, metadata_path = write_nonlens_fixture(tmp_path)
    nonlens_root = metadata_path.parent
    lens_root = write_lens_fixture(tmp_path, normalized=True)
    return build_real_manifest(
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


def _build_imputed_manifest(tmp_path: Path) -> Path:
    nonlens_index_path, metadata_path = write_nonlens_fixture(tmp_path)
    lens_root = write_lens_fixture(tmp_path, normalized=True)
    train_results = write_imputation_results_json(
        tmp_path / "train_results.json",
        [
            write_saved_prediction(tmp_path / "preds" / "train_0.pt", 0, "train_record_0", "train_sample_0", 0.2),
            write_saved_prediction(tmp_path / "preds" / "train_1.pt", 1, "train_record_1", "train_sample_1", 0.3),
        ],
        split="train",
    )
    return build_imputed_manifest(
        argparse.Namespace(
            nonlens_index_path=str(nonlens_index_path),
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


def test_real_track_train_and_eval_smoke(tmp_path) -> None:
    manifest_path = _build_real_manifest(tmp_path)
    config = make_train_config(tmp_path, manifest_path, name="real_smoke")
    summary = train(config)
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["periodic_checkpoints"][0]).exists()
    eval_summary = evaluate(config)
    assert eval_summary["prediction_count"] > 0
    assert "accuracy" in eval_summary["metrics"]


def test_imputed_track_train_and_eval_smoke(tmp_path) -> None:
    manifest_path = _build_imputed_manifest(tmp_path)
    config = make_train_config(tmp_path, manifest_path, name="imputed_smoke")
    summary = train(config)
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["periodic_checkpoints"][0]).exists()
    eval_summary = evaluate(config)
    assert eval_summary["prediction_count"] > 0
    assert "auroc" in eval_summary["metrics"]

