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

from helpers import make_test_config
from asyndiff.training import sample_prediction, train_asyn_diffusion


def test_asyndiff_smoke(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    train_asyn_diffusion(config)

    root = Path(config["experiment"]["output_root"]) / config["experiment"]["name"] / "asyn_diffusion"
    ckpt = root / "checkpoints" / "asyndiff_best.pt"
    assert ckpt.exists()

    history_path = root / "history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    row = rows[-1]
    assert row["crps_estimator"] == "fair"
    assert row["crps_samples"] == config["evaluation"]["num_crps_samples"]
    assert "roman_both_crps" in row
    assert "roman_lsst_only_crps" in row
    assert "roman_euclid_only_crps" in row
    assert "roman_mean_crps" in row
    assert "roman_both_psnr" in row
    assert "roman_lsst_only_psnr" in row
    assert "roman_euclid_only_psnr" in row


def test_asyndiff_smoke_with_asinh_normalized_data(tmp_path: Path) -> None:
    config = make_test_config(tmp_path, normalized_with_asinh=True)
    train_asyn_diffusion(config)

    root = Path(config["experiment"]["output_root"]) / config["experiment"]["name"] / "asyn_diffusion"
    ckpt = root / "checkpoints" / "asyndiff_best.pt"
    assert ckpt.exists()


def test_asyndiff_single_head_smoke(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    config["model"]["head_mode"] = "single"

    train_asyn_diffusion(config)

    root = Path(config["experiment"]["output_root"]) / config["experiment"]["name"] / "asyn_diffusion"
    ckpt = root / "checkpoints" / "asyndiff_best.pt"
    assert ckpt.exists()

    prediction = sample_prediction(config, split="test", sample_index=0, known_modalities=["lsst", "euclid"])
    assert Path(prediction).exists()


def test_sample_known_variants_and_invalid_known(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    train_asyn_diffusion(config)

    both = sample_prediction(config, split="test", sample_index=0, known_modalities=["lsst", "euclid"])
    lsst_only = sample_prediction(config, split="test", sample_index=0, known_modalities=["lsst"])
    euclid_only = sample_prediction(config, split="test", sample_index=0, known_modalities=["euclid"])
    lsst_roman = sample_prediction(config, split="test", sample_index=0, known_modalities=["lsst", "roman"])
    assert Path(both).exists()
    assert Path(lsst_only).exists()
    assert Path(euclid_only).exists()
    assert Path(lsst_roman).exists()

    try:
        sample_prediction(config, split="test", sample_index=0, known_modalities=[""])
    except ValueError as exc:
        assert "Known modalities must include" in str(exc)
    else:
        raise AssertionError("Expected sample_prediction to reject an empty known list")

    try:
        sample_prediction(config, split="test", sample_index=0, known_modalities=["roman"])
    except ValueError as exc:
        assert "only supports known modality combinations" in str(exc)
    else:
        raise AssertionError("Expected sample_prediction to reject unsupported known modalities")


def test_lsst_roman_eval_logs_euclid_metrics(tmp_path: Path) -> None:
    config = make_test_config(tmp_path)
    config["experiment"]["name"] = "lsst_roman_eval"
    config["train"]["known_combinations"] = [["lsst", "roman"]]
    config["evaluation"]["visualization_known_modalities"] = ["lsst", "roman"]
    config["evaluation"]["known_combinations"] = [["lsst", "roman"]]

    train_asyn_diffusion(config)

    root = Path(config["experiment"]["output_root"]) / config["experiment"]["name"] / "asyn_diffusion"
    history_path = root / "history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    row = rows[-1]
    assert "euclid_lsst_roman_crps" in row
    assert "euclid_mean_crps" in row
    assert "mean_crps" in row

