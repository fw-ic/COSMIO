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

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "asyn-diffusion" / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_imputation_methods as evaluator
from asyndiff.ops.metrics import fair_crps_ensemble
from asyndiff.training import pipeline


@pytest.mark.parametrize("score", [fair_crps_ensemble, evaluator._crps_ensemble, pipeline._crps_ensemble])
@pytest.mark.parametrize(
    "draws, observation, expected",
    [([-1.0, 1.0], 0.0, 0.0),
     ([-1.0, 1.0], 3.0, 2.0),
     ([0.0, 1.0, 2.0, 3.0, 4.0], 0.0, 1.0),
     ([3.0, 3.0, 3.0], 1.0, 2.0)],
)
def test_fair_crps_exact_values(score, draws, observation, expected) -> None:
    # Analytic finite cases distinguish M(M-1) from M^2 without Monte Carlo noise.
    ensemble = torch.tensor(draws, dtype=torch.float64).reshape(-1, 1, 1, 1)
    target = torch.full((1, 1, 1), observation, dtype=torch.float64)
    assert score(ensemble, target) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("score", [fair_crps_ensemble, evaluator._crps_ensemble, pipeline._crps_ensemble])
def test_fair_crps_averages_batch_channels_and_pixels(score) -> None:
    ensemble = torch.tensor([[-1.0, -2.0], [1.0, 2.0]], dtype=torch.float64).reshape(2, 1, 1, 1, 2)
    target = torch.tensor([3.0, 6.0], dtype=torch.float64).reshape(1, 1, 1, 2)
    assert score(ensemble, target) == pytest.approx(3.0)


@pytest.mark.parametrize("score", [fair_crps_ensemble, evaluator._crps_ensemble, pipeline._crps_ensemble])
@pytest.mark.parametrize("count", [0, 1])
def test_fair_crps_rejects_insufficient_draws(score, count) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        score(torch.zeros(count, 1, 1, 1), torch.zeros(1, 1, 1))


@pytest.mark.parametrize("count", [-1, 0, 1])
def test_evaluator_cli_rejects_insufficient_draws(monkeypatch, count) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_imputation_methods.py", "--known", "lsst,euclid", "--crps-samples", str(count)])
    with pytest.raises(ValueError, match="--crps-samples must be >= 2"):
        evaluator.main()


@pytest.mark.parametrize("count", [-1, 0, 1])
def test_training_evaluation_rejects_insufficient_draws(count) -> None:
    with pytest.raises(ValueError, match="evaluation.num_crps_samples must be >= 2"):
        pipeline._evaluate(
            {"evaluation": {"num_crps_samples": count}}, None, [],
            SimpleNamespace(eval=lambda: None), torch.device("cpu"),
        )


def test_evaluation_pipeline_reports_fair_crps() -> None:
    class FixedDrawAdapter(evaluator.MethodAdapter):
        def __init__(self):
            super().__init__("synthetic")
            self.draws = iter([0.0, 1.0, 2.0, 3.0, 4.0])

        def native_targets(self, split, sample_index):
            return "synthetic:0", "source", {"roman": torch.zeros(1, 2, 2)}

        def infer_example(self, split, sample_index, known_modalities, output_stem):
            return {"roman": torch.full((1, 2, 2), next(self.draws))}

    result = evaluator._evaluate_method(
        FixedDrawAdapter(), split="val", sample_indices=[0],
        known_modalities=["lsst", "euclid"], target_modalities=["roman"],
        crps_samples=5, seed=1234, batch_size=1, num_workers=0,
    )
    assert result["crps_estimator"] == "fair"
    assert result["crps_samples"] == 5
    assert result["examples"][0]["metrics"]["roman"]["crps"] == pytest.approx(1.0)
    assert result["aggregate"]["by_target"]["roman"]["crps"] == pytest.approx(1.0)
