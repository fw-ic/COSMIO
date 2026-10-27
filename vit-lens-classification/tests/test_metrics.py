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

"""Regression tests for the binary classification metrics.

No test in the suite reached the AUROC/AUPRC branch before this file. It is
guarded by `positives > 0 and negatives > 0`, and the fixtures evaluate a single
unshuffled batch of two whose labels are both 0, so the branch stayed dead while
179 tests reported green. It contained two defects: a call to np.trapz, removed
in NumPy 2.0, and an AUPRC that was a trapezoidal integral of an anchored PR
curve rather than average precision.
"""

from __future__ import annotations

import numpy as np

from vit_lens_classification.metrics import binary_classification_metrics


def test_mixed_labels_produce_finite_curve_metrics() -> None:
    """The branch that never ran. On NumPy 2.x this raised AttributeError."""
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    metrics = binary_classification_metrics(labels, scores, threshold=0.5)
    assert np.isfinite(metrics["auroc"]), "auroc must be finite when both classes are present"
    assert np.isfinite(metrics["auprc"]), "auprc must be finite when both classes are present"


def test_curve_metrics_match_hand_computed_reference() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    metrics = binary_classification_metrics(labels, scores, threshold=0.5)
    # AUROC by pairwise ranking: 3 of 4 positive/negative pairs ordered correctly.
    assert metrics["auroc"] == 0.75
    # Average precision as the step sum: 0.5*1.0 + 0.5*(2/3).
    assert abs(metrics["auprc"] - 0.8333333333333333) < 1e-12


def test_perfect_separation_scores_one() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    metrics = binary_classification_metrics(labels, scores, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert abs(metrics["auprc"] - 1.0) < 1e-12


def test_collapsed_classifier_reports_prevalence_not_inflated_precision() -> None:
    """A constant score is the case the old definition flattered.

    Every score ties, so the PR curve is a single point at (recall 1, precision
    p). Average precision is p. The previous implementation anchored the curve
    at (0, 1.0) and integrated with the trapezoid rule, reporting (1 + p) / 2 --
    0.575 here -- next to a correctly-chance auroc of 0.5.
    """
    labels = np.array([1] * 15 + [0] * 85)
    scores = np.full(labels.shape, 0.5)
    metrics = binary_classification_metrics(labels, scores, threshold=0.5)
    prevalence = 0.15
    assert metrics["auroc"] == 0.5
    assert abs(metrics["auprc"] - prevalence) < 1e-12
    assert metrics["auprc"] < 0.575, "auprc must not be inflated toward (1 + prevalence) / 2"


def test_single_class_batch_yields_nan_curves() -> None:
    labels = np.zeros(4, dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = binary_classification_metrics(labels, scores, threshold=0.5)
    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["auprc"])
    assert metrics["count"] == 4
