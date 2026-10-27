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

from typing import Any

import numpy as np


# numpy removed np.trapz in 2.0 in favour of np.trapezoid. Both packages declare
# numpy>=1.26 with no upper bound, so a clean install resolves 2.x and every
# call below raised AttributeError before this shim existed.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_curve_arrays(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_idxs = np.r_[distinct, sorted_labels.size - 1]
    true_positives = np.cumsum(sorted_labels)[threshold_idxs].astype(np.float64)
    false_positives = (1 + threshold_idxs - true_positives).astype(np.float64)
    return threshold_idxs, true_positives, false_positives


def binary_classification_metrics(labels: np.ndarray, positive_scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if y_true.shape != scores.shape:
        raise ValueError(f"Label and score shape mismatch: {y_true.shape} vs {scores.shape}")

    pred = (scores >= float(threshold)).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)

    auroc = float("nan")
    auprc = float("nan")
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives > 0 and negatives > 0:
        _threshold_idxs, tps, fps = _binary_curve_arrays(y_true, scores)
        tpr = np.r_[0.0, tps / max(positives, 1)]
        fpr = np.r_[0.0, fps / max(negatives, 1)]
        auroc = float(_trapezoid(tpr, fpr))

        # Average precision, as the step sum over recall increments. The
        # trapezoid rule is wrong here: it interpolates precision linearly
        # across each recall jump, and tied scores produce exactly those jumps.
        # Anchoring the curve at (recall 0, precision 1) compounds it with a
        # precision no threshold attains. Together they reported (1 + p) / 2 for
        # a classifier at chance with prevalence p -- 0.575 for a coin flip on a
        # 15% positive split, beside a correct auroc of 0.5.
        precision_curve = tps / np.maximum(tps + fps, 1.0)
        recall_curve = tps / max(positives, 1)
        auprc = float(np.sum(np.diff(np.r_[0.0, recall_curve]) * precision_curve))

    return {
        "count": int(y_true.size),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }

