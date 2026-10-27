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

import numpy as np
import torch
from asyndiff.ops.validation import require_finite

PSNR_PROTOCOL = "native-flux-gt-image-max-mean-db-v1"


def fair_crps_ensemble(ensemble: torch.Tensor, target: torch.Tensor) -> float:
    """Fair CRPS for M independent draws, averaged over target dimensions.

    The spread term uses ordered distinct pairs, with denominator M(M-1).
    Diagonal distances are zero, so summing all pairs gives the same numerator.
    """
    num_samples = ensemble.shape[0]
    if num_samples < 2:
        raise ValueError("Fair CRPS requires at least 2 ensemble samples (M >= 2).")
    require_finite(ensemble, "CRPS ensemble")
    require_finite(target, "CRPS target")
    term_obs = torch.abs(ensemble - target.unsqueeze(0)).mean(dim=0)
    pairwise = torch.abs(ensemble[:, None] - ensemble[None, :]).sum(dim=(0, 1))
    crps_map = term_obs - pairwise / (2 * num_samples * (num_samples - 1))
    require_finite(crps_map, "CRPS result")
    return float(crps_map.mean().item())


def reconstruction_metrics_per_image(pred: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    """Score BCHW images in native flux; peak is each GT image's full CHW max."""
    if pred.ndim != 4 or pred.shape != target.shape:
        raise ValueError("Reconstruction metrics require matching BCHW prediction/target tensors.")
    require_finite(pred, "reconstruction prediction")
    require_finite(target, "reconstruction target")
    pred_np = pred.detach().cpu().double().numpy().reshape(pred.shape[0], -1)
    target_np = target.detach().cpu().double().numpy().reshape(target.shape[0], -1)
    mse = np.mean((pred_np - target_np) ** 2, axis=1)
    peaks = np.max(target_np, axis=1)
    # Preserve the independent evaluator's existing numerical floors.
    psnr = 20.0 * np.log10(np.maximum(peaks, 1.0e-12)) - 10.0 * np.log10(np.maximum(mse, 1.0e-12))
    require_finite(mse, "reconstruction MSE")
    require_finite(psnr, "reconstruction PSNR")
    return {"mse": mse, "rmse": np.sqrt(mse), "psnr": psnr}


def batch_reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    per_image = reconstruction_metrics_per_image(pred, target)
    return {key: float(np.mean(per_image[key])) for key in ("mse", "psnr")}


def merge_metric_dict(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}

