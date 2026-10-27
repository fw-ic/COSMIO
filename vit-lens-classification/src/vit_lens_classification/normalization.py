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
import torch
SIGNED_CAPPED_ASINH_SCHEMA = "signed_capped_asinh"
ROMAN_BANDS = ("F106", "F129", "F158")
def load_normalization_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if str(metadata.get("method", {}).get("name", "")) != SIGNED_CAPPED_ASINH_SCHEMA:
        raise ValueError(f"Unsupported normalization method in {metadata_path}: {metadata.get('method', {}).get('name')}")
    if "modalities" not in metadata or "roman" not in metadata["modalities"]:
        raise ValueError(f"Missing Roman normalization metadata in {metadata_path}.")
    return metadata
def _broadcast_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return (1,) * max(0, tensor.ndim - 3) + (-1, 1, 1)
def _param_tensor(tensor: torch.Tensor, metadata: dict[str, Any], field: str) -> torch.Tensor:
    params = metadata["modalities"]["roman"]
    values = [float(params[band][field]) for band in ROMAN_BANDS]
    return torch.tensor(values, dtype=tensor.dtype, device=tensor.device).view(*_broadcast_shape(tensor))
def normalize_roman_tensor(tensor: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    center = _param_tensor(tensor, metadata, "center")
    sigma = _param_tensor(tensor, metadata, "sigma").clamp_min(1.0e-6)
    a_neg = _param_tensor(tensor, metadata, "a_neg").clamp_min(1.0e-6)
    a_pos = _param_tensor(tensor, metadata, "a_pos").clamp_min(1.0e-6)
    delta = tensor - center
    positive = delta >= 0
    positive_branch = torch.asinh(torch.clamp_min(delta, 0.0) / sigma) / a_pos
    negative_branch = -torch.asinh(torch.clamp_min(-delta, 0.0) / sigma) / a_neg
    return torch.where(positive, positive_branch, negative_branch).clamp(-1.0, 1.0)
def denormalize_roman_tensor(tensor: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    tensor = tensor.clamp(-1.0, 1.0)
    center = _param_tensor(tensor, metadata, "center")
    sigma = _param_tensor(tensor, metadata, "sigma").clamp_min(1.0e-6)
    a_neg = _param_tensor(tensor, metadata, "a_neg").clamp_min(1.0e-6)
    a_pos = _param_tensor(tensor, metadata, "a_pos").clamp_min(1.0e-6)
    positive = tensor >= 0
    positive_branch = center + sigma * torch.sinh(tensor * a_pos)
    negative_branch = center - sigma * torch.sinh((-tensor) * a_neg)
    return torch.where(positive, positive_branch, negative_branch)
