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
import torch
import torch.nn.functional as F
from asyndiff.constants import MODALITY_CHANNELS, MODALITY_ORDER
def center_pad_tensor(tensor: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    target_h, target_w = int(size[0]), int(size[1])
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(tensor.shape)}")
    channels, height, width = tensor.shape
    if height > target_h or width > target_w:
        raise ValueError(f"Cannot pad tensor shape={tuple(tensor.shape)} into smaller size={size}")
    padded = tensor.new_zeros((channels, target_h, target_w))
    mask = tensor.new_zeros((1, target_h, target_w))
    top = (target_h - height) // 2
    left = (target_w - width) // 2
    padded[:, top : top + height, left : left + width] = tensor
    mask[:, top : top + height, left : left + width] = 1.0
    return padded, mask
def fit_tensor_to_size(tensor: torch.Tensor, size: tuple[int, int], *, use_padding: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if use_padding:
        return center_pad_tensor(tensor, size)
    resized = F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    mask = resized.new_ones((1, int(size[0]), int(size[1])))
    return resized, mask
def restore_to_native(tensor: torch.Tensor, native_tensor: torch.Tensor, *, spatial_mode: str) -> torch.Tensor:
    if spatial_mode not in {"padding", "resize", "native"}:
        raise ValueError(f"Unknown spatial_mode: {spatial_mode!r}")
    target_h, target_w = native_tensor.shape[-2:]
    height, width = tensor.shape[-2:]
    if (height, width) == (target_h, target_w):
        return tensor
    if spatial_mode == "resize":
        batched = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        output = F.interpolate(batched, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return output[0] if tensor.ndim == 3 else output
    if spatial_mode == "native":
        raise ValueError("An adapter declaring native output returned the wrong spatial size.")
    if height < target_h or width < target_w:
        raise ValueError("Padded prediction is smaller than its native target.")
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return tensor[..., top:top + target_h, left:left + target_w]


def crop_to_native(tensor: torch.Tensor, native_tensor: torch.Tensor) -> torch.Tensor:
    """Restore a centre-padded tensor. Resized tensors must use restore_to_native."""
    return restore_to_native(tensor, native_tensor, spatial_mode="padding")


def stack_valid_masks(valid_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    stacked: list[torch.Tensor] = []
    for name in MODALITY_ORDER:
        mask = valid_masks[name]
        if mask.ndim == 3:
            if mask.shape[0] != 1:
                raise ValueError(f"Expected single-channel mask for {name}, got {tuple(mask.shape)}")
            repeated = mask.expand(int(MODALITY_CHANNELS[name]), -1, -1)
        elif mask.ndim == 4:
            if mask.shape[1] != 1:
                raise ValueError(f"Expected single-channel batched mask for {name}, got {tuple(mask.shape)}")
            repeated = mask.expand(mask.shape[0], int(MODALITY_CHANNELS[name]), mask.shape[2], mask.shape[3])
        else:
            raise ValueError(f"Unsupported valid mask shape for {name}: {tuple(mask.shape)}")
        stacked.append(repeated)
    return torch.cat(stacked, dim=-3)
