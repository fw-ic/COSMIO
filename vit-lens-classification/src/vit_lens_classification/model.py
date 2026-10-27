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

from torchvision.models.vision_transformer import VisionTransformer


def build_model(model_config: dict[str, Any]) -> VisionTransformer:
    in_channels = int(model_config.get("in_channels", 3))
    if in_channels != 3:
        raise ValueError(f"Roman-only ViT expects 3 input channels, got {in_channels}.")
    image_size = int(model_config["image_size"])
    patch_size = int(model_config["patch_size"])
    if image_size % patch_size != 0:
        raise ValueError(f"image_size must be divisible by patch_size, got {image_size} and {patch_size}.")
    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        hidden_dim=int(model_config["hidden_dim"]),
        mlp_dim=int(model_config["mlp_dim"]),
        dropout=float(model_config.get("dropout", 0.0)),
        attention_dropout=float(model_config.get("attention_dropout", 0.0)),
        num_classes=2,
    )

