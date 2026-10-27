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

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from asyndiff.data.dataset import NonlensJointDataset, collate_samples
from asyndiff.data.protocol import restore_to_native


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_run_directories(config: dict[str, Any], stage: str) -> dict[str, Path]:
    experiment = config["experiment"]["name"]
    root = Path(config["experiment"]["output_root"]) / experiment / stage
    directories = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "tb": root / "tb",
        "media": root / "media",
        "history": root / "history.jsonl",
    }
    for path in directories.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return directories


def create_dataset(config: dict[str, Any], split: str, visualization_subset: bool = False, *, training: bool = False) -> NonlensJointDataset:
    data_cfg = config["data"]
    subset = data_cfg["visualization_subset_path"] if visualization_subset else None
    return NonlensJointDataset(
        split=split,
        index_path=data_cfg["index_path"],
        stats_path=data_cfg["stats_path"],
        target_group=data_cfg["target_group"],
        resize=int(data_cfg["resize"]),
        train=training and not visualization_subset,
        visualization_subset=subset,
        augment_flip_prob=float(data_cfg.get("augment_flip_prob", 0.0)),
        augment_shift_prob=float(data_cfg.get("augment_shift_prob", 0.0)),
        augment_noise_prob=(float(data_cfg["augment_noise_prob"]) if "augment_noise_prob" in data_cfg else None),
        augment_scale_prob=float(data_cfg.get("augment_scale_prob", 0.0)),
    )


def create_loader(
    dataset: NonlensJointDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collate_samples,
    )


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().data.norm(2).item() ** 2)
    return total**0.5


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def resize_batch_to_native(tensor: torch.Tensor, native_tensor: torch.Tensor, *, spatial_mode: str = "padding") -> torch.Tensor:
    return restore_to_native(tensor, native_tensor, spatial_mode=spatial_mode)

