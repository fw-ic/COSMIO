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
import zarr
from torch.utils.data import Dataset, get_worker_info
from asyndiff.config import load_json
from asyndiff.constants import MODALITY_CHANNELS, MODALITY_ORDER
from asyndiff.data.index import backfill_record_ids
from asyndiff.data.normalization import (
    denormalize_tensor as _denormalize_tensor,
    normalize_tensor as _normalize_tensor,
    stats_use_signed_capped_asinh,
)
from asyndiff.data.protocol import fit_tensor_to_size
from asyndiff.ops.validation import require_finite
def _subset_record_and_sample_ids(path: str | Path, split: str) -> tuple[set[str] | None, set[str] | None]:
    manifest = load_json(path)
    if "subsets" in manifest:
        subset = manifest["subsets"].get(split, {})
    elif manifest.get("split") == split or "split" not in manifest:
        subset = manifest
    else:
        subset = {}
    record_ids = set(subset.get("record_ids", [])) or None
    sample_ids = set(subset.get("sample_ids", [])) or None
    return record_ids, sample_ids
def _to_channel_first(array: np.ndarray) -> torch.Tensor:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(np.moveaxis(array, -1, 0)).float()
def _normalize(modality: str, tensor: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    if stats_use_signed_capped_asinh(stats):
        return tensor
    return _normalize_tensor(modality, tensor, stats)
def denormalize_tensor(modality: str, tensor: torch.Tensor, stats: dict[str, Any]) -> torch.Tensor:
    return _denormalize_tensor(modality, tensor, stats)
def denormalize_modalities(modalities: dict[str, torch.Tensor], stats: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: denormalize_tensor(name, tensor, stats) for name, tensor in modalities.items()}
def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "record_id": [sample["record_id"] for sample in samples],
        "sample_id": [sample["sample_id"] for sample in samples],
        "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "inputs": {},
        "valid_masks": {},
        "native_targets": {},
        "native_sizes": samples[0]["native_sizes"],
    }
    for name in MODALITY_ORDER:
        batch["inputs"][name] = torch.stack([sample["inputs"][name] for sample in samples], dim=0)
        batch["valid_masks"][name] = torch.stack([sample["valid_masks"][name] for sample in samples], dim=0)
        batch["native_targets"][name] = torch.stack([sample["native_targets"][name] for sample in samples], dim=0)
    return batch
class NonlensJointDataset(Dataset):
    def __init__(
        self,
        split: str,
        index_path: str | Path,
        stats_path: str | Path,
        target_group: str,
        resize: int,
        train: bool = False,
        visualization_subset: str | Path | None = None,
        augment_flip_prob: float = 0.0,
        augment_shift_prob: float = 0.0,
        augment_scale_prob: float = 0.0,
        augment_noise_prob: float | None = None,
    ) -> None:
        self.index_manifest = backfill_record_ids(load_json(index_path))
        self.stats = load_json(stats_path)
        self.spatial_mode = "padding" if stats_use_signed_capped_asinh(self.stats) else "resize"
        self.target_group = target_group
        self.resize = resize
        self.train = train
        self.augment_flip_prob = augment_flip_prob
        # Legacy configurations use shift to mean additive Gaussian noise.
        self.augment_noise_prob = augment_shift_prob if augment_noise_prob is None else augment_noise_prob
        self.augment_scale_prob = augment_scale_prob
        subset_record_ids = None
        subset_sample_ids = None
        if visualization_subset is not None:
            subset_record_ids, subset_sample_ids = _subset_record_and_sample_ids(visualization_subset, split)
        self.records = [
            record
            for record in self.index_manifest["records"]
            if record["split"] == split
            and (
                (subset_record_ids is None and subset_sample_ids is None)
                or (subset_record_ids is not None and record["record_id"] in subset_record_ids)
                or (subset_record_ids is None and subset_sample_ids is not None and record["sample_id"] in subset_sample_ids)
            )
        ]
        self._group_cache: dict[int, dict[str, zarr.Group]] = {}
    def __len__(self) -> int:
        return len(self.records)
    def _get_group(self, shard_path: str) -> zarr.Group:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        cache = self._group_cache.setdefault(worker_id, {})
        group = cache.get(shard_path)
        if group is None:
            group = zarr.open_group(shard_path, mode="r")[self.target_group]
            cache[shard_path] = group
        return group
    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.augment_noise_prob:
            tensor = tensor + torch.randn_like(tensor) * 0.02
        if random.random() < self.augment_scale_prob:
            tensor = tensor * (1.0 + (random.random() - 0.5) * 0.2)
        return tensor
    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        group = self._get_group(record["shard_path"])
        local_index = int(record["local_index"])
        native_targets: dict[str, torch.Tensor] = {}
        resized_inputs: dict[str, torch.Tensor] = {}
        valid_masks: dict[str, torch.Tensor] = {}
        native_sizes: dict[str, tuple[int, int]] = {}
        use_signed_asinh = stats_use_signed_capped_asinh(self.stats)
        # One spatial transform per sample, before padding (including odd margins).
        flip_dims = [dim for dim in (-1, -2) if self.train and random.random() < self.augment_flip_prob]
        for name in MODALITY_ORDER:
            native = _to_channel_first(group[name][local_index])
            context = f"dataset record_id={record['record_id']} modality={name}"
            require_finite(native, context + " raw input")
            if native.shape[0] != MODALITY_CHANNELS[name]:
                raise ValueError(f"Unexpected channel count for {name}: {tuple(native.shape)}")
            if flip_dims:
                native = torch.flip(native, dims=flip_dims)
            native_targets[name] = _denormalize_tensor(name, native, self.stats) if use_signed_asinh else native
            native_sizes[name] = tuple(native.shape[-2:])
            resized, valid_mask = fit_tensor_to_size(
                native,
                (self.resize, self.resize),
                use_padding=use_signed_asinh,
            )
            if self.train:
                resized = self._augment(resized)
            resized_inputs[name] = _normalize(name, resized, self.stats)
            valid_masks[name] = valid_mask
            require_finite(resized_inputs[name], context + " normalized input")
            require_finite(native_targets[name], context + " native target")
        return {
            "record_id": record["record_id"],
            "sample_id": record["sample_id"],
            "label": int(record["label"]),
            "inputs": resized_inputs,
            "valid_masks": valid_masks,
            "native_targets": native_targets,
            "native_sizes": native_sizes,
        }
