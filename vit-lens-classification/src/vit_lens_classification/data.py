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

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, get_worker_info

from vit_lens_classification.manifest import load_json
from vit_lens_classification.normalization import load_normalization_metadata, normalize_roman_tensor


def _to_chw(array: np.ndarray) -> torch.Tensor:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    return torch.from_numpy(np.moveaxis(data, -1, 0)).float()


def _resize_image(tensor: torch.Tensor, image_size: int) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(tensor.shape)}")
    if tensor.shape[-2:] == (image_size, image_size):
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )[0]


class RomanManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str, image_size: int) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_json(self.manifest_path)
        self.entries = [item for item in self.manifest.get("entries", []) if item.get("split") == split]
        self.image_size = int(image_size)
        self.split = str(split)
        self._group_cache: dict[int, dict[str, zarr.Group]] = {}
        self._prediction_cache: dict[int, dict[str, Any]] = {}
        metadata_path = self.manifest.get("normalization_metadata_path")
        self._normalization_metadata_path = metadata_path
        self.normalization_metadata = None

    def __len__(self) -> int:
        return len(self.entries)

    def _worker_cache(self, store: dict[int, dict[str, Any]]) -> dict[str, Any]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        return store.setdefault(worker_id, {})

    def _get_group(self, shard_path: str, target_group: str) -> zarr.Group:
        cache = self._worker_cache(self._group_cache)
        cache_key = f"{shard_path}::{target_group}"
        group = cache.get(cache_key)
        if group is None:
            group = zarr.open_group(str(shard_path), mode="r")[target_group]
            cache[cache_key] = group
        return group

    def _load_prediction(self, prediction_path: str) -> Any:
        cache = self._worker_cache(self._prediction_cache)
        payload = cache.get(prediction_path)
        if payload is None:
            payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
            cache[prediction_path] = payload
        return payload

    def _load_image(self, entry: dict[str, Any]) -> torch.Tensor:
        source_kind = str(entry["source_kind"])
        if source_kind in {"nonlens_normalized", "lens_normalized"}:
            group = self._get_group(str(entry["shard_path"]), str(entry.get("target_group", "images")))
            return _to_chw(group["roman"][int(entry["local_index"])])
        if source_kind == "lens_raw":
            if self.normalization_metadata is None:
                if not self._normalization_metadata_path:
                    raise ValueError("Lens raw entries require normalization_metadata_path in the manifest.")
                self.normalization_metadata = load_normalization_metadata(self._normalization_metadata_path)
            group = self._get_group(str(entry["shard_path"]), str(entry.get("target_group", "images")))
            tensor = _to_chw(group["roman"][int(entry["local_index"])])
            return normalize_roman_tensor(tensor, self.normalization_metadata)
        if source_kind == "pred_sample_normalized":
            payload = self._load_prediction(str(entry["prediction_path"]))
            tensor = payload.get("pred_normalized")
            if isinstance(tensor, dict):
                target_modality = str(entry.get("target_modality", "roman"))
                if target_modality not in tensor:
                    raise KeyError(
                        f"Saved prediction payload {entry['prediction_path']} does not contain `{target_modality}`."
                    )
                tensor = tensor[target_modality]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Saved prediction payload {entry['prediction_path']} is missing a tensor.")
            return tensor.detach().cpu().float()
        raise KeyError(f"Unsupported source_kind `{source_kind}`.")

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image = _resize_image(self._load_image(entry), self.image_size)
        return {
            "image": image,
            "label": int(entry["label"]),
            "record_id": str(entry["record_id"]),
            "sample_id": str(entry["sample_id"]),
        }


def create_dataloader(
    manifest_path: str | Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    balanced_sampling: bool = False,
    training: bool = False,
) -> DataLoader:
    dataset = RomanManifestDataset(manifest_path=manifest_path, split=split, image_size=image_size)
    labels_present = {int(entry["label"]) for entry in dataset.entries}
    if labels_present != {0, 1}:
        raise ValueError(f"Classifier {split} split requires both labels 0 and 1; got {sorted(labels_present)}.")
    sampler = None
    shuffle = training and not balanced_sampling
    if training and balanced_sampling:
        labels = torch.tensor([int(item["label"]) for item in dataset.entries], dtype=torch.long)
        class_counts = torch.bincount(labels, minlength=2).clamp_min(1)
        sample_weights = torch.tensor([1.0 / float(class_counts[label]) for label in labels.tolist()], dtype=torch.float32)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=False,
    )

