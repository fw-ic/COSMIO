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

import argparse
import copy
import concurrent.futures
import importlib
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "asyn-diffusion" / "src"))
from asyndiff.data.protocol import restore_to_native
from asyndiff.data.normalization import stats_use_signed_capped_asinh
from asyndiff.ops.random import SEED_SCHEME, sample_seed
from asyndiff.ops.metrics import PSNR_PROTOCOL, reconstruction_metrics_per_image
from asyndiff.ops.validation import require_finite, require_finite_batch

ALL_METHODS = ("asyn-diffusion",)
ALL_MODALITIES = ("lsst", "euclid", "roman")
DEFAULT_MODALITY_CHANNELS = {"lsst": 5, "euclid": 1, "roman": 3}
DEFAULT_LENS_DATA_ROOT = REPO_ROOT / "data/lens"
DEFAULT_LENS_RUNS = ("run_000",)
DEFAULT_NONLENS_DATA_ROOT = REPO_ROOT / "data/regular"
DEFAULT_IMPUTATION_RUNS = ("run_007",)
DEFAULT_IMPUTATION_SET_SIZE = 8000
CUSTOM_SPLIT_RECORD_COUNT = 10**12
IMPUTATION_SPLIT_ALIASES = {"imputation", "imputation_set"}


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stabilize_default_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    if data_cfg:
        data_cfg["num_workers"] = 0
        data_cfg["pin_memory"] = False
    return config


def _parse_modalities(value: str | None) -> list[str]:
    if value is None:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in ALL_MODALITIES]
    if invalid:
        raise ValueError(f"Invalid modalities: {invalid}. Expected subset of {ALL_MODALITIES}.")
    ordered = [name for name in ALL_MODALITIES if name in set(items)]
    if not ordered:
        raise ValueError("Expected at least one modality.")
    return ordered


def _parse_run_ids(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _configure_all_records_to_split(config: dict[str, Any], split: str) -> None:
    data_cfg = config.setdefault("data", {})
    normalized_split = _normalize_nonlens_split(split)
    if normalized_split not in {"train", "val", "test"}:
        raise ValueError(
            "Custom nonlens run evaluation only supports split=train,val,test,imputation; "
            f"got split={split!r}."
        )
    data_cfg["train_count"] = 0
    data_cfg["val_count"] = 0
    data_cfg["test_count"] = 0
    data_cfg[f"{normalized_split}_count"] = CUSTOM_SPLIT_RECORD_COUNT


def _is_lens_split(split: str) -> bool:
    return str(split).strip().lower() == "lens"


def _is_imputation_split(split: str) -> bool:
    return str(split).strip().lower() in IMPUTATION_SPLIT_ALIASES


def _normalize_nonlens_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized in IMPUTATION_SPLIT_ALIASES:
        return "train"
    return normalized


def _runtime_dataset_config(
    base_config: dict[str, Any],
    data_root: Path,
    manifest_name: str,
    selected_runs: tuple[str, ...],
    artifact_root: Path,
    split: str,
) -> dict[str, Any]:
    runtime_config = copy.deepcopy(base_config)
    runtime_config["data"] = copy.deepcopy(runtime_config.get("data", {}))
    runtime_config["evaluation"] = copy.deepcopy(runtime_config.get("evaluation", {}))
    runtime_config.setdefault("inference", {})
    runtime_config["data"]["data_root"] = str(data_root)
    runtime_config["data"]["manifest_name"] = str(manifest_name)
    runtime_config["data"]["selected_runs"] = list(selected_runs)
    runtime_config["data"]["index_path"] = str(artifact_root / "index.json")
    runtime_config["data"]["stats_path"] = str(artifact_root / "stats.json")
    runtime_config["data"]["visualization_subset_path"] = str(artifact_root / "visualization_subset.json")
    if _is_lens_split(split):
        runtime_config["data"]["train_count"] = 0
        runtime_config["data"]["val_count"] = 0
        runtime_config["data"]["test_count"] = CUSTOM_SPLIT_RECORD_COUNT
        runtime_config["evaluation"]["visualization_split"] = "test"
        runtime_config["evaluation"]["visualization_splits"] = ["test"]
    else:
        dataset_split = _normalize_nonlens_split(split)
        _configure_all_records_to_split(runtime_config, dataset_split)
        runtime_config["evaluation"]["visualization_split"] = dataset_split
        runtime_config["evaluation"]["visualization_splits"] = [dataset_split]
    return runtime_config


def _resolve_sample_indices(start_index: int, requested_num_examples: int | None, split_size: int) -> list[int]:
    start = int(start_index)
    total = int(split_size)
    if start < 0:
        raise ValueError("--start-index must be >= 0.")
    if start >= total:
        raise ValueError(f"--start-index {start} is out of range for split size {total}.")
    if requested_num_examples is None:
        end = total
    else:
        count = int(requested_num_examples)
        if count < 1:
            raise ValueError("--num-examples must be >= 1 when provided.")
        end = min(total, start + count)
    return list(range(start, end))


def _tensor_to_chw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW or 1CHW tensor, got shape={tuple(tensor.shape)}")
    return tensor


def _resize_to_native(prediction: torch.Tensor, target: torch.Tensor, *, spatial_mode: str = "native") -> torch.Tensor:
    return restore_to_native(_tensor_to_chw(prediction), _tensor_to_chw(target), spatial_mode=spatial_mode)


def _reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    per_image = reconstruction_metrics_per_image(prediction.unsqueeze(0), target.unsqueeze(0))
    return {key: float(per_image[key][0]) for key in ("rmse", "psnr")}


def _crps_ensemble(ensemble: torch.Tensor, target: torch.Tensor) -> float:
    # Resolve the bundled package just as the model adapter does, so the root
    # evaluator also works from a source checkout without an editable install.
    _prepend_sys_path(REPO_ROOT / "asyn-diffusion" / "src")
    from asyndiff.ops.metrics import fair_crps_ensemble

    return fair_crps_ensemble(ensemble, target)


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _stack_tensor_dict(samples: list[dict[str, Any]], field: str) -> dict[str, torch.Tensor]:
    keys = samples[0][field].keys()
    return {
        name: torch.stack([sample[field][name] for sample in samples], dim=0)
        for name in keys
    }


def _stack_tensor_field(samples: list[dict[str, Any]], field: str) -> torch.Tensor:
    return torch.stack([sample[field] for sample in samples], dim=0)


def _split_prediction_batch(predictions: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    batch_size = next(iter(predictions.values())).shape[0]
    outputs: list[dict[str, torch.Tensor]] = []
    for batch_index in range(batch_size):
        outputs.append({name: tensor[batch_index].detach().cpu() for name, tensor in predictions.items()})
    return outputs


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[start : start + chunk_size] for start in range(0, len(values), chunk_size)]


def _identity_collate(samples: list[Any]) -> list[Any]:
    return samples


def _load_samples_for_indices(dataset: Any, sample_indices: list[int], num_workers: int) -> list[Any]:
    if not sample_indices:
        return []
    if int(num_workers) <= 0:
        return [dataset[sample_index] for sample_index in sample_indices]
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
        return list(executor.map(dataset.__getitem__, sample_indices))


@dataclass
class ExampleResult:
    sample_index: int
    record_id: str
    sample_id: str
    metrics_by_target: dict[str, dict[str, float]]
    visualization_path: str | None = None
    pred_normalized_path: str | None = None
    pred_sample_normalized_paths: dict[str, str] | None = None


class MethodAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.num_workers = 0

    def validate_request(self, known_modalities: list[str], target_modalities: list[str]) -> None:
        return None

    def infer_example(self, split: str, sample_index: int, known_modalities: list[str], output_stem: Path) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def infer_examples(
        self,
        split: str,
        sample_indices: list[int],
        known_modalities: list[str],
        output_root: Path,
    ) -> list[dict[str, torch.Tensor]]:
        outputs: list[dict[str, torch.Tensor]] = []
        for sample_index in sample_indices:
            outputs.append(
                self.infer_example(
                    split=split,
                    sample_index=sample_index,
                    known_modalities=known_modalities,
                    output_stem=output_root / f"{self.name}_{sample_index:05d}",
                )
            )
        return outputs

    def infer_seeded_examples(self, split, sample_indices, known_modalities, output_root, sample_seeds):
        """Compatibility path for adapters without per-record batched generators."""
        if len(sample_seeds) != len(sample_indices):
            raise ValueError("Expected one seed per requested record.")
        outputs = []
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng():
                for index, record_seed in zip(sample_indices, sample_seeds):
                    _seed_everything(record_seed)
                    outputs.append(self.infer_example(split, index, known_modalities, output_root / str(index)))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        return outputs

    def prediction_spatial_mode(self, split):
        # Adapters must explicitly declare model-space outputs requiring restoration.
        return "native"

    def native_targets(self, split: str, sample_index: int) -> tuple[str, str, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def normalize_prediction_tensor(self, split: str, modality: str, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def split_size(self, split: str) -> int:
        raise NotImplementedError

    def display_inputs(self, split: str, sample_index: int, known_modalities: list[str]) -> dict[str, torch.Tensor]:
        del split, sample_index, known_modalities
        return {}

    def native_targets_batch(self, split: str, sample_indices: list[int]) -> list[tuple[str, str, dict[str, torch.Tensor]]]:
        return [self.native_targets(split=split, sample_index=sample_index) for sample_index in sample_indices]

    def display_inputs_batch(
        self,
        split: str,
        sample_indices: list[int],
        known_modalities: list[str],
    ) -> list[dict[str, torch.Tensor]]:
        return [
            self.display_inputs(split=split, sample_index=sample_index, known_modalities=known_modalities)
            for sample_index in sample_indices
        ]

    def set_num_workers(self, num_workers: int) -> None:
        self.num_workers = max(0, int(num_workers))

    def geometry_policy(self) -> str:
        return "adapter_specific"

    def close(self) -> None:
        return None



class AsynDiffusionAdapter(MethodAdapter):
    def __init__(
        self,
        lens_data_root: Path | None = None,
        lens_runs: tuple[str, ...] = DEFAULT_LENS_RUNS,
        nonlens_data_root: Path | None = None,
        nonlens_runs: tuple[str, ...] = (),
        checkpoint_path: Path | None = None,
    ) -> None:
        super().__init__("asyn-diffusion")
        self.root = REPO_ROOT / "asyn-diffusion"
        self.src = self.root / "src"
        _prepend_sys_path(self.src)
        self.lens_data_root = Path(lens_data_root) if lens_data_root is not None else DEFAULT_LENS_DATA_ROOT
        self.lens_runs = tuple(str(item) for item in lens_runs if str(item).strip()) or DEFAULT_LENS_RUNS
        self.nonlens_data_root = Path(nonlens_data_root) if nonlens_data_root is not None else None
        self.nonlens_runs = tuple(str(item) for item in nonlens_runs if str(item).strip())
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._config = None
        self._create_dataset = None
        self._datasets: dict[str, Any] = {}
        self._normalize_tensor = None
        self._build_device = None
        self._build_diffusion = None
        self._load_model_state = None
        self._default_checkpoint = None
        self._batch_to_device = None
        self._build_joint_tensor = None
        self._known_mask_from_modalities = None
        self._masked_from_known = None
        self._sample_with_mask = None
        self._split_joint_tensor = None
        self._denormalize_modalities = None
        self._resize_batch_to_native = None
        self._modality_order = None
        self._allowed_known = None
        self._device = None
        self._diffusion = None
        self._lens_tmpdir = None
        self._nonlens_tmpdir = None
        self._lens_config = None
        self._nonlens_configs: dict[str, dict[str, Any]] = {}
        self._resolved_checkpoint = None
        self._build_and_store_artifacts = None

    @staticmethod
    def _is_lens_split(split: str) -> bool:
        return str(split).strip().lower() == "lens"

    def _use_custom_nonlens_split(self, split: str) -> bool:
        return (
            not self._is_lens_split(split)
            and self.nonlens_data_root is not None
            and bool(self.nonlens_runs)
        )

    def _lens_dataset_config(self) -> dict[str, Any]:
        self._load()
        if self._lens_config is not None:
            return self._lens_config

        if not self.lens_data_root.exists():
            raise FileNotFoundError(f"Missing lens data root for --split lens: {self.lens_data_root}")

        self._lens_tmpdir = tempfile.TemporaryDirectory(prefix="asyndiff_lens_eval_", dir=None)
        runtime_config = _runtime_dataset_config(
            base_config=self.config,
            data_root=self.lens_data_root,
            manifest_name="manifest_lens.json",
            selected_runs=self.lens_runs,
            artifact_root=Path(self._lens_tmpdir.name),
            split="lens",
        )
        assert self._build_and_store_artifacts is not None
        self._build_and_store_artifacts(runtime_config)
        self._lens_config = runtime_config
        return runtime_config

    def _nonlens_dataset_config(self, split: str) -> dict[str, Any]:
        self._load()
        if split in self._nonlens_configs:
            return self._nonlens_configs[split]
        if self.nonlens_data_root is None or not self.nonlens_data_root.exists():
            raise FileNotFoundError(f"Missing nonlens data root for custom run evaluation: {self.nonlens_data_root}")
        assert self._build_and_store_artifacts is not None

        runtime_config = copy.deepcopy(self.config)
        runtime_config["data"] = copy.deepcopy(runtime_config.get("data", {}))
        runtime_config["evaluation"] = copy.deepcopy(runtime_config.get("evaluation", {}))
        if self._nonlens_tmpdir is None:
            self._nonlens_tmpdir = tempfile.TemporaryDirectory(prefix="asyndiff_nonlens_eval_", dir=None)
        artifact_root = Path(self._nonlens_tmpdir.name) / str(split)
        artifact_root.mkdir(parents=True, exist_ok=True)
        runtime_config["data"]["data_root"] = str(self.nonlens_data_root)
        runtime_config["data"]["selected_runs"] = list(self.nonlens_runs)
        runtime_config["data"]["index_path"] = str(artifact_root / "index.json")
        runtime_config["data"]["stats_path"] = str(artifact_root / "stats.json")
        runtime_config["data"]["visualization_subset_path"] = str(artifact_root / "visualization_subset.json")
        _configure_all_records_to_split(runtime_config, _normalize_nonlens_split(split))
        self._build_and_store_artifacts(runtime_config)
        self._nonlens_configs[split] = runtime_config
        return runtime_config

    def _load(self) -> None:
        if self._diffusion is not None:
            return
        module_root = self.root
        module_src = self.src
        module_prefix = "asyndiff"
        _prepend_sys_path(module_src)

        config_module = importlib.import_module(f"{module_prefix}.config")
        common_module = importlib.import_module(f"{module_prefix}.training.common")
        pipeline_module = importlib.import_module(f"{module_prefix}.training.pipeline")
        dataset_module = importlib.import_module(f"{module_prefix}.data.dataset")
        index_module = importlib.import_module(f"{module_prefix}.data.index")
        normalization_module = importlib.import_module(f"{module_prefix}.data.normalization")
        constants_module = importlib.import_module(f"{module_prefix}.constants")

        self._config = _stabilize_default_config(config_module.load_with_overrides(module_root / "configs" / "train.yaml"))
        self._create_dataset = common_module.create_dataset
        self._normalize_tensor = normalization_module.normalize_tensor
        self._build_device = common_module.build_device
        self._build_diffusion = pipeline_module._build_diffusion
        self._load_model_state = pipeline_module._load_model_state
        self._default_checkpoint = pipeline_module._default_checkpoint
        self._batch_to_device = pipeline_module._batch_to_device
        self._build_joint_tensor = pipeline_module._build_joint_tensor
        self._known_mask_from_modalities = pipeline_module._known_mask_from_modalities
        self._masked_from_known = pipeline_module._masked_from_known
        self._sample_with_mask = pipeline_module._sample_with_mask
        self._split_joint_tensor = pipeline_module._split_joint_tensor
        self._denormalize_modalities = dataset_module.denormalize_modalities
        self._resize_batch_to_native = common_module.resize_batch_to_native
        self._modality_order = constants_module.MODALITY_ORDER
        self._allowed_known = pipeline_module._ALLOWED_KNOWN
        self._build_and_store_artifacts = index_module.build_and_store_artifacts

        assert self._build_device is not None
        assert self._build_diffusion is not None
        assert self._load_model_state is not None
        assert self._default_checkpoint is not None
        assert self._config is not None
        checkpoint = self.checkpoint_path or self._default_checkpoint(self._config)
        checkpoint_fallback_reason = None
        if not checkpoint.exists():
            fallback = checkpoint.parent / checkpoint.name.replace("_best", "_last")
            if fallback.exists():
                checkpoint_fallback_reason = f"Requested checkpoint {checkpoint} is missing; using last checkpoint {fallback}."
                print(checkpoint_fallback_reason)
                checkpoint = fallback
            else:
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoint_cfg = payload.get("config")
        if isinstance(checkpoint_cfg, dict):
            runtime_config = copy.deepcopy(self._config)
            runtime_config["model"] = copy.deepcopy(checkpoint_cfg.get("model", runtime_config.get("model", {})))
            runtime_config["diffusion"] = copy.deepcopy(checkpoint_cfg.get("diffusion", runtime_config.get("diffusion", {})))
            self._config = runtime_config

        self._device = self._build_device()
        self._diffusion = self._build_diffusion(self._config, self._device, stats={})
        # Actual scoring-dataset normalization is selected in infer_examples.
        self._geometry_policy = "dataset_dependent"
        self._load_model_state(self._diffusion.model, payload["model"])
        self._diffusion.eval()
        self._resolved_checkpoint = checkpoint.resolve()
        self._checkpoint_fallback_reason = checkpoint_fallback_reason

    @property
    def config(self) -> dict[str, Any]:
        self._load()
        assert self._config is not None
        return self._config

    def _dataset(self, split: str):
        self._load()
        if split not in self._datasets:
            assert self._create_dataset is not None
            if self._is_lens_split(split):
                dataset_config = self._lens_dataset_config()
                dataset_split = "test"
            elif self._use_custom_nonlens_split(split):
                dataset_config = self._nonlens_dataset_config(split)
                dataset_split = _normalize_nonlens_split(split)
            else:
                dataset_config = self.config
                dataset_split = split
            self._datasets[split] = self._create_dataset(dataset_config, split=dataset_split, training=False)
        return self._datasets[split]

    def _batch_samples(self, split: str, sample_indices: list[int]) -> list[Any]:
        return _load_samples_for_indices(self._dataset(split), sample_indices, self.num_workers)

    def validate_request(self, known_modalities: list[str], target_modalities: list[str]) -> None:
        self._load()
        key = tuple(name for name in ALL_MODALITIES if name in set(known_modalities))
        assert self._allowed_known is not None
        if key not in self._allowed_known:
            allowed = [list(item) for item in sorted(self._allowed_known)]
            raise ValueError(f"asyn-diffusion only supports known modality combinations {allowed}; got {known_modalities}.")
        missing = {name for name in ALL_MODALITIES if name not in set(key)}
        invalid_targets = [name for name in target_modalities if name not in missing]
        if invalid_targets:
            raise ValueError(
                f"asyn-diffusion can only score missing target modalities {sorted(missing)} for known={known_modalities}; "
                f"got invalid targets {invalid_targets}."
            )

    def infer_example(self, split: str, sample_index: int, known_modalities: list[str], output_stem: Path) -> dict[str, torch.Tensor]:
        del output_stem
        return self.infer_examples(split=split, sample_indices=[sample_index], known_modalities=known_modalities, output_root=None)[0]

    def infer_examples(
        self,
        split: str,
        sample_indices: list[int],
        known_modalities: list[str],
        output_root: Path,
        *, sample_seeds: list[int] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        del output_root
        self._load()
        dataset = self._dataset(split)
        self._diffusion.clip_x0 = stats_use_signed_capped_asinh(dataset.stats) and bool(self.config["diffusion"].get("clip_x0", True))
        self._geometry_policy = dataset.spatial_mode + "_then_restore_native"
        samples = self._batch_samples(split, sample_indices)
        batch = {
            "record_id": [sample["record_id"] for sample in samples],
            "sample_id": [sample["sample_id"] for sample in samples],
            "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
            "inputs": _stack_tensor_dict(samples, "inputs"),
            "valid_masks": _stack_tensor_dict(samples, "valid_masks"),
            "native_targets": _stack_tensor_dict(samples, "native_targets"),
            "native_sizes": [sample["native_sizes"] for sample in samples],
        }
        assert self._batch_to_device is not None
        assert self._build_joint_tensor is not None
        assert self._known_mask_from_modalities is not None
        assert self._masked_from_known is not None
        assert self._sample_with_mask is not None
        assert self._split_joint_tensor is not None
        assert self._denormalize_modalities is not None
        assert self._resize_batch_to_native is not None
        assert self._diffusion is not None
        assert self._device is not None
        assert self._modality_order is not None
        device_batch = self._batch_to_device(batch, self._device)
        x_full = self._build_joint_tensor(device_batch["inputs"])
        require_finite_batch(x_full, batch["record_id"], "AsynDiff normalized input")
        masked_modalities = self._masked_from_known(known_modalities)
        known_mask = self._known_mask_from_modalities(known_modalities, x_full)
        pred_full = self._sample_with_mask(
            self.config,
            self._diffusion,
            x_full,
            known_mask,
            known_modalities,
            masked_modalities,
            self._device,
            sample_seeds=sample_seeds,
            record_ids=batch["record_id"],
        )
        require_finite_batch(pred_full, batch["record_id"], "AsynDiff sampled prediction")
        pred_modalities_norm = self._split_joint_tensor(pred_full)
        denorm_predictions = self._denormalize_modalities(pred_modalities_norm, dataset.stats)
        for name, tensor in denorm_predictions.items():
            require_finite_batch(tensor, batch["record_id"], f"AsynDiff denormalized {name}")
        native_predictions = {
            name: self._resize_batch_to_native(denorm_predictions[name], device_batch["native_targets"][name], spatial_mode=dataset.spatial_mode)
            for name in self._modality_order
        }
        return _split_prediction_batch(native_predictions)

    def infer_seeded_examples(self, split, sample_indices, known_modalities, output_root, sample_seeds):
        return self.infer_examples(split, sample_indices, known_modalities, output_root, sample_seeds=sample_seeds)

    def native_targets(self, split: str, sample_index: int) -> tuple[str, str, dict[str, torch.Tensor]]:
        sample = self._dataset(split)[sample_index]
        return (
            str(sample["record_id"]),
            str(sample["sample_id"]),
            {name: sample["native_targets"][name].cpu() for name in sample["native_targets"]},
        )

    def normalize_prediction_tensor(self, split: str, modality: str, tensor: torch.Tensor) -> torch.Tensor:
        self._load()
        assert self._normalize_tensor is not None
        stats = self._dataset(split).stats
        require_finite(tensor, f"AsynDiff {modality} native prediction before normalization")
        normalized = self._normalize_tensor(modality, tensor, stats).detach().cpu()
        require_finite(normalized, f"AsynDiff {modality} normalized prediction")
        return normalized

    def split_size(self, split: str) -> int:
        return len(self._dataset(split))

    def display_inputs(self, split: str, sample_index: int, known_modalities: list[str]) -> dict[str, torch.Tensor]:
        sample = self._dataset(split)[sample_index]
        return {
            name: sample["native_targets"][name].cpu()
            for name in known_modalities
            if name in sample["native_targets"]
        }

    def native_targets_batch(self, split: str, sample_indices: list[int]) -> list[tuple[str, str, dict[str, torch.Tensor]]]:
        rows: list[tuple[str, str, dict[str, torch.Tensor]]] = []
        for sample in self._batch_samples(split, sample_indices):
            rows.append(
                (
                    str(sample["record_id"]),
                    str(sample["sample_id"]),
                    {name: sample["native_targets"][name].cpu() for name in sample["native_targets"]},
                )
            )
        return rows

    def display_inputs_batch(
        self,
        split: str,
        sample_indices: list[int],
        known_modalities: list[str],
    ) -> list[dict[str, torch.Tensor]]:
        outputs: list[dict[str, torch.Tensor]] = []
        for sample in self._batch_samples(split, sample_indices):
            outputs.append(
                {
                    name: sample["native_targets"][name].cpu()
                    for name in known_modalities
                    if name in sample["native_targets"]
                }
            )
        return outputs

    def geometry_policy(self) -> str:
        return self._geometry_policy

    def close(self) -> None:
        if self._lens_tmpdir is not None:
            self._lens_tmpdir.cleanup()
        if self._nonlens_tmpdir is not None:
            self._nonlens_tmpdir.cleanup()
            self._lens_tmpdir = None
            self._lens_config = None



def _blank_modality(name: str, reference: torch.Tensor | None = None) -> torch.Tensor:
    if reference is not None:
        ref = _tensor_to_chw(reference).cpu()
        return torch.zeros_like(ref)
    channels = int(DEFAULT_MODALITY_CHANNELS[name])
    return torch.zeros((channels, 64, 64), dtype=torch.float32)


def _robust_limits(images: list[np.ndarray], lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    values = [np.asarray(image, dtype=np.float32)[np.isfinite(image)] for image in images]
    values = [item for item in values if item.size > 0]
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    vmin, vmax = np.percentile(merged, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(merged))
        vmax = float(np.max(merged))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, max(1.0, float(vmax))
    return float(vmin), float(vmax)


def _lupton_rgb(image_r: np.ndarray, image_g: np.ndarray, image_b: np.ndarray, stretch: float = 0.5, q: float = 8.0) -> np.ndarray:
    image_r = np.asarray(image_r, dtype=np.float32)
    image_g = np.asarray(image_g, dtype=np.float32)
    image_b = np.asarray(image_b, dtype=np.float32)
    intensity = (image_r + image_g + image_b) / 3.0

    epsilon = 1.0 / 2**23
    q = 0.1 if abs(q) < epsilon else min(float(q), 1.0e10)
    slope = 0.1 * 255.0 / np.arcsinh(0.1 * q)
    soften = q / float(stretch)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = np.where(intensity <= 0.0, 0.0, np.arcsinh(intensity * soften) * slope / intensity)

    channels = []
    for image in (image_r, image_g, image_b):
        channel = np.maximum(image, 0.0) * factor
        channel = np.where(np.isfinite(channel), channel, 0.0)
        channels.append(channel)

    peak = np.maximum.reduce(channels)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(peak > 255.0, 255.0 / peak, 1.0)
    channels = [np.clip(channel * scale, 0.0, 255.0) / 255.0 for channel in channels]
    return np.stack(channels, axis=-1).astype(np.float32)


def _modality_display(name: str, tensor: torch.Tensor) -> np.ndarray:
    array = _tensor_to_chw(tensor).detach().cpu().numpy().astype(np.float32)
    if name == "euclid":
        return array[0]
    if name in {"lsst", "roman"}:
        return _lupton_rgb(array[2], array[1], array[0])
    raise KeyError(name)


def _slugify(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value]
    collapsed = "".join(safe).strip("._")
    return collapsed or "sample"


def _render_example_panel(
    output_path: Path,
    method_name: str,
    split: str,
    sample_index: int,
    record_id: str,
    sample_id: str,
    known_modalities: list[str],
    target_modalities: list[str],
    inputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
) -> None:
    columns = len(ALL_MODALITIES) + 2 * len(target_modalities)
    fig, axes = plt.subplots(1, columns, figsize=(max(12, columns * 2.2), 3.2))
    if columns == 1:
        axes = [axes]
    else:
        axes = list(np.ravel(axes))

    ref_by_modality: dict[str, torch.Tensor | None] = {}
    for name in ALL_MODALITIES:
        reference = inputs.get(name)
        if reference is None:
            reference = targets.get(name)
        if reference is None:
            reference = predictions.get(name)
        ref_by_modality[name] = reference

    input_panels: list[tuple[str, np.ndarray]] = []
    for name in ALL_MODALITIES:
        tensor = inputs.get(name, _blank_modality(name, ref_by_modality[name]))
        input_panels.append((f"{name} input", _modality_display(name, tensor)))

    target_panels: list[tuple[str, np.ndarray]] = []
    pred_panels: list[tuple[str, np.ndarray]] = []
    for name in target_modalities:
        target_panels.append((f"{name} target", _modality_display(name, targets[name])))
        pred_panels.append((f"{name} pred", _modality_display(name, predictions[name])))

    panels = input_panels + target_panels + pred_panels
    scalar_images = [panel for _title, panel in panels if panel.ndim == 2]
    scalar_limits = _robust_limits(scalar_images) if scalar_images else (0.0, 1.0)

    for axis, (title, panel) in zip(axes, panels):
        if panel.ndim == 2:
            axis.imshow(panel, cmap="gray", origin="lower", vmin=scalar_limits[0], vmax=scalar_limits[1])
        else:
            axis.imshow(np.clip(panel, 0.0, 1.0), origin="lower")
        axis.set_title(title, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        (
            f"{method_name} | {split}[{sample_index}] | sample_id={sample_id}\n"
            f"known={','.join(known_modalities)} | targets={','.join(target_modalities)} | record_id={record_id}"
        ),
        fontsize=10,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_pred_normalized(
    output_path: Path,
    method_name: str,
    split: str,
    sample_index: int,
    record_id: str,
    sample_id: str,
    known_modalities: list[str],
    target_modalities: list[str],
    predictions_normalized: dict[str, torch.Tensor],
    ensemble_size: int,
) -> None:
    payload = {
        "method": method_name,
        "split": split,
        "sample_index": int(sample_index),
        "record_id": str(record_id),
        "sample_id": str(sample_id),
        "known_modalities": list(known_modalities),
        "target_modalities": list(target_modalities),
        "prediction_kind": "ensemble_mean",
        "ensemble_size": int(ensemble_size),
        "mean_domain": "native_flux",
        "scored_metrics": ["rmse", "psnr"],
        "pred_normalized": {
            name: tensor.detach().cpu()
            for name, tensor in predictions_normalized.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _save_pred_sample_normalized(
    output_path: Path,
    method_name: str,
    split: str,
    sample_index: int,
    record_id: str,
    sample_id: str,
    known_modalities: list[str],
    target_modality: str,
    seed: int,
    prediction_normalized: torch.Tensor,
) -> None:
    payload = {
        "method": method_name,
        "split": split,
        "sample_index": int(sample_index),
        "record_id": str(record_id),
        "sample_id": str(sample_id),
        "known_modalities": list(known_modalities),
        "target_modality": str(target_modality),
        "seed": int(seed),
        "seed_scheme": SEED_SCHEME,
        "prediction_kind": "single_sample",
        "draw_index": 0,
        "pred_normalized": prediction_normalized.detach().cpu(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _write_prediction_manifest(
    output_path: Path,
    kind: str,
    method_name: str,
    split: str,
    known_modalities: list[str],
    target_modalities: list[str],
    entries: list[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": 1,
        "kind": str(kind),
        "method": method_name,
        "split": split,
        "known_modalities": list(known_modalities),
        "target_modalities": list(target_modalities),
        "count": len(entries),
        "files": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")




def _build_adapter(
    lens_data_root: Path,
    lens_runs: tuple[str, ...],
    nonlens_data_root: Path | None,
    nonlens_runs: tuple[str, ...],
    checkpoint_path: Path | None,
) -> AsynDiffusionAdapter:
    return AsynDiffusionAdapter(
        lens_data_root=lens_data_root,
        lens_runs=lens_runs,
        nonlens_data_root=nonlens_data_root,
        nonlens_runs=nonlens_runs,
        checkpoint_path=checkpoint_path,
    )


def _evaluate_method(
    adapter: MethodAdapter,
    split: str,
    sample_indices: list[int],
    known_modalities: list[str],
    target_modalities: list[str],
    crps_samples: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    visualization_root: Path | None = None,
    pred_normalized_root: Path | None = None,
    pred_sample_normalized_root: Path | None = None,
) -> dict[str, Any]:
    if crps_samples < 2:
        raise ValueError("crps_samples must be >= 2 for fair CRPS.")
    if batch_size < 1 or not sample_indices:
        raise ValueError("Evaluation needs a positive batch_size and at least one sample.")
    # Every saved downstream sample has a saved point estimate matching RMSE/PSNR.
    if pred_sample_normalized_root is not None and pred_normalized_root is None:
        pred_normalized_root = Path(pred_sample_normalized_root) / "ensemble_mean"
    adapter.validate_request(known_modalities, target_modalities)
    adapter.set_num_workers(num_workers)
    examples: list[ExampleResult] = []
    per_target_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in target_modalities}
    visualization_paths: list[str] = []
    pred_normalized_paths: list[str] = []
    pred_sample_normalized_paths: list[str] = []
    pred_sample_manifest_entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"{adapter.name}_eval_", dir=None) as tmpdir:
        tmp_root = Path(tmpdir)
        total_examples = len(sample_indices)
        chunks = _chunked(sample_indices, batch_size)
        print(
            f"[{adapter.name}] split={split} known={','.join(known_modalities)} "
            f"targets={','.join(target_modalities)} examples={total_examples} batch_size={batch_size} num_workers={num_workers}"
        )
        processed_examples = 0
        for chunk_number, chunk_indices in enumerate(chunks, start=1):
            start_example = processed_examples + 1
            end_example = processed_examples + len(chunk_indices)
            print(
                f"[{adapter.name}] batch {chunk_number}/{len(chunks)} "
                f"examples={start_example}-{end_example} sample_indices={chunk_indices[0]}-{chunk_indices[-1]}"
            )
            metadata_rows = adapter.native_targets_batch(split=split, sample_indices=chunk_indices)
            display_rows = (
                adapter.display_inputs_batch(
                    split=split,
                    sample_indices=chunk_indices,
                    known_modalities=known_modalities,
                )
                if visualization_root is not None
                else [{} for _ in chunk_indices]
            )

            batch_ensembles: list[dict[str, list[torch.Tensor]]] = [
                {name: [] for name in target_modalities}
                for _ in chunk_indices
            ]
            for record_id, _, targets in metadata_rows:
                for modality, target in targets.items():
                    require_finite(target, f"{adapter.name} record_id={record_id} target={modality}")
            for repeat_idx in range(crps_samples):
                seeds = [sample_seed(seed, row[0], repeat_idx) for row in metadata_rows]
                batch_predictions = adapter.infer_seeded_examples(
                    split=split,
                    sample_indices=chunk_indices,
                    known_modalities=known_modalities,
                    output_root=tmp_root / f"{adapter.name}_batch_{chunk_number:03d}_{repeat_idx:02d}",
                    sample_seeds=seeds,
                )
                if len(batch_predictions) != len(chunk_indices):
                    raise ValueError(
                        f"{adapter.name} returned {len(batch_predictions)} predictions for {len(chunk_indices)} requested samples."
                    )
                for batch_offset, sample_index in enumerate(chunk_indices):
                    record_id, sample_id, native_targets = metadata_rows[batch_offset]
                    del sample_id
                    predictions = batch_predictions[batch_offset]
                    missing_targets = [name for name in target_modalities if name not in native_targets]
                    if missing_targets:
                        raise ValueError(
                            f"{adapter.name} does not expose native targets for {missing_targets}. "
                            f"Available targets: {sorted(native_targets.keys())}"
                        )
                    for modality in target_modalities:
                        if modality not in predictions:
                            raise ValueError(
                                f"{adapter.name} did not produce `{modality}` for known={known_modalities}. "
                                f"Available predictions: {sorted(predictions.keys())}"
                            )
                        require_finite(predictions[modality], f"{adapter.name} record_id={record_id} draw={repeat_idx} prediction={modality}")
                        pred_native = _resize_to_native(predictions[modality].cpu(), native_targets[modality], spatial_mode=adapter.prediction_spatial_mode(split))
                        batch_ensembles[batch_offset][modality].append(pred_native)

            for batch_offset, sample_index in enumerate(chunk_indices):
                record_id, sample_id, native_targets = metadata_rows[batch_offset]
                first_repeat_seed = sample_seed(seed, record_id, 0)
                metrics_by_target: dict[str, dict[str, float]] = {}
                point_predictions: dict[str, torch.Tensor] = {}
                point_predictions_normalized: dict[str, torch.Tensor] = {}
                example_pred_sample_normalized_paths: dict[str, str] = {}
                for modality in target_modalities:
                    ensemble = torch.stack(batch_ensembles[batch_offset][modality], dim=0)
                    target = native_targets[modality]
                    point_prediction = ensemble.mean(dim=0)
                    point_predictions[modality] = point_prediction
                    try:
                        metrics = _reconstruction_metrics(
                            point_prediction,
                            target,
                        )
                        metrics["crps"] = _crps_ensemble(ensemble, target)
                        require_finite(list(metrics.values()), f"{adapter.name} record_id={record_id} metrics={modality}")
                    except ValueError as error:
                        raise ValueError(f"{adapter.name} record_id={record_id} target={modality}: {error}") from error
                    metrics_by_target[modality] = metrics
                    per_target_metrics[modality].append(metrics)

                    if pred_sample_normalized_root is not None:
                        first_prediction_normalized = adapter.normalize_prediction_tensor(split, modality, ensemble[0])
                        method_dir = pred_sample_normalized_root / adapter.name
                        tensor_path = method_dir / f"{sample_index:05d}_{_slugify(sample_id)}_{modality}.pt"
                        _save_pred_sample_normalized(
                            output_path=tensor_path,
                            method_name=adapter.name,
                            split=split,
                            sample_index=sample_index,
                            record_id=record_id,
                            sample_id=sample_id,
                            known_modalities=known_modalities,
                            target_modality=modality,
                            seed=first_repeat_seed,
                            prediction_normalized=first_prediction_normalized,
                        )
                        path_value = str(tensor_path)
                        pred_sample_normalized_paths.append(path_value)
                        example_pred_sample_normalized_paths[modality] = path_value
                        pred_sample_manifest_entries.append(
                            {
                                "path": path_value,
                                "sample_index": int(sample_index),
                                "record_id": str(record_id),
                                "sample_id": str(sample_id),
                                "known_modalities": list(known_modalities),
                                "target_modality": str(modality),
                                "seed": int(first_repeat_seed),
                                "seed_scheme": SEED_SCHEME,
                                "draw_index": 0,
                            }
                        )

                example_visualization_path = None
                if visualization_root is not None:
                    method_dir = visualization_root / adapter.name
                    image_path = method_dir / f"{sample_index:05d}_{_slugify(sample_id)}.png"
                    _render_example_panel(
                        output_path=image_path,
                        method_name=adapter.name,
                        split=split,
                        sample_index=sample_index,
                        record_id=record_id,
                        sample_id=sample_id,
                        known_modalities=known_modalities,
                        target_modalities=target_modalities,
                        inputs=display_rows[batch_offset],
                        targets={name: native_targets[name] for name in target_modalities},
                        predictions=point_predictions,
                    )
                    example_visualization_path = str(image_path)
                    visualization_paths.append(example_visualization_path)

                example_pred_normalized_path = None
                if pred_normalized_root is not None:
                    point_predictions_normalized = {
                        modality: adapter.normalize_prediction_tensor(split, modality, prediction)
                        for modality, prediction in point_predictions.items()
                    }
                    method_dir = pred_normalized_root / adapter.name
                    tensor_path = method_dir / f"{sample_index:05d}_{_slugify(sample_id)}.pt"
                    _save_pred_normalized(
                        output_path=tensor_path,
                        method_name=adapter.name,
                        split=split,
                        sample_index=sample_index,
                        record_id=record_id,
                        sample_id=sample_id,
                        known_modalities=known_modalities,
                        target_modalities=target_modalities,
                        predictions_normalized=point_predictions_normalized,
                        ensemble_size=crps_samples,
                    )
                    example_pred_normalized_path = str(tensor_path)
                    pred_normalized_paths.append(example_pred_normalized_path)

                examples.append(
                    ExampleResult(
                        sample_index=sample_index,
                        record_id=record_id,
                        sample_id=sample_id,
                        metrics_by_target=metrics_by_target,
                        visualization_path=example_visualization_path,
                        pred_normalized_path=example_pred_normalized_path,
                        pred_sample_normalized_paths=example_pred_sample_normalized_paths or None,
                    )
                )
            processed_examples += len(chunk_indices)

    aggregate_by_target = {name: _mean_dict(rows) for name, rows in per_target_metrics.items()}
    overall_rows = [row for rows in per_target_metrics.values() for row in rows]
    pred_sample_manifest_path = None
    if pred_sample_normalized_root is not None:
        method_dir = pred_sample_normalized_root / adapter.name
        pred_sample_manifest_path = str(method_dir / "manifest.json")
        _write_prediction_manifest(
            output_path=Path(pred_sample_manifest_path),
            kind="pred_sample_normalized",
            method_name=adapter.name,
            split=split,
            known_modalities=known_modalities,
            target_modalities=target_modalities,
            entries=pred_sample_manifest_entries,
        )
    example_payloads: list[dict[str, Any]] = []
    for item in examples:
        payload = {
            "sample_index": item.sample_index,
            "record_id": item.record_id,
            "sample_id": item.sample_id,
            "metrics": item.metrics_by_target,
        }
        if item.visualization_path is not None:
            payload["visualization_path"] = item.visualization_path
        if item.pred_normalized_path is not None:
            payload["pred_normalized_path"] = item.pred_normalized_path
        if item.pred_sample_normalized_paths:
            payload["pred_sample_normalized_paths"] = item.pred_sample_normalized_paths
        example_payloads.append(payload)
    return {
        "method": adapter.name,
        "geometry_policy": adapter.geometry_policy(),
        "clip_x0": getattr(getattr(adapter, "_diffusion", None), "clip_x0", None),
        "split": split,
        "input_modalities": known_modalities,
        "target_modalities": target_modalities,
        "sample_indices": list(sample_indices),
        "crps_samples": crps_samples,
        "crps_estimator": "fair",
        "metric_prediction_kind": {"rmse": "ensemble_mean", "psnr": "ensemble_mean", "crps": "ensemble"},
        "downstream_prediction_kind": "single_sample",
        "downstream_draw_index": 0,
        "psnr_protocol": PSNR_PROTOCOL,
        "seed": int(seed),
        "seed_scheme": SEED_SCHEME,
        "batch_size": batch_size,
        "num_workers": int(num_workers),
        "examples": example_payloads,
        "aggregate": {
            "by_target": aggregate_by_target,
            "overall": _mean_dict(overall_rows),
        },
        "visualizations": visualization_paths,
        "pred_normalized_root": str(pred_normalized_root) if pred_normalized_root is not None else None,
        "pred_normalized_paths": pred_normalized_paths,
        "pred_sample_normalized_paths": pred_sample_normalized_paths,
        "pred_sample_normalized_manifest": pred_sample_manifest_path,
    }


def _print_summary(results: dict[str, Any]) -> None:
    print(f"split={results['split']}")
    print(f"input_modalities={','.join(results['input_modalities'])}")
    print(f"target_modalities={','.join(results['target_modalities'])}")
    sample_indices = results.get("sample_indices")
    if sample_indices is None:
        print("sample_indices=per-method")
    else:
        print(f"sample_indices={sample_indices}")
    print(f"crps_samples={results['crps_samples']}")
    print(f"batch_size={results['batch_size']}")
    print(f"num_workers={results['num_workers']}")
    print("geometry_policies:")
    for method_name in results["methods"]:
        print(f"  {method_name}: {results['methods'][method_name].get('geometry_policy', 'unknown')}")
    print()
    header = (
        f"{'method':<12} {'target':<8} "
        f"{'rmse':>10} {'psnr':>10} {'crps':>10}"
    )
    print(header)
    print("-" * len(header))
    for method_name in results["methods"]:
        aggregate = results["methods"][method_name]["aggregate"]["by_target"]
        for modality, metrics in aggregate.items():
            print(
                f"{method_name:<12} {modality:<8} "
                f"{metrics.get('rmse', float('nan')):>10.6f} "
                f"{metrics.get('psnr', float('nan')):>10.6f} "
                f"{metrics.get('crps', float('nan')):>10.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the released AsynDiff model on COSMIO and report RMSE, PSNR, and CRPS."
    )
    parser.add_argument("--split", default="val", help="Use lens for rare events or imputation for regular run_007.")
    parser.add_argument("--checkpoint", default=None, help="Optional AsynDiff checkpoint path.")
    parser.add_argument("--lens-data-root", default=str(DEFAULT_LENS_DATA_ROOT))
    parser.add_argument("--lens-runs", default="run_000")
    parser.add_argument("--nonlens-data-root", default=None)
    parser.add_argument("--nonlens-runs", default=None)
    parser.add_argument("--known", required=True, help="Comma-separated observed modalities, e.g. lsst,euclid.")
    parser.add_argument("--targets", default=None, help="Missing modalities; defaults to complement of --known.")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--crps-samples", type=int, default=5, help="Number of draws for fair CRPS (>= 2).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=str(REPO_ROOT / "evaluation_outputs" / "asyndiff_metrics.json"))
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visualization-root", default=str(REPO_ROOT / "evaluation_outputs" / "imputation_visualizations"))
    parser.add_argument("--save-pred-normalized", action="store_true", help="Save the ensemble mean scored by RMSE/PSNR (also enabled when saving downstream samples).")
    parser.add_argument("--pred-normalized-root", default=str(REPO_ROOT / "evaluation_outputs" / "pred_normalized"))
    parser.add_argument("--save-pred-sample-normalized", action="store_true", help="Save draw 0 for downstream AEE/classification and also save the scored ensemble mean.")
    parser.add_argument("--pred-sample-normalized-root", default=str(REPO_ROOT / "evaluation_outputs" / "pred_sample_normalized"))
    args = parser.parse_args()

    if _is_imputation_split(args.split):
        args.nonlens_data_root = args.nonlens_data_root or str(DEFAULT_NONLENS_DATA_ROOT)
        args.nonlens_runs = args.nonlens_runs or ",".join(DEFAULT_IMPUTATION_RUNS)
        args.num_examples = args.num_examples or DEFAULT_IMPUTATION_SET_SIZE

    known_modalities = _parse_modalities(args.known)
    target_modalities = (
        _parse_modalities(args.targets)
        if args.targets is not None
        else [name for name in ALL_MODALITIES if name not in set(known_modalities)]
    )
    if not target_modalities:
        raise ValueError("Target modalities resolved to an empty set.")
    overlap = sorted(set(known_modalities) & set(target_modalities))
    if overlap:
        raise ValueError(f"Known and target modalities must be disjoint; overlap={overlap}.")
    if args.crps_samples < 2:
        raise ValueError("--crps-samples must be >= 2 for fair CRPS.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if args.num_examples is not None and args.num_examples < 1:
        raise ValueError("--num-examples must be >= 1.")

    lens_runs = tuple(_parse_run_ids(args.lens_runs)) or DEFAULT_LENS_RUNS
    nonlens_runs = tuple(_parse_run_ids(args.nonlens_runs))
    custom_nonlens = args.nonlens_data_root is not None or bool(nonlens_runs)
    if custom_nonlens:
        if str(args.split).strip().lower() == "lens":
            raise ValueError("Custom nonlens arguments cannot be used with --split lens.")
        if args.nonlens_data_root is None or not nonlens_runs:
            raise ValueError("Custom nonlens evaluation requires both --nonlens-data-root and --nonlens-runs.")

    adapter = _build_adapter(
        lens_data_root=Path(args.lens_data_root),
        lens_runs=lens_runs,
        nonlens_data_root=Path(args.nonlens_data_root) if args.nonlens_data_root else None,
        nonlens_runs=nonlens_runs,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )
    tag = f"{args.split}_known-{'-'.join(known_modalities)}_targets-{'-'.join(target_modalities)}"
    visualization_root = Path(args.visualization_root) / tag if args.save_visualizations else None
    pred_normalized_root = (
        Path(args.pred_normalized_root) / tag
        if args.save_pred_normalized or args.save_pred_sample_normalized else None
    )
    pred_sample_normalized_root = (
        Path(args.pred_sample_normalized_root) / tag if args.save_pred_sample_normalized else None
    )

    try:
        adapter.validate_request(known_modalities, target_modalities)
        sample_indices = _resolve_sample_indices(
            int(args.start_index),
            int(args.num_examples) if args.num_examples is not None else None,
            adapter.split_size(args.split),
        )
        method_result = _evaluate_method(
            adapter=adapter,
            split=args.split,
            sample_indices=sample_indices,
            known_modalities=known_modalities,
            target_modalities=target_modalities,
            crps_samples=int(args.crps_samples),
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            visualization_root=visualization_root,
            pred_normalized_root=pred_normalized_root,
            pred_sample_normalized_root=pred_sample_normalized_root,
        )
    finally:
        adapter.close()

    results: dict[str, Any] = {
        "split": args.split,
        "input_modalities": known_modalities,
        "target_modalities": target_modalities,
        "sample_indices": sample_indices,
        "requested_num_examples": int(args.num_examples) if args.num_examples is not None else None,
        "start_index": int(args.start_index),
        "crps_samples": int(args.crps_samples),
        "crps_estimator": "fair",
        "metric_prediction_kind": {"rmse": "ensemble_mean", "psnr": "ensemble_mean", "crps": "ensemble"},
        "downstream_prediction_kind": "single_sample",
        "downstream_draw_index": 0,
        "psnr_protocol": PSNR_PROTOCOL,
        "seed": int(args.seed),
        "seed_scheme": SEED_SCHEME,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "visualization_root": str(visualization_root) if visualization_root else None,
        "pred_normalized_root": str(pred_normalized_root) if pred_normalized_root else None,
        "pred_sample_normalized_root": str(pred_sample_normalized_root) if pred_sample_normalized_root else None,
        "requested_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint": str(adapter._resolved_checkpoint) if getattr(adapter, "_resolved_checkpoint", None) else None,
        "checkpoint_fallback_reason": getattr(adapter, "_checkpoint_fallback_reason", None),
        "lens_data_root": str(args.lens_data_root) if str(args.split).lower() == "lens" else None,
        "lens_runs": list(lens_runs) if str(args.split).lower() == "lens" else None,
        "nonlens_data_root": str(args.nonlens_data_root) if custom_nonlens else None,
        "nonlens_runs": list(nonlens_runs) if custom_nonlens else None,
        "methods": {"asyn-diffusion": method_result},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    _print_summary(results)
    print(f"\n{output_path}")


if __name__ == "__main__":
    main()
