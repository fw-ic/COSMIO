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

import math

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    SummaryWriter = None  # type: ignore[assignment]

from asyndiff.constants import MODALITY_CHANNELS, MODALITY_ORDER
from asyndiff.data.dataset import denormalize_modalities, denormalize_tensor
from asyndiff.data.index import artifacts_need_rebuild, build_and_store_artifacts
from asyndiff.ops.random import SEED_SCHEME, sample_seed, sample_generators
from asyndiff.ops.validation import require_finite, require_finite_batch
from asyndiff.data.normalization import stats_use_signed_capped_asinh
from asyndiff.data.protocol import stack_valid_masks
from asyndiff.models import GaussianDiffusion, UNet
from asyndiff.ops.metrics import PSNR_PROTOCOL, batch_reconstruction_metrics, merge_metric_dict
from asyndiff.ops.metrics import fair_crps_ensemble as _crps_ensemble
from asyndiff.ops.visualization import append_history, render_training_curves, save_imputation_epoch_panel
from asyndiff.training.common import (
    build_device,
    build_run_directories,
    create_dataset,
    create_loader,
    gradient_norm,
    resize_batch_to_native,
    save_checkpoint,
    set_seed,
)


def _create_summary_writer(log_dir: Path) -> "SummaryWriter":
    if SummaryWriter is None:
        raise ModuleNotFoundError(
            "tensorboard is required for training asyn-diffusion, but it is not installed in the current environment."
        )
    return SummaryWriter(log_dir=str(log_dir))

_MODALITY_INDEX = {name: idx for idx, name in enumerate(MODALITY_ORDER)}

# Required masking policy:
# 1) mask roman           -> known {lsst, euclid}
# 2) mask roman + euclid  -> known {lsst}
# 3) mask roman + lsst    -> known {euclid}
# 4) mask euclid          -> known {lsst, roman}
_MASK_PATTERNS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("both", ("lsst", "euclid"), ("roman",)),
    ("lsst_only", ("lsst",), ("roman", "euclid")),
    ("euclid_only", ("euclid",), ("roman", "lsst")),
    ("lsst_roman", ("lsst", "roman"), ("euclid",)),
)
_ALLOWED_KNOWN = {pattern[1] for pattern in _MASK_PATTERNS}
_PATTERN_INDEX_BY_KNOWN = {pattern[1]: idx for idx, pattern in enumerate(_MASK_PATTERNS)}
_DEFAULT_TRAIN_KNOWN: tuple[tuple[str, ...], ...] = (
    ("lsst", "euclid"),
    ("lsst",),
    ("euclid",),
)


def _normalize_head_mode(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "single": "single",
        "single_head": "single",
        "one": "single",
        "one_head": "single",
        "multi": "multi",
        "multi_head": "multi",
        "multihead": "multi",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model.head_mode={value!r}. Expected 'single' or 'multi'.")
    return aliases[normalized]


def _configured_num_heads(model_cfg: dict[str, Any]) -> int:
    head_mode = model_cfg.get("head_mode")
    if head_mode is not None:
        return 1 if _normalize_head_mode(head_mode) == "single" else len(_MASK_PATTERNS)

    num_heads = int(model_cfg.get("num_heads", len(_MASK_PATTERNS)))
    if num_heads not in {1, len(_MASK_PATTERNS)}:
        raise ValueError(
            "Asyn diffusion only supports a single shared head or one head per mask pattern; "
            f"got model.num_heads={num_heads}. Use model.head_mode='single' or 'multi'."
        )
    return num_heads


def _default_checkpoint(config: dict[str, Any]) -> Path:
    return (
        Path(config["experiment"]["output_root"])
        / config["experiment"]["name"]
        / "asyn_diffusion"
        / "checkpoints"
        / "asyndiff_best.pt"
    )


def _modality_slices() -> dict[str, slice]:
    offset = 0
    slices: dict[str, slice] = {}
    for name in MODALITY_ORDER:
        width = int(MODALITY_CHANNELS[name])
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices


def _build_joint_tensor(modalities: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([modalities[name] for name in MODALITY_ORDER], dim=1)


def _split_joint_tensor(stacked: torch.Tensor) -> dict[str, torch.Tensor]:
    slices = _modality_slices()
    return {name: stacked[:, slices[name]] for name in MODALITY_ORDER}


def _validate_known_modalities(values: list[str]) -> list[str]:
    cleaned = [item.strip() for item in values if item.strip()]
    if not cleaned:
        raise ValueError("Known modalities must include at least one modality name.")

    invalid = [item for item in cleaned if item not in MODALITY_ORDER]
    if invalid:
        raise ValueError(f"Invalid known modalities: {invalid}. Expected subset of {MODALITY_ORDER}.")

    ordered = [name for name in MODALITY_ORDER if name in set(cleaned)]
    key = tuple(ordered)
    if key not in _ALLOWED_KNOWN:
        allowed = [list(item) for item in sorted(_ALLOWED_KNOWN)]
        raise ValueError(
            "Asyn diffusion only supports known modality combinations "
            f"{allowed}; got {ordered}."
        )
    return ordered


def _known_label(known_modalities: list[str]) -> str:
    cleaned = tuple(name for name in MODALITY_ORDER if name in set(known_modalities))
    if cleaned == ("lsst", "euclid"):
        return "both"
    if cleaned == ("lsst",):
        return "lsst_only"
    if cleaned == ("euclid",):
        return "euclid_only"
    if cleaned == ("lsst", "roman"):
        return "lsst_roman"
    return "_".join(cleaned)


def _known_combinations(config: dict[str, Any]) -> list[tuple[str, list[str]]]:
    combos = config["evaluation"].get("known_combinations", [["lsst", "euclid"], ["lsst"], ["euclid"]])
    outputs: list[tuple[str, list[str]]] = []
    for combo in combos:
        known = _validate_known_modalities(list(combo))
        outputs.append((_known_label(known), known))
    return outputs


def _train_pattern_specs(config: dict[str, Any]) -> list[tuple[int, tuple[str, tuple[str, ...], tuple[str, ...]]]]:
    configured = config.get("train", {}).get("known_combinations")
    combos = configured if configured is not None else [list(item) for item in _DEFAULT_TRAIN_KNOWN]
    specs: list[tuple[int, tuple[str, tuple[str, ...], tuple[str, ...]]]] = []
    for combo in combos:
        known = tuple(_validate_known_modalities(list(combo)))
        pattern_index = int(_PATTERN_INDEX_BY_KNOWN[known])
        specs.append((pattern_index, _MASK_PATTERNS[pattern_index]))
    if not specs:
        raise ValueError("train.known_combinations must include at least one supported known-modality combination.")
    return specs


def _masked_from_known(known_modalities: list[str]) -> list[str]:
    known_set = set(known_modalities)
    return [name for name in MODALITY_ORDER if name not in known_set]


def _head_index_from_known(known_modalities: list[str]) -> int:
    key = tuple(name for name in MODALITY_ORDER if name in set(known_modalities))
    if key not in _PATTERN_INDEX_BY_KNOWN:
        raise ValueError(f"Unsupported known modalities for multi-head routing: {list(key)}")
    return int(_PATTERN_INDEX_BY_KNOWN[key])


def _head_index_for_model(model: UNet, known_modalities: list[str]) -> int | None:
    if int(getattr(model, "num_heads", 1)) == 1:
        return None
    return _head_index_from_known(known_modalities)


def _pattern_head_index_for_model(model: UNet, pattern_ids: torch.Tensor) -> torch.Tensor | None:
    if int(getattr(model, "num_heads", 1)) == 1:
        return None
    return pattern_ids


def _known_mask_from_modalities(
    known_modalities: list[str],
    reference: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros_like(reference)
    channel_slices = _modality_slices()
    for name in known_modalities:
        mask[:, channel_slices[name]] = 1.0
    return mask


def _channel_time_mask_from_modalities(masked_modalities: list[str], reference: torch.Tensor) -> torch.Tensor:
    batch, channels = int(reference.shape[0]), int(reference.shape[1])
    mask = torch.zeros((batch, channels), device=reference.device, dtype=reference.dtype)
    channel_slices = _modality_slices()
    for name in masked_modalities:
        mask[:, channel_slices[name]] = 1.0
    return mask


def _modality_time_mask_from_modalities(masked_modalities: list[str], batch: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((batch, len(MODALITY_ORDER)), device=device, dtype=torch.float32)
    for name in masked_modalities:
        mask[:, _MODALITY_INDEX[name]] = 1.0
    return mask


def _sample_train_async_targets(
    x_full: torch.Tensor,
    timesteps: int,
    pattern_specs: list[tuple[int, tuple[str, tuple[str, ...], tuple[str, ...]]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample per-instance async timesteps and masked-only loss mask.

    - Unmasked modalities always get timestep 0.
    - One/two masked modalities get non-zero timestep in [1, T-1].
    - For the 2-masked patterns, both masked modalities share the same non-zero timestep.
    """

    if timesteps <= 1:
        raise ValueError("diffusion.timesteps must be >= 2 for async masking.")

    batch = int(x_full.shape[0])
    device = x_full.device
    channel_slices = _modality_slices()
    if pattern_specs is None:
        pattern_specs = [(idx, pattern) for idx, pattern in enumerate(_MASK_PATTERNS)]

    sampled_pattern_specs = torch.randint(0, len(pattern_specs), (batch,), device=device, dtype=torch.long)
    pattern_ids = torch.empty((batch,), device=device, dtype=torch.long)
    shared_nonzero_t = torch.randint(1, timesteps, (batch,), device=device, dtype=torch.long)

    modality_t = torch.zeros((batch, len(MODALITY_ORDER)), device=device, dtype=torch.long)
    channel_t = torch.zeros((batch, x_full.shape[1]), device=device, dtype=torch.long)
    loss_mask = torch.zeros_like(x_full)

    for local_idx, (pattern_idx, (_label, _known, masked_modalities)) in enumerate(pattern_specs):
        rows = sampled_pattern_specs == local_idx
        if not bool(rows.any()):
            continue

        row_idx = rows.nonzero(as_tuple=False).squeeze(1)
        pattern_ids[row_idx] = pattern_idx
        t_values = shared_nonzero_t[row_idx]

        for name in masked_modalities:
            modality_t[row_idx, _MODALITY_INDEX[name]] = t_values
            channel_slice = channel_slices[name]
            channel_t[row_idx, channel_slice] = t_values.unsqueeze(1)
            loss_mask[row_idx, channel_slice] = 1.0

    return pattern_ids, modality_t, channel_t, loss_mask


def _sample_with_mask(
    config: dict[str, Any],
    diffusion: GaussianDiffusion,
    x_full: torch.Tensor,
    known_mask: torch.Tensor,
    known_modalities: list[str],
    masked_modalities: list[str],
    device: torch.device,
    *, sample_seeds: list[int] | None = None, record_ids: list[str] | None = None,
) -> torch.Tensor:
    try:
        require_finite(x_full, "AsynDiff sampling input")
        generators = sample_generators(sample_seeds, device)
        eval_cfg = config["evaluation"]
        inference_steps = int(eval_cfg.get("inference_steps", 200))
        fix_mode = str(eval_cfg.get("fix_mode", "noised"))
        ddim_eta = float(eval_cfg.get("ddim_eta", 1.0))

        channel_time_mask = _channel_time_mask_from_modalities(masked_modalities, x_full)
        model_time_mask = _modality_time_mask_from_modalities(masked_modalities, int(x_full.shape[0]), device)
        head_index = _head_index_for_model(diffusion.model, known_modalities)

        if inference_steps > 0:
            steps = diffusion.build_linear_steps(int(diffusion.timesteps), inference_steps)
            assert steps is not None
            steps = steps.to(device)
            return diffusion.sample_ddim_with_mask(
                x_full,
                known_mask,
                device=device,
                fix_mode=fix_mode,
                steps=steps,
                eta=ddim_eta,
                time_mask=channel_time_mask,
                model_time_mask=model_time_mask,
                head_index=head_index,
                generators=generators,
            )

        return diffusion.sample_with_mask(
            x_full,
            known_mask,
            device=device,
            fix_mode=fix_mode,
            time_mask=channel_time_mask,
            model_time_mask=model_time_mask,
            head_index=head_index,
            generators=generators,
        )
    except ValueError as error:
        raise ValueError(f"AsynDiff known={known_modalities} record_ids={record_ids}: {error}") from error


def _build_diffusion(config: dict[str, Any], device: torch.device, *, stats: dict[str, Any] | None = None) -> GaussianDiffusion:
    model_cfg = config["model"]
    diffusion_cfg = config["diffusion"]
    if stats is None:
        from asyndiff.config import load_json
        stats = load_json(config["data"]["stats_path"])
    bounded = stats_use_signed_capped_asinh(stats)

    total_channels = sum(int(MODALITY_CHANNELS[name]) for name in MODALITY_ORDER)
    in_channels = int(model_cfg.get("in_channels", total_channels) or total_channels)
    out_channels = int(model_cfg.get("out_channels", total_channels) or total_channels)
    if in_channels != total_channels or out_channels != total_channels:
        raise ValueError(
            "Asyn diffusion expects in/out channels equal to concatenated modalities; "
            f"got in={in_channels}, out={out_channels}, expected={total_channels}."
        )
    num_heads = _configured_num_heads(model_cfg)

    unet = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=int(model_cfg["base_channels"]),
        channel_mults=tuple(int(value) for value in model_cfg["channel_mults"]),
        num_res_blocks=int(model_cfg["num_res_blocks"]),
        time_dim=int(model_cfg["time_dim"]),
        num_modalities=int(model_cfg.get("num_modalities", len(MODALITY_ORDER))),
        num_heads=num_heads,
    ).to(device)

    return GaussianDiffusion(
        model=unet,
        timesteps=int(diffusion_cfg["timesteps"]),
        beta_schedule=str(diffusion_cfg["beta_schedule"]),
        beta_start=float(diffusion_cfg["beta_start"]),
        beta_end=float(diffusion_cfg["beta_end"]),
        objective=str(diffusion_cfg.get("objective", "eps")),
        clip_x0=bounded and bool(diffusion_cfg.get("clip_x0", True)),
    ).to(device)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    for field in ("inputs", "native_targets", "valid_masks"):
        for name, tensor in batch[field].items():
            require_finite_batch(tensor, batch["record_id"], f"AsynDiff {field}/{name}")
    return {
        "record_id": batch["record_id"],
        "sample_id": batch["sample_id"],
        "label": batch["label"].to(device, non_blocking=True),
        "inputs": {name: tensor.to(device, non_blocking=True) for name, tensor in batch["inputs"].items()},
        "valid_masks": {
            name: tensor.to(device, non_blocking=True) for name, tensor in batch["valid_masks"].items()
        },
        "native_targets": {
            name: tensor.to(device, non_blocking=True) for name, tensor in batch["native_targets"].items()
        },
        "native_sizes": batch["native_sizes"],
    }


def _load_model_state(model: UNet, state_dict: dict[str, torch.Tensor]) -> None:
    if "final_conv.weight" not in state_dict:
        model.load_state_dict(state_dict)
        return

    upgraded = dict(state_dict)
    weight = upgraded.pop("final_conv.weight")
    bias = upgraded.pop("final_conv.bias")
    for head_idx in range(int(model.num_heads)):
        upgraded[f"final_convs.{head_idx}.weight"] = weight.clone()
        upgraded[f"final_convs.{head_idx}.bias"] = bias.clone()
    model.load_state_dict(upgraded)


@torch.no_grad()
def _evaluate(
    config: dict[str, Any],
    dataset,
    loader,
    diffusion: GaussianDiffusion,
    device: torch.device,
) -> dict[str, float]:
    diffusion.eval()
    max_batches = config["evaluation"].get("max_val_batches")
    combinations = _known_combinations(config)
    num_crps_samples = int(config["evaluation"].get("num_crps_samples", 5))
    if num_crps_samples < 2:
        raise ValueError("evaluation.num_crps_samples must be >= 2 for fair CRPS.")
    stats = dataset.stats
    per_target_metrics: dict[tuple[str, str], list[dict[str, float]]] = {}
    per_target_counts: dict[tuple[str, str], list[int]] = {}
    evaluated_sample_count = 0

    for batch_index, batch in enumerate(loader):
        evaluated_sample_count += len(batch["record_id"])
        device_batch = _batch_to_device(batch, device)
        x_full = _build_joint_tensor(device_batch["inputs"])

        for label, known in combinations:
            masked = _masked_from_known(known)
            metric_target = "roman" if "roman" in masked else masked[0]
            known_mask = _known_mask_from_modalities(known, x_full)
            ensemble_samples: list[torch.Tensor] = []
            for draw in range(num_crps_samples):
                seeds = [sample_seed(config["experiment"]["seed"], record_id, draw) for record_id in batch["record_id"]]
                pred_full = _sample_with_mask(config, diffusion, x_full, known_mask, known, masked, device, sample_seeds=seeds, record_ids=batch["record_id"])
                require_finite_batch(pred_full, batch["record_id"], f"AsynDiff validation {label} draw={draw}")
                pred_target_norm = _split_joint_tensor(pred_full)[metric_target]
                pred_target = denormalize_tensor(metric_target, pred_target_norm, stats)
                require_finite_batch(pred_target, batch["record_id"], f"AsynDiff denormalized {metric_target}")
                pred_target_native = resize_batch_to_native(pred_target, device_batch["native_targets"][metric_target], spatial_mode=dataset.spatial_mode)
                ensemble_samples.append(pred_target_native)

            ensemble = torch.stack(ensemble_samples, dim=0)
            ensemble_mean = ensemble.mean(dim=0)
            metrics_row = batch_reconstruction_metrics(ensemble_mean, device_batch["native_targets"][metric_target])
            metrics_row["crps"] = _crps_ensemble(ensemble, device_batch["native_targets"][metric_target])
            per_target_metrics.setdefault((metric_target, label), []).append(metrics_row)
            per_target_counts.setdefault((metric_target, label), []).append(len(batch["record_id"]))

        if max_batches is not None and (batch_index + 1) >= int(max_batches):
            break

    metrics: dict[str, float] = {}
    means_by_target: dict[str, dict[str, list[float]]] = {}
    overall_means = {"mse": [], "psnr": [], "crps": []}
    for (metric_target, label), rows in per_target_metrics.items():
        merged = merge_metric_dict(rows)
        if rows:
            # PSNR is an equal-weight mean over images, including a short last batch.
            merged["psnr"] = float(np.average([row["psnr"] for row in rows], weights=per_target_counts[(metric_target, label)]))
        if not merged:
            merged = {"mse": float("nan"), "psnr": float("nan"), "crps": float("nan")}
        for key, value in merged.items():
            metrics[f"{metric_target}_{label}_{key}"] = value
            if not np.isnan(value):
                means_by_target.setdefault(metric_target, {}).setdefault(key, []).append(value)
                overall_means[key].append(value)

    for metric_target, target_means in means_by_target.items():
        for key, values in target_means.items():
            metrics[f"{metric_target}_mean_{key}"] = float(sum(values) / max(1, len(values))) if values else float("nan")
    for key, values in overall_means.items():
        metrics[f"mean_{key}"] = float(sum(values) / max(1, len(values))) if values else float("nan")

    preferred_val_loss = metrics.get("roman_mean_crps", float("nan"))
    if np.isnan(preferred_val_loss):
        preferred_val_loss = metrics.get("mean_crps", float("nan"))
    metrics["val_loss"] = preferred_val_loss
    metrics["evaluated_sample_count"] = evaluated_sample_count
    return metrics


@torch.no_grad()
def _render_epoch_media(
    config: dict[str, Any],
    dataset,
    loader,
    diffusion: GaussianDiffusion,
    device: torch.device,
    epoch: int,
    media_dir: Path,
    split_label: str,
) -> None:
    if len(dataset) == 0:
        return

    diffusion.eval()
    batch = next(iter(loader))
    device_batch = _batch_to_device(batch, device)
    x_full = _build_joint_tensor(device_batch["inputs"])

    known_modalities = _validate_known_modalities(
        [str(item) for item in config["evaluation"].get("visualization_known_modalities", ["lsst", "euclid"])]
    )
    masked_modalities = _masked_from_known(known_modalities)
    known_mask = _known_mask_from_modalities(known_modalities, x_full)
    pred_full = _sample_with_mask(
        config,
        diffusion,
        x_full,
        known_mask,
        known_modalities,
        masked_modalities,
        device,
        sample_seeds=[sample_seed(config["experiment"]["seed"], rid, 0) for rid in batch["record_id"]],
        record_ids=batch["record_id"],
    )
    require_finite_batch(pred_full, batch["record_id"], "AsynDiff prediction")

    pred_modalities_norm = _split_joint_tensor(pred_full)
    denorm_predictions = denormalize_modalities(pred_modalities_norm, dataset.stats)
    for name, tensor in denorm_predictions.items():
        require_finite_batch(tensor, batch["record_id"], f"AsynDiff denormalized {name}")
    display_inputs = denormalize_modalities(device_batch["inputs"], dataset.stats)
    known_set = set(known_modalities)
    for name in MODALITY_ORDER:
        if name not in known_set:
            display_inputs[name] = torch.zeros_like(display_inputs[name])

    torch.save(
        {
            "seed_scheme": SEED_SCHEME,
            "seed": int(config["experiment"]["seed"]),
            "spatial_mode": dataset.spatial_mode,
            "clip_x0": diffusion.clip_x0,
            "record_id": batch["record_id"],
            "sample_id": batch["sample_id"],
            "split": split_label,
            "known_modalities": known_modalities,
            "masked_modalities": masked_modalities,
            "predictions": {name: tensor.cpu() for name, tensor in denorm_predictions.items()},
            "targets": {name: batch["native_targets"][name].cpu() for name in MODALITY_ORDER},
        },
        media_dir / f"{split_label}_imputation_epoch_{epoch:03d}_samples.pt",
    )

    save_imputation_epoch_panel(
        output_path=media_dir / f"{split_label}_imputation_epoch_{epoch:03d}.png",
        known_modalities=known_modalities,
        inputs={name: display_inputs[name].cpu() for name in MODALITY_ORDER},
        targets={name: batch["native_targets"][name].cpu() for name in MODALITY_ORDER},
        predictions={
            name: resize_batch_to_native(denorm_predictions[name], batch["native_targets"][name], spatial_mode=dataset.spatial_mode).cpu()
            for name in MODALITY_ORDER
        },
        split_label=split_label,
    )


def _log_scalars(writer: SummaryWriter, prefix: str, metrics: dict[str, float], step: int) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", float(value), step)


def _checkpoint_payload(
    config: dict[str, Any],
    epoch: int,
    model: UNet,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "stage": "asyn_diffusion",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": copy.deepcopy(config),
    }


def train_asyn_diffusion(config: dict[str, Any]) -> None:
    if artifacts_need_rebuild(config):
        build_and_store_artifacts(config)

    set_seed(int(config["experiment"]["seed"]))
    train_cfg = config["train"]
    data_cfg = config["data"]

    train_dataset = create_dataset(config, split="train", training=True)
    val_dataset = create_dataset(config, split="val", training=False)
    for split, dataset in (("train", train_dataset), ("val", val_dataset)):
        if len(dataset) == 0:
            raise ValueError(f"Training requires a nonempty {split} split; got 0 records.")
    for key, value in (("epochs", train_cfg["epochs"]), ("val_every", train_cfg["val_every"]),
                       ("max_train_batches", train_cfg.get("max_train_batches")),
                       ("max_val_batches", config["evaluation"].get("max_val_batches"))):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{key} must be positive, got {value}.")
    device = build_device()
    run_dirs = build_run_directories(config, stage="asyn_diffusion")
    writer = _create_summary_writer(run_dirs["tb"])
    train_loader = create_loader(
        train_dataset,
        int(train_cfg["batch_size"]),
        int(data_cfg["num_workers"]),
        True,
        bool(data_cfg["pin_memory"]),
    )
    val_loader = create_loader(
        val_dataset,
        int(train_cfg["batch_size"]),
        int(data_cfg["num_workers"]),
        False,
        bool(data_cfg["pin_memory"]),
    )

    viz_splits = [
        str(split)
        for split in config["evaluation"].get(
            "visualization_splits", [config["evaluation"].get("visualization_split", "train")]
        )
    ]
    viz_datasets = {split: create_dataset(config, split=split, visualization_subset=True, training=False) for split in viz_splits}
    viz_loaders = {
        split: create_loader(dataset, max(1, len(dataset)), 0, False, False)
        for split, dataset in viz_datasets.items()
        if len(dataset) > 0
    }

    diffusion = _build_diffusion(config, device, stats=train_dataset.stats)
    optimizer = torch.optim.AdamW(
        diffusion.model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_crps = float("inf")
    total_epochs = int(train_cfg["epochs"])
    train_pattern_specs = _train_pattern_specs(config)

    for epoch in range(1, total_epochs + 1):
        diffusion.train()
        rows: list[dict[str, float]] = []

        for step, batch in enumerate(train_loader, start=1):
            device_batch = _batch_to_device(batch, device)
            x_full = _build_joint_tensor(device_batch["inputs"])
            valid_mask = stack_valid_masks(device_batch["valid_masks"]).to(dtype=x_full.dtype)
            pattern_ids, modality_t, channel_t, loss_mask = _sample_train_async_targets(
                x_full,
                int(diffusion.timesteps),
                pattern_specs=train_pattern_specs,
            )
            masked_valid_loss = loss_mask * valid_mask

            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.loss(
                x_full,
                t=channel_t,
                model_t=modality_t,
                loss_mask=masked_valid_loss,
                head_index=_pattern_head_index_for_model(diffusion.model, pattern_ids),
            )
            loss.backward()
            grad = gradient_norm(diffusion.model.parameters())
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), float(train_cfg["grad_clip"]))
            optimizer.step()

            row = {
                "loss": float(loss.item()),
                "grad_norm": grad,
            }
            rows.append(row)

            global_step = (epoch - 1) * len(train_loader) + step
            if step % int(train_cfg["log_every"]) == 0:
                _log_scalars(writer, "train", row, global_step)

            max_batches = train_cfg.get("max_train_batches")
            if max_batches is not None and step >= int(max_batches):
                break

        train_loss = float(sum(item["loss"] for item in rows) / max(1, len(rows)))

        if epoch % int(train_cfg["val_every"]) != 0 and epoch != total_epochs:
            continue

        val_metrics = _evaluate(config, val_dataset, val_loader, diffusion, device)
        history_row = {
            "epoch": epoch,
            "timestamp": time.time(),
            "train_loss": train_loss,
            "seed_scheme": SEED_SCHEME,
            "evaluation_seed": int(config["experiment"]["seed"]),
            "spatial_mode": val_dataset.spatial_mode,
            "clip_x0": diffusion.clip_x0,
            "crps_estimator": "fair",
            "psnr_protocol": PSNR_PROTOCOL,
            "crps_samples": int(config["evaluation"].get("num_crps_samples", 5)),
            **val_metrics,
        }
        undefined = {key: "no_evaluated_samples" for key, value in val_metrics.items() if not math.isfinite(float(value))}
        if undefined and val_metrics["evaluated_sample_count"]:
            raise ValueError(f"AsynDiff validation produced unexpected nonfinite metrics: {list(undefined)}")
        for key in undefined:
            history_row[key] = None
        history_row["undefined_metrics"] = undefined
        append_history(run_dirs["history"], history_row)
        render_training_curves(run_dirs["history"], run_dirs["media"] / "training_curves.png")

        for split, loader in viz_loaders.items():
            _render_epoch_media(config, viz_datasets[split], loader, diffusion, device, epoch, run_dirs["media"], split)

        _log_scalars(writer, "val", val_metrics, epoch)

        last_path = run_dirs["checkpoints"] / "asyndiff_last.pt"
        save_checkpoint(last_path, _checkpoint_payload(config, epoch, diffusion.model, optimizer))
        selection_crps = float(val_metrics.get("roman_mean_crps", val_metrics.get("mean_crps", float("nan"))))
        # NaN fails every comparison, so an empty validation split would leave
        # asyndiff_best.pt unwritten and sampling would silently use the last epoch.
        if not math.isfinite(selection_crps):
            writer.close()
            raise ValueError(f"epoch {epoch}: no finite validation CRPS; cannot select a best checkpoint.")
        if math.isfinite(selection_crps) and selection_crps < best_crps:
            best_crps = selection_crps
            save_checkpoint(
                run_dirs["checkpoints"] / "asyndiff_best.pt",
                _checkpoint_payload(config, epoch, diffusion.model, optimizer),
            )

        print(
            f"[asyndiff] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['val_loss']:.6f} "
            f"selection_crps={selection_crps:.6f}"
        )

    writer.close()


def _resolve_sample_output(checkpoint_path: Path, output_path: Path | None) -> tuple[Path, Path]:
    if output_path is None:
        stem = checkpoint_path.parent.parent / "media" / "sample_prediction"
    else:
        stem = output_path.with_suffix("") if output_path.suffix else output_path
    stem.parent.mkdir(parents=True, exist_ok=True)
    return stem.with_suffix(".pt"), stem.with_suffix(".png")


@torch.no_grad()
def sample_prediction(
    config: dict[str, Any],
    split: str,
    sample_index: int,
    known_modalities: list[str],
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
) -> str:
    if artifacts_need_rebuild(config):
        build_and_store_artifacts(config)

    known_modalities = _validate_known_modalities(known_modalities)
    masked_modalities = _masked_from_known(known_modalities)

    checkpoint = checkpoint_path or _default_checkpoint(config)
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
    device = build_device()
    dataset = create_dataset(config, split=split, training=False)
    diffusion = _build_diffusion(config, device, stats=dataset.stats)
    _load_model_state(diffusion.model, payload["model"])
    diffusion.eval()
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"Sample index {sample_index} is out of range for split `{split}` with {len(dataset)} samples."
        )

    sample = dataset[sample_index]
    batch = {
        "record_id": [sample["record_id"]],
        "sample_id": [sample["sample_id"]],
        "label": torch.tensor([sample["label"]], dtype=torch.long),
        "inputs": {name: sample["inputs"][name].unsqueeze(0) for name in MODALITY_ORDER},
        "valid_masks": {name: sample["valid_masks"][name].unsqueeze(0) for name in MODALITY_ORDER},
        "native_targets": {name: sample["native_targets"][name].unsqueeze(0) for name in MODALITY_ORDER},
        "native_sizes": sample["native_sizes"],
    }

    device_batch = _batch_to_device(batch, device)
    x_full = _build_joint_tensor(device_batch["inputs"])
    known_mask = _known_mask_from_modalities(known_modalities, x_full)
    pred_full = _sample_with_mask(
        config,
        diffusion,
        x_full,
        known_mask,
        known_modalities,
        masked_modalities,
        device,
        sample_seeds=[sample_seed(config["experiment"]["seed"], rid, 0) for rid in batch["record_id"]],
        record_ids=batch["record_id"],
    )
    require_finite_batch(pred_full, batch["record_id"], "AsynDiff prediction")

    pred_modalities = denormalize_modalities(_split_joint_tensor(pred_full), dataset.stats)
    for name, tensor in pred_modalities.items():
        require_finite_batch(tensor, batch["record_id"], f"AsynDiff denormalized {name}")
    display_inputs = denormalize_modalities(device_batch["inputs"], dataset.stats)
    known_set = set(known_modalities)
    for name in MODALITY_ORDER:
        if name not in known_set:
            display_inputs[name] = torch.zeros_like(display_inputs[name])

    pt_path, png_path = _resolve_sample_output(checkpoint, output_path)

    torch.save(
        {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_fallback_reason": checkpoint_fallback_reason,
            "seed_scheme": SEED_SCHEME,
            "seed": int(config["experiment"]["seed"]),
            "spatial_mode": dataset.spatial_mode,
            "clip_x0": diffusion.clip_x0,
            "record_id": batch["record_id"],
            "sample_id": batch["sample_id"],
            "known_modalities": known_modalities,
            "masked_modalities": masked_modalities,
            "predictions": {name: tensor.cpu() for name, tensor in pred_modalities.items()},
            "targets": {name: tensor.cpu() for name, tensor in batch["native_targets"].items()},
        },
        pt_path,
    )

    save_imputation_epoch_panel(
        output_path=png_path,
        known_modalities=known_modalities,
        inputs={name: display_inputs[name].cpu() for name in MODALITY_ORDER},
        targets={name: tensor.cpu() for name, tensor in batch["native_targets"].items()},
        predictions={
            name: resize_batch_to_native(pred_modalities[name], batch["native_targets"][name], spatial_mode=dataset.spatial_mode).cpu()
            for name in MODALITY_ORDER
        },
    )
    return str(pt_path)

