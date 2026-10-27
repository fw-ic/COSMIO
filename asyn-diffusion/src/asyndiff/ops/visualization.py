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

import matplotlib.pyplot as plt
import numpy as np
import torch

from asyndiff.constants import MODALITY_ORDER

_RGB_STRETCH = 0.5
_RGB_Q = 8.0
_UINT8_MAX = 255.0


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


def _lupton_rgb(image_r: np.ndarray, image_g: np.ndarray, image_b: np.ndarray, stretch: float = _RGB_STRETCH, q: float = _RGB_Q) -> np.ndarray:
    image_r = np.asarray(image_r, dtype=np.float32)
    image_g = np.asarray(image_g, dtype=np.float32)
    image_b = np.asarray(image_b, dtype=np.float32)
    intensity = (image_r + image_g + image_b) / 3.0

    epsilon = 1.0 / 2**23
    q = 0.1 if abs(q) < epsilon else min(float(q), 1.0e10)
    slope = 0.1 * _UINT8_MAX / np.arcsinh(0.1 * q)
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
        scale = np.where(peak > _UINT8_MAX, _UINT8_MAX / peak, 1.0)
    channels = [np.clip(channel * scale, 0.0, _UINT8_MAX) / _UINT8_MAX for channel in channels]
    return np.stack(channels, axis=-1).astype(np.float32)


def modality_display(name: str, tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy().astype(np.float32)
    if name == "euclid":
        return array[0]
    if name in {"lsst", "roman"}:
        # Notebook-equivalent RGB mapping: rgb_image_from_image_list([R, G, B], stretch=0.5)
        return _lupton_rgb(array[2], array[1], array[0], stretch=_RGB_STRETCH)
    raise KeyError(name)


def error_display(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return torch.abs(pred - target).mean(dim=0).detach().cpu().numpy().astype(np.float32)


def append_history(history_path: Path, row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def load_history(history_path: Path) -> list[dict[str, Any]]:
    if not history_path.exists():
        return []
    rows = []
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_training_curves(history_path: Path, output_path: Path) -> None:
    rows = load_history(history_path)
    if not rows:
        return

    epochs = [row["epoch"] for row in rows]
    train_loss = [row.get("train_loss", np.nan) for row in rows]
    val_loss = [row.get("val_loss", np.nan) for row in rows]
    mean_crps = [row.get("roman_mean_crps", row.get("mean_crps", np.nan)) for row in rows]
    mean_psnr = [row.get("roman_mean_psnr", row.get("mean_psnr", np.nan)) for row in rows]
    mean_mse = [row.get("roman_mean_mse", row.get("mean_mse", np.nan)) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes.ravel()
    ax[0].plot(epochs, train_loss, label="train")
    ax[0].plot(epochs, val_loss, label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(epochs, mean_crps)
    ax[1].set_title("Mean CRPS")
    ax[2].plot(epochs, mean_mse)
    ax[2].set_title("Mean MSE")
    ax[3].plot(epochs, mean_psnr)
    ax[3].set_title("Mean PSNR")
    for axis in ax:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_imputation_epoch_panel(
    output_path: Path,
    known_modalities: list[str],
    inputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    predictions: dict[str, torch.Tensor],
    split_label: str | None = None,
) -> None:
    batch = min(4, targets[MODALITY_ORDER[0]].shape[0])
    columns = 12
    fig, axes = plt.subplots(batch, columns, figsize=(columns * 2.0, batch * 2.1))
    if batch == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = [
        "LSST in",
        "Euclid in",
        "Roman in",
        "LSST gt",
        "Euclid gt",
        "Roman gt",
        "LSST pred",
        "Euclid pred",
        "Roman pred",
        "LSST err",
        "Euclid err",
        "Roman err",
    ]

    euclid_images: list[np.ndarray] = []
    all_error_images: list[np.ndarray] = []
    row_payloads: list[dict[str, dict[str, np.ndarray]]] = []

    for row in range(batch):
        input_panels = {name: modality_display(name, inputs[name][row]) for name in MODALITY_ORDER}
        target_panels = {name: modality_display(name, targets[name][row]) for name in MODALITY_ORDER}
        pred_panels = {name: modality_display(name, predictions[name][row]) for name in MODALITY_ORDER}
        error_panels = {
            name: np.zeros(targets[name][row].shape[-2:], dtype=np.float32)
            if name in known_modalities
            else error_display(predictions[name][row], targets[name][row])
            for name in MODALITY_ORDER
        }

        euclid_images.extend([input_panels["euclid"], target_panels["euclid"], pred_panels["euclid"]])
        all_error_images.extend([error_panels["lsst"], error_panels["euclid"], error_panels["roman"]])

        row_payloads.append(
            {
                "input": input_panels,
                "target": target_panels,
                "pred": pred_panels,
                "error": error_panels,
            }
        )

    euclid_vmin, euclid_vmax = _robust_limits(euclid_images)
    _err_vmin, err_vmax = _robust_limits(all_error_images, lo=0.0, hi=99.0)

    for row in range(batch):
        payload = row_payloads[row]
        panels = [
            payload["input"]["lsst"],
            payload["input"]["euclid"],
            payload["input"]["roman"],
            payload["target"]["lsst"],
            payload["target"]["euclid"],
            payload["target"]["roman"],
            payload["pred"]["lsst"],
            payload["pred"]["euclid"],
            payload["pred"]["roman"],
            payload["error"]["lsst"],
            payload["error"]["euclid"],
            payload["error"]["roman"],
        ]

        for col, panel in enumerate(panels):
            axis = axes[row, col]
            if panel.ndim == 2:
                if col in {1, 4, 7}:
                    axis.imshow(panel, cmap="gray", origin="lower", vmin=euclid_vmin, vmax=euclid_vmax)
                else:
                    axis.imshow(panel, cmap="magma", origin="lower", vmin=0.0, vmax=err_vmax)
            else:
                axis.imshow(np.clip(panel, 0.0, 1.0), origin="lower")
            if row == 0:
                axis.set_title(titles[col], fontsize=8)
            if col == 0 and split_label is not None:
                axis.text(
                    0.03,
                    0.97,
                    split_label,
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
                )
            axis.set_xticks([])
            axis.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.02, hspace=0.05)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

