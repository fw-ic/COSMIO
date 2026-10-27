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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from vit_lens_classification.manifest import load_json
from vit_lens_classification.normalization import denormalize_roman_tensor, load_normalization_metadata

RGB_STRETCH = 0.5
RGB_Q = 8.0
UINT8_MAX = 255.0
DISPLAY_INTERPOLATION = "bilinear"


def _resolve_method(results: dict[str, Any], method_name: str | None) -> str:
    methods = results.get("methods", {})
    if method_name is not None:
        if method_name not in methods:
            raise KeyError(f"Method `{method_name}` not present in {sorted(methods)}")
        return method_name
    if len(methods) != 1:
        raise ValueError(f"Expected exactly one method in results JSON, found {sorted(methods)}. Pass --method.")
    return next(iter(methods))


def _record_to_shard_path(data_root: Path, record_id: str) -> tuple[Path, int]:
    shard_relpath, local_index = str(record_id).rsplit(":", 1)
    shard_path = data_root / shard_relpath
    # See _require_within in evaluate_lensing_arc_metrics.py: record ids are
    # manifest data, and `/` does not confine an absolute or traversing suffix.
    resolved, root_resolved = shard_path.resolve(), Path(data_root).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"Record {record_id!r} resolves to {resolved}, outside the data root {root_resolved}."
        )
    return shard_path, int(local_index)


def _to_chw(array: np.ndarray) -> torch.Tensor:
    data = np.asarray(array, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    return torch.from_numpy(np.moveaxis(data, -1, 0)).float()


def _lupton_rgb(image_r: np.ndarray, image_g: np.ndarray, image_b: np.ndarray, stretch: float = RGB_STRETCH, q: float = RGB_Q) -> np.ndarray:
    image_r = np.asarray(image_r, dtype=np.float32)
    image_g = np.asarray(image_g, dtype=np.float32)
    image_b = np.asarray(image_b, dtype=np.float32)
    intensity = (image_r + image_g + image_b) / 3.0

    epsilon = 1.0 / 2**23
    q = 0.1 if abs(q) < epsilon else min(float(q), 1.0e10)
    slope = 0.1 * UINT8_MAX / np.arcsinh(0.1 * q)
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
        scale = np.where(peak > UINT8_MAX, UINT8_MAX / peak, 1.0)
    channels = [np.clip(channel * scale, 0.0, UINT8_MAX) / UINT8_MAX for channel in channels]
    return np.stack(channels, axis=-1).astype(np.float32)


def _roman_notebook_display(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy().astype(np.float32)
    return _lupton_rgb(array[2], array[1], array[0], stretch=RGB_STRETCH)


def _positive_rgb_display(channels: np.ndarray, high_percentile: float = 99.5) -> np.ndarray:
    positive = np.clip(channels.astype(np.float32), 0.0, None)
    values = positive[np.isfinite(positive) & (positive > 0.0)]
    if values.size == 0:
        return np.zeros_like(positive, dtype=np.float32)
    scale = float(np.percentile(values, high_percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(values))
    scale = max(scale, 1.0e-6)
    return np.clip(positive / scale, 0.0, 1.0).astype(np.float32)


def _roman_normalized_display(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy().astype(np.float32)
    rgb = np.stack([array[2], array[1], array[0]], axis=-1)
    return _positive_rgb_display(rgb)


def _load_target_normalized(data_root: Path, record_id: str, target_group: str, target_modality: str) -> torch.Tensor:
    shard_path, local_index = _record_to_shard_path(data_root, record_id)
    group = zarr.open_group(str(shard_path), mode="r")[target_group]
    return _to_chw(group[target_modality][local_index])


def _comparison_rows(
    results_json: Path,
    data_root: Path,
    method_name: str | None,
    target_modality: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    results = load_json(results_json)
    resolved_method = _resolve_method(results, method_name)
    method_results = results["methods"][resolved_method]
    examples = method_results.get("examples", [])
    if limit is not None:
        examples = examples[: int(limit)]
    rows: list[dict[str, Any]] = []
    for example in examples:
        pred_paths = example.get("pred_sample_normalized_paths", {})
        pred_path = pred_paths.get(target_modality)
        if pred_path is None:
            raise KeyError(
                f"Missing pred_sample_normalized_paths[{target_modality!r}] for record_id={example.get('record_id')}"
            )
        rows.append(
            {
                "record_id": str(example["record_id"]),
                "sample_id": str(example["sample_id"]),
                "prediction_path": str(pred_path),
                "target_group": str(example.get("target_group", "images")),
            }
        )
    return rows


def build_visualizations(
    results_json: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    *,
    method_name: str | None = None,
    target_modality: str = "roman",
    limit: int | None = 8,
    display_space: str = "notebook",
) -> Path:
    results_path = Path(results_json)
    lens_root = Path(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = None
    if str(display_space) == "notebook":
        metadata = load_normalization_metadata(lens_root / "normalization_metadata.json")
    rows = _comparison_rows(results_path, lens_root, method_name, target_modality, limit)
    if not rows:
        raise ValueError(f"No examples found in {results_path}")

    montage_fig, montage_axes = plt.subplots(len(rows), 2, figsize=(6.4, max(2.6, len(rows) * 2.5)))
    if len(rows) == 1:
        montage_axes = np.expand_dims(montage_axes, axis=0)

    for row_index, row in enumerate(rows):
        pred_payload = torch.load(row["prediction_path"], map_location="cpu", weights_only=True)
        pred_normalized = pred_payload["pred_normalized"]
        if isinstance(pred_normalized, dict):
            pred_normalized = pred_normalized[target_modality]
        if not isinstance(pred_normalized, torch.Tensor):
            raise TypeError(f"Saved prediction payload is missing a tensor: {row['prediction_path']}")

        target_normalized = _load_target_normalized(
            data_root=lens_root,
            record_id=row["record_id"],
            target_group=str(row["target_group"]),
            target_modality=target_modality,
        )
        if str(display_space) == "normalized":
            target_panel = _roman_normalized_display(target_normalized)
            pred_panel = _roman_normalized_display(pred_normalized.float())
            panel_label = "normalized linear Roman display"
            montage_suffix = "_normalized"
        else:
            assert metadata is not None
            target_flux = denormalize_roman_tensor(target_normalized, metadata)
            pred_flux = denormalize_roman_tensor(pred_normalized.float(), metadata)
            target_panel = _roman_notebook_display(target_flux)
            pred_panel = _roman_notebook_display(pred_flux)
            panel_label = "notebook-style Roman display"
            montage_suffix = ""

        sample_path = output_path / f"{row['sample_id']}.png"
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
        for axis, title, panel in zip(
            axes,
            ("target", "imputed sample"),
            (target_panel, pred_panel),
        ):
            axis.imshow(np.clip(panel, 0.0, 1.0), origin="lower", interpolation=DISPLAY_INTERPOLATION)
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(f"{row['sample_id']} | {panel_label}", fontsize=10)
        plt.tight_layout()
        fig.savefig(sample_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        for col_index, (title, panel) in enumerate((("target", target_panel), ("imputed sample", pred_panel))):
            axis = montage_axes[row_index, col_index]
            axis.imshow(np.clip(panel, 0.0, 1.0), origin="lower", interpolation=DISPLAY_INTERPOLATION)
            if row_index == 0:
                axis.set_title(title, fontsize=9)
            if col_index == 0:
                axis.text(
                    0.03,
                    0.97,
                    str(row["sample_id"]),
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 2},
                )
            axis.set_xticks([])
            axis.set_yticks([])

    prefix = results_path.stem.replace("_imputation", "")
    montage_path = output_path / f"{prefix}_first{len(rows)}_target_vs_imputed_sample{montage_suffix}.png"
    montage_fig.suptitle(
        f"{prefix} | {'normalized linear' if str(display_space) == 'normalized' else 'notebook-style'} Roman target vs imputed sample",
        fontsize=11,
        y=0.995,
    )
    plt.tight_layout()
    montage_fig.subplots_adjust(top=0.94, wspace=0.03, hspace=0.06)
    montage_fig.savefig(montage_path, dpi=180, bbox_inches="tight")
    plt.close(montage_fig)
    return montage_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render saved imputed Roman samples using notebook-style denormalized visualization.")
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--data-root", required=True, help="Normalized lens root containing the real Roman targets.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--target-modality", default="roman")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--display-space", choices=("notebook", "normalized"), default="notebook")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    montage_path = build_visualizations(
        results_json=args.results_json,
        data_root=args.data_root,
        output_dir=args.output_dir,
        method_name=args.method,
        target_modality=args.target_modality,
        limit=int(args.limit),
        display_space=str(args.display_space),
    )
    print(montage_path)


if __name__ == "__main__":
    main()

