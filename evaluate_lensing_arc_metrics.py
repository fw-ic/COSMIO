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
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from scipy import ndimage as ndi


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "asyn-diffusion" / "src"))
from asyndiff.ops.validation import require_finite

DEFAULT_LENS_DATA_ROOT = REPO_ROOT / "data/lens"
DEFAULT_OUTPUT = REPO_ROOT / "evaluation_outputs" / "lens_arc_metrics_lsst_euclid_to_roman.json"
DEFAULT_METHOD_JSONS = {
    "asyn-diffusion": REPO_ROOT
    / "evaluation_outputs/downstream_cls_roman_imputation/lens_run000/lsst_euclid_to_roman/asyn-diffusion.json",
}
ROMAN_BANDS = ("F106", "F129", "F158")


@dataclass(frozen=True)
class ArcMaskConfig:
    smooth_sigma: float = 1.0
    min_area: int = 6
    gt_percentile: float = 90.0
    gt_min_fraction_of_peak: float = 0.03
    pred_percentile: float = 97.0
    pred_snr: float = 3.0
    pred_min_fraction_of_peak: float = 0.03
    pred_inner_radius_pixels: float = 1.5
    pred_inner_radius_fraction_of_gt: float = 0.35
    pred_component_mode: str = "union"
    center_percentile: float = 99.0

    def __post_init__(self) -> None:
        if self.pred_component_mode != "union":
            raise ValueError(
                "Arc geometry requires pred_component_mode='union': both GT and "
                "prediction must retain all components that pass their filters."
            )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, tuple):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _collapse_to_2d(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        work = array
    elif array.ndim == 3 and array.shape[0] <= 8 and array.shape[-1] > 8:
        work = np.mean(np.clip(array, 0.0, None), axis=0)
    elif array.ndim == 3 and array.shape[-1] <= 8:
        work = np.mean(np.clip(array, 0.0, None), axis=-1)
    else:
        raise ValueError(f"Expected 2D image or 3D channel image, got shape={array.shape}.")
    return np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)


def _remove_small_components(mask: np.ndarray, min_area: int, radii: np.ndarray | None = None, min_radius: float = 0.0) -> np.ndarray:
    labeled, num_components = ndi.label(mask.astype(bool))
    if num_components == 0:
        return np.zeros_like(mask, dtype=bool)

    keep = np.zeros(num_components + 1, dtype=bool)
    areas = np.bincount(labeled.ravel())
    component_scores: list[tuple[int, int]] = []
    for label_id in range(1, num_components + 1):
        area = int(areas[label_id])
        if area < int(min_area):
            continue
        if radii is not None and float(np.median(radii[labeled == label_id])) < float(min_radius):
            continue
        keep[label_id] = True
        component_scores.append((area, label_id))

    if not component_scores:
        return np.zeros_like(mask, dtype=bool)
    return keep[labeled]


def estimate_lens_center(image: np.ndarray, config: ArcMaskConfig = ArcMaskConfig()) -> tuple[float, float]:
    signal = _collapse_to_2d(image)
    smooth = ndi.gaussian_filter(signal, sigma=float(config.smooth_sigma))
    finite = smooth[np.isfinite(smooth)]
    if finite.size == 0 or float(np.max(finite)) <= 0.0:
        height, width = smooth.shape
        return (0.5 * (width - 1), 0.5 * (height - 1))

    threshold = float(np.percentile(finite, float(config.center_percentile)))
    mask = smooth >= threshold
    if not np.any(mask):
        y_peak, x_peak = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
        return (float(x_peak), float(y_peak))

    yy, xx = np.indices(smooth.shape)
    weights = np.clip(smooth - threshold, 0.0, None)
    weights = np.where(mask, weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        y_peak, x_peak = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
        return (float(x_peak), float(y_peak))
    cx = float((xx * weights).sum() / total)
    cy = float((yy * weights).sum() / total)
    return (cx, cy)


def _source_arc_threshold(signal: np.ndarray, config: ArcMaskConfig) -> float:
    positive = signal[signal > 0.0]
    if positive.size == 0:
        return float("inf")
    percentile_threshold = float(np.percentile(positive, float(config.gt_percentile)))
    peak_threshold = float(config.gt_min_fraction_of_peak) * float(np.max(positive))
    return max(percentile_threshold, peak_threshold)


def extract_source_arc_mask(image: np.ndarray, config: ArcMaskConfig = ArcMaskConfig()) -> np.ndarray:
    signal = ndi.gaussian_filter(_collapse_to_2d(image), sigma=float(config.smooth_sigma))
    threshold = _source_arc_threshold(signal, config)
    mask = signal >= threshold
    return _remove_small_components(mask, min_area=int(config.min_area))


def _radial_profile_subtract(signal: np.ndarray, lens_center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    cx, cy = lens_center
    yy, xx = np.indices(signal.shape)
    radii = np.hypot(xx - cx, yy - cy)
    bins = np.floor(radii).astype(np.int64)
    max_bin = int(bins.max())
    bin_centers: list[float] = []
    medians: list[float] = []
    for bin_id in range(max_bin + 1):
        values = signal[bins == bin_id]
        if values.size == 0:
            continue
        bin_centers.append(float(bin_id))
        medians.append(float(np.median(values)))
    if not bin_centers:
        return signal.copy(), radii
    profile = np.interp(radii.ravel(), np.asarray(bin_centers), np.asarray(medians)).reshape(signal.shape)
    return signal - profile, radii


def extract_arc_mask(
    image: np.ndarray,
    lens_center: tuple[float, float],
    config: ArcMaskConfig = ArcMaskConfig(),
    gt_arc_radius_pixels: float | None = None,
) -> np.ndarray:
    signal = ndi.gaussian_filter(_collapse_to_2d(image), sigma=float(config.smooth_sigma))
    residual, radii = _radial_profile_subtract(signal, lens_center)
    inner_radius = float(config.pred_inner_radius_pixels)
    if gt_arc_radius_pixels is not None and math.isfinite(float(gt_arc_radius_pixels)):
        inner_radius = max(inner_radius, float(config.pred_inner_radius_fraction_of_gt) * float(gt_arc_radius_pixels))

    valid_region = radii >= inner_radius
    values = residual[valid_region]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(signal, dtype=bool)

    median = float(np.median(values))
    mad_sigma = 1.4826 * float(np.median(np.abs(values - median)))
    percentile_threshold = float(np.percentile(values, float(config.pred_percentile)))
    peak_threshold = float(config.pred_min_fraction_of_peak) * max(float(np.max(values)), 0.0)
    threshold = max(median + float(config.pred_snr) * mad_sigma, percentile_threshold, peak_threshold)
    mask = valid_region & (residual >= threshold)
    filtered = _remove_small_components(mask, min_area=int(config.min_area), radii=radii, min_radius=inner_radius)

    # Compare all retained arc components on both sides. Selecting only the
    # largest prediction while GT retains its union changes the comparison scope.
    return filtered


def _circular_angular_span(theta: np.ndarray) -> float:
    if theta.size == 0:
        return float("nan")
    if theta.size == 1:
        return 0.0
    wrapped = np.sort(np.mod(theta, 2.0 * np.pi))
    gaps = np.diff(wrapped)
    wrap_gap = wrapped[0] + 2.0 * np.pi - wrapped[-1]
    max_gap = float(max(np.max(gaps), wrap_gap))
    return float(2.0 * np.pi - max_gap)


def _arc_summary(
    mask: np.ndarray,
    lens_center: tuple[float, float],
    pixel_scale: float | None,
    eps: float,
) -> dict[str, float | int | bool]:
    mask = np.asarray(mask, dtype=bool)
    num_pixels = int(mask.sum())
    valid = num_pixels > 0
    if not valid:
        return {
            "arc_radius": float("nan"),
            "angular_extent": float("nan"),
            "arc_length": float("nan"),
            "arc_width": float("nan"),
            "lw_ratio": float("nan"),
            "valid_arc": False,
            "num_arc_pixels": 0,
        }

    cx, cy = lens_center
    yy, xx = np.nonzero(mask)
    radii_pixels = np.hypot(xx.astype(np.float64) - cx, yy.astype(np.float64) - cy)
    scale = float(pixel_scale) if pixel_scale is not None else 1.0
    radii = radii_pixels * scale
    radius = float(np.median(radii))
    theta = np.arctan2(yy.astype(np.float64) - cy, xx.astype(np.float64) - cx)
    angular_extent = _circular_angular_span(theta)
    arc_length = float(radius * angular_extent)
    pixel_area = scale * scale
    arc_width = float((num_pixels * pixel_area) / (arc_length + eps))
    lw_ratio = float(arc_length / (arc_width + eps))
    return {
        "arc_radius": radius,
        "angular_extent": angular_extent,
        "arc_length": arc_length,
        "arc_width": arc_width,
        "lw_ratio": lw_ratio,
        "valid_arc": True,
        "num_arc_pixels": num_pixels,
    }


def compute_lensing_arc_metrics(
    pred_image: np.ndarray,
    gt_image: np.ndarray,
    lens_center: tuple[float, float] | None = None,
    gt_arc_mask: np.ndarray | None = None,
    pred_arc_mask: np.ndarray | None = None,
    pixel_scale: float | None = None,
    mode: str = "separate_masks",
    eps: float = 1.0e-8,
    config: ArcMaskConfig = ArcMaskConfig(),
) -> dict[str, Any]:
    if mode != "separate_masks":
        raise ValueError(f"Unsupported arc metric mode: {mode!r}; use separate_masks.")
    require_finite(pred_image, "arc prediction")
    require_finite(gt_image, "arc ground truth")
    if lens_center is None:
        lens_center = estimate_lens_center(gt_image, config)
    require_finite(lens_center, "arc lens center")
    if pixel_scale is not None:
        require_finite(pixel_scale, "arc pixel scale")
    if gt_arc_mask is None:
        gt_arc_mask = extract_source_arc_mask(gt_image, config)

    gt_radius_pixels = None
    if np.any(gt_arc_mask):
        cx, cy = lens_center
        yy, xx = np.nonzero(gt_arc_mask)
        gt_radius_pixels = float(np.median(np.hypot(xx.astype(np.float64) - cx, yy.astype(np.float64) - cy)))

    if pred_arc_mask is None:
        pred_arc_mask = extract_arc_mask(pred_image, lens_center, config, gt_arc_radius_pixels=gt_radius_pixels)

    gt_summary = _arc_summary(gt_arc_mask, lens_center, pixel_scale, eps)
    pred_summary = _arc_summary(pred_arc_mask, lens_center, pixel_scale, eps)

    out: dict[str, Any] = {
        "lens_center_x": float(lens_center[0]),
        "lens_center_y": float(lens_center[1]),
        "valid_gt_arc": bool(gt_summary["valid_arc"]),
        "valid_pred_arc": bool(pred_summary["valid_arc"]),
        "num_gt_arc_pixels": int(gt_summary["num_arc_pixels"]),
        "num_pred_arc_pixels": int(pred_summary["num_arc_pixels"]),
    }
    reasons = []
    if not out["valid_gt_arc"]:
        reasons.append("no_gt_arc")
    if not out["valid_pred_arc"]:
        reasons.append("no_pred_arc")
    out["undefined_reason"] = "+".join(reasons) if reasons else None
    for key in ("angular_extent",):
        gt_value = _as_float(gt_summary[key])
        pred_value = _as_float(pred_summary[key])
        out[f"{key}_gt"] = gt_value
        out[f"{key}_pred"] = pred_value
        out[f"{key}_abs_error"] = abs(pred_value - gt_value) if math.isfinite(pred_value) and math.isfinite(gt_value) else float("nan")
    return out


def _load_normalization_stats(lens_data_root: Path) -> dict[str, Any]:
    path = lens_data_root / "normalization_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing normalization metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata["modalities"]["roman"]


def _denormalize_roman(prediction_chw: np.ndarray, roman_stats: dict[str, Any]) -> np.ndarray:
    require_finite(prediction_chw, "arc normalized prediction")
    tensor = np.asarray(prediction_chw, dtype=np.float64)
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW prediction tensor, got shape={tensor.shape}.")
    output = np.empty_like(tensor, dtype=np.float64)
    for channel_index, band in enumerate(ROMAN_BANDS):
        params = roman_stats[band]
        values = np.clip(tensor[channel_index], -1.0, 1.0)
        center = float(params["center"])
        sigma = max(float(params["sigma"]), 1.0e-6)
        a_neg = max(float(params["a_neg"]), 1.0e-6)
        a_pos = max(float(params["a_pos"]), 1.0e-6)
        positive = values >= 0.0
        output[channel_index] = np.where(
            positive,
            center + sigma * np.sinh(values * a_pos),
            center - sigma * np.sinh((-values) * a_neg),
        )
    return output


def _prediction_tensor_from_payload(path: Path, target_modality: str) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    value = payload.get("pred_normalized")
    if isinstance(value, dict):
        value = value[target_modality]
    if value is None:
        raise KeyError(f"{path} is missing pred_normalized.")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_prediction_path(example: dict[str, Any], target_modality: str, prediction_field: str) -> Path:
    if prediction_field in {"auto", "sample"}:
        paths = example.get("pred_sample_normalized_paths")
        if isinstance(paths, dict) and target_modality in paths:
            return Path(paths[target_modality])
    if prediction_field in {"auto", "point"} and example.get("pred_normalized_path"):
        return Path(example["pred_normalized_path"])
    raise KeyError(
        f"Could not resolve prediction path for sample_index={example.get('sample_index')} "
        f"with prediction_field={prediction_field!r}."
    )



def _require_within(candidate: Path, root: Path, source: str) -> None:
    """Reject a manifest-supplied path that escapes the declared data root."""
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except OSError as error:  # pragma: no cover - filesystem dependent
        raise ValueError(f"Could not resolve {candidate} from {source!r}.") from error
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"Record {source!r} resolves to {resolved}, outside the data root {root_resolved}."
        )


class LensZarrCache:
    def __init__(self, lens_data_root: Path, target_modality: str, gt_arc_group: str, deflector_group: str) -> None:
        self.lens_data_root = lens_data_root
        self.target_modality = target_modality
        self.gt_arc_group = gt_arc_group
        self.deflector_group = deflector_group
        self._groups: dict[Path, zarr.hierarchy.Group] = {}
        self._pixel_scale: float | None = None

    def _group(self, shard_path: Path) -> zarr.hierarchy.Group:
        group = self._groups.get(shard_path)
        if group is None:
            group = zarr.open_group(shard_path, mode="r")
            self._groups[shard_path] = group
        return group

    def _parse_record_id(self, record_id: str) -> tuple[Path, int]:
        shard_rel, local_index_text = str(record_id).rsplit(":", 1)
        shard_path = self.lens_data_root / shard_rel
        # record_id comes verbatim out of an evaluation JSON. `/` replaces rather
        # than appends on an absolute right-hand side, and "../" walks upward, so
        # the containment this expression looks like it enforces has to be checked.
        _require_within(shard_path, self.lens_data_root, record_id)
        return shard_path, int(local_index_text)

    def pixel_scale(self, record_id: str) -> float | None:
        if self._pixel_scale is not None:
            return self._pixel_scale
        shard_path, _ = self._parse_record_id(record_id)
        arr = self._group(shard_path)[self.gt_arc_group][self.target_modality]
        band_config = arr.attrs.get("band_config")
        if band_config:
            configs = json.loads(band_config)
            scales = [float(item["pixel_scale"]) for item in configs if "pixel_scale" in item]
            if scales:
                self._pixel_scale = float(np.mean(scales))
                return self._pixel_scale
        return None

    def source_and_deflector(self, record_id: str) -> tuple[np.ndarray, np.ndarray]:
        shard_path, local_index = self._parse_record_id(record_id)
        group = self._group(shard_path)
        source = np.asarray(group[self.gt_arc_group][self.target_modality][local_index])
        deflector = np.asarray(group[self.deflector_group][self.target_modality][local_index])
        return source, deflector


def _method_examples(eval_json: Path, method_name: str) -> list[dict[str, Any]]:
    with eval_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    methods = payload.get("methods", {})
    if method_name in methods:
        return list(methods[method_name].get("examples", []))
    # Falling back to the sole method when the name does not match would attribute
    # its numbers to whatever the caller mistyped.
    raise KeyError(
        f"Could not find method {method_name!r} in {eval_json}. Available: {sorted(methods)}."
    )


def _aggregate_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([_as_float(row.get(key)) for row in rows], dtype=np.float64)
    valid = np.isfinite(values)
    for row, value in zip(rows, values):
        if not np.isfinite(value) and not row.get("undefined_reason"):
            raise ValueError(f"Unexpected nonfinite {key} for record_id={row.get('record_id', 'unknown')}")
    finite = values[valid]
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "finite_count": 0}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "finite_count": int(finite.size),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "angular_extent_abs_error",
        "angular_extent_gt",
        "angular_extent_pred",
    ]
    aggregate = {key: _aggregate_metric(rows, key) for key in keys}
    total = max(1, len(rows))
    aggregate["undefined_reason_counts"] = dict(Counter(row["undefined_reason"] for row in rows if row.get("undefined_reason")))
    aggregate["valid_pair_count"] = sum(bool(row.get("valid_gt_arc")) and bool(row.get("valid_pred_arc")) for row in rows)
    aggregate["undefined_pair_count"] = len(rows) - aggregate["valid_pair_count"]
    aggregate["count"] = len(rows)
    aggregate["valid_gt_arc_rate"] = float(sum(bool(row.get("valid_gt_arc")) for row in rows) / total)
    aggregate["valid_pred_arc_rate"] = float(sum(bool(row.get("valid_pred_arc")) for row in rows) / total)
    aggregate["valid_pair_rate"] = float(
        sum(bool(row.get("valid_gt_arc")) and bool(row.get("valid_pred_arc")) for row in rows) / total
    )
    return aggregate


def _parse_method_jsons(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_METHOD_JSONS)
    outputs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--eval-json must use METHOD=PATH, got {value!r}.")
        method, path = value.split("=", 1)
        outputs[method.strip()] = Path(path).expanduser().resolve()
    return outputs


def _write_summary_tsv(output_path: Path, results: dict[str, Any]) -> None:
    summary_path = output_path.with_suffix(".tsv")
    columns = [
        "method",
        "count",
        "valid_pred_arc_rate",
        "valid_pair_count",
        "undefined_pair_count",
        "valid_pair_rate",
        "angular_extent_abs_error_median",
    ]
    lines = ["\t".join(columns)]
    for method_name, method_payload in results["methods"].items():
        agg = method_payload["aggregate"]
        row = {
            "method": method_name,
            "count": agg["count"],
            "valid_pred_arc_rate": agg["valid_pred_arc_rate"],
            "valid_pair_count": agg["valid_pair_count"],
            "undefined_pair_count": agg["undefined_pair_count"],
            "valid_pair_rate": agg["valid_pair_rate"],
            "angular_extent_abs_error_median": _finite_or_none(agg["angular_extent_abs_error"]["median"]),
        }
        lines.append("\t".join(("null" if row[column] is None else str(row[column])) for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_method(
    method_name: str,
    eval_json: Path,
    lens_cache: LensZarrCache,
    roman_stats: dict[str, Any],
    args: argparse.Namespace,
    config: ArcMaskConfig,
) -> dict[str, Any]:
    examples = _method_examples(eval_json, method_name)
    if args.limit is not None:
        examples = examples[: int(args.limit)]
    if not examples:
        raise ValueError(f"method={method_name}: no examples to evaluate.")
    rows: list[dict[str, Any]] = []
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[float, float], float | None, float | None]] = {}

    for offset, example in enumerate(examples, start=1):
        record_id = str(example["record_id"])
        sample_index = int(example["sample_index"])
        sample_id = str(example["sample_id"])
        pred_path = _resolve_prediction_path(example, args.target_modality, args.prediction_field)
        pred_norm = _prediction_tensor_from_payload(pred_path, args.target_modality)
        context = f"method={method_name} record_id={record_id}"
        require_finite(pred_norm, context + " normalized prediction")
        pred_flux = _denormalize_roman(pred_norm, roman_stats)
        require_finite(pred_flux, context + " native prediction")

        cached = gt_cache.get(record_id)
        if cached is None:
            gt_source, deflector = lens_cache.source_and_deflector(record_id)
            require_finite(gt_source, context + " GT source")
            require_finite(deflector, context + " deflector")
            lens_center = estimate_lens_center(deflector, config)
            gt_mask = extract_source_arc_mask(gt_source, config)
            pixel_scale = lens_cache.pixel_scale(record_id)
            gt_radius_pixels = None
            if np.any(gt_mask):
                cx, cy = lens_center
                yy, xx = np.nonzero(gt_mask)
                gt_radius_pixels = float(np.median(np.hypot(xx.astype(np.float64) - cx, yy.astype(np.float64) - cy)))
            cached = (gt_source, gt_mask, lens_center, pixel_scale, gt_radius_pixels)
            gt_cache[record_id] = cached
        gt_source, gt_mask, lens_center, pixel_scale, gt_radius_pixels = cached

        pred_mask = extract_arc_mask(pred_flux, lens_center, config, gt_arc_radius_pixels=gt_radius_pixels)
        metrics = compute_lensing_arc_metrics(
            pred_flux,
            gt_source,
            lens_center=lens_center,
            gt_arc_mask=gt_mask,
            pred_arc_mask=pred_mask,
            pixel_scale=pixel_scale,
            mode=args.mode,
            eps=float(args.eps),
            config=config,
        )
        metrics.update(
            {
                "sample_index": sample_index,
                "sample_id": sample_id,
                "record_id": record_id,
                "prediction_path": str(pred_path),
            }
        )
        rows.append(metrics)

        if args.progress_every > 0 and (offset % int(args.progress_every) == 0 or offset == len(examples)):
            print(f"[{method_name}] {offset}/{len(examples)}")

    return {
        "method": method_name,
        "eval_json": str(eval_json),
        "prediction_field": args.prediction_field,
        "examples": [] if args.no_examples else rows,
        "aggregate": _aggregate_rows(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate strong-lensing arc geometry metrics from saved imputation predictions."
    )
    parser.add_argument(
        "--eval-json",
        action="append",
        help="Evaluation JSON in METHOD=PATH form. Defaults to the released AsynDiff lens-run000 result.",
    )
    parser.add_argument("--lens-data-root", default=str(DEFAULT_LENS_DATA_ROOT))
    parser.add_argument("--target-modality", default="roman")
    parser.add_argument("--gt-arc-group", default="source_lensed_clean_psf")
    parser.add_argument("--deflector-group", default="deflector_clean_psf")
    parser.add_argument("--prediction-field", choices=("auto", "sample", "point"), default="sample")
    parser.add_argument("--mode", choices=("separate_masks",), default="separate_masks")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--min-area", type=int, default=6)
    parser.add_argument("--gt-percentile", type=float, default=90.0)
    parser.add_argument("--pred-percentile", type=float, default=97.0)
    parser.add_argument("--pred-snr", type=float, default=3.0)
    parser.add_argument(
        "--pred-component-mode", choices=("union",), default="union",
        help="Retain all qualifying arc components in both GT and prediction (union only).",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--no-examples", action="store_true", help="Write aggregate metrics only.")
    args = parser.parse_args()

    lens_data_root = Path(args.lens_data_root).expanduser().resolve()
    method_jsons = _parse_method_jsons(args.eval_json)
    config = ArcMaskConfig(
        min_area=int(args.min_area),
        gt_percentile=float(args.gt_percentile),
        pred_percentile=float(args.pred_percentile),
        pred_snr=float(args.pred_snr),
        pred_component_mode=str(args.pred_component_mode),
    )
    lens_cache = LensZarrCache(
        lens_data_root=lens_data_root,
        target_modality=str(args.target_modality),
        gt_arc_group=str(args.gt_arc_group),
        deflector_group=str(args.deflector_group),
    )
    roman_stats = _load_normalization_stats(lens_data_root)

    results: dict[str, Any] = {
        "metric": "lensing_arc_geometry",
        "target_modality": str(args.target_modality),
        "mode": str(args.mode),
        "prediction_field": str(args.prediction_field),
        "lens_data_root": str(lens_data_root),
        "gt_arc_group": str(args.gt_arc_group),
        "deflector_group": str(args.deflector_group),
        # angular_extent_* are radians; centres and counts are pixels. The
        # arcsec-valued length metrics are computed but not exported.
        "units": "radians_and_pixels",
        "config": {
            "min_area": config.min_area,
            "gt_percentile": config.gt_percentile,
            "pred_percentile": config.pred_percentile,
            "pred_snr": config.pred_snr,
            "pred_component_mode": config.pred_component_mode,
        },
        "methods": {},
    }

    for method_name, eval_json in method_jsons.items():
        print(f"[{method_name}] reading {eval_json}")
        results["methods"][method_name] = evaluate_method(
            method_name=method_name,
            eval_json=eval_json,
            lens_cache=lens_cache,
            roman_stats=roman_stats,
            args=args,
            config=config,
        )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_finite_or_none(results), indent=2, allow_nan=False), encoding="utf-8")
    _write_summary_tsv(output_path, results)

    print()
    print("method\tvalid_pred_arc_rate\tvalid_pair_count\tcount\tvalid_pair_rate\tangular_extent_abs_error_median")
    for method_name, method_payload in results["methods"].items():
        agg = method_payload["aggregate"]
        print(
            f"{method_name}\t"
            f"{agg['valid_pred_arc_rate']:.4f}\t"
            f"{agg['valid_pair_count']}\t"
            f"{agg['count']}\t"
            f"{agg['valid_pair_rate']:.4f}\t"
            f"{agg['angular_extent_abs_error']['median']:.6f}"
        )
    print()
    print(output_path)
    print(output_path.with_suffix(".tsv"))


if __name__ == "__main__":
    main()
