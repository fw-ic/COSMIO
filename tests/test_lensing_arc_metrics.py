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

"""Coverage for evaluate_lensing_arc_metrics.py.

The script produces the angular extent error that docs/data_schema.md describes,
reading the `source_lensed_clean_psf` and `deflector_clean_psf` groups. No other
fixture in the repository writes those groups, so the shard reader and the arc
geometry are exercised here against synthetic lensing images built in-module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

ROOT = Path(__file__).resolve().parents[1]
# The script sits at the repository root, which pytest does not put on sys.path
# when it collects from tests/.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_lensing_arc_metrics import (  # noqa: E402
    ArcMaskConfig,
    LensZarrCache,
    extract_arc_mask,
    _circular_angular_span,
    compute_lensing_arc_metrics,
    estimate_lens_center,
    extract_source_arc_mask,
)

SIZE = 59
CENTRE = 0.5 * (SIZE - 1)


def _blob(cx: float, cy: float, sigma: float = 3.0) -> np.ndarray:
    yy, xx = np.indices((SIZE, SIZE)).astype(np.float64)
    return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))).astype(np.float32)


def _arc(radius: float, span: float = 2.0 * np.pi, width: float = 1.5) -> np.ndarray:
    """A ring of the given radius, optionally truncated to `span` radians."""
    yy, xx = np.indices((SIZE, SIZE)).astype(np.float64)
    dx, dy = xx - CENTRE, yy - CENTRE
    ring = np.exp(-((np.hypot(dx, dy) - radius) ** 2) / (2.0 * width**2))
    if span < 2.0 * np.pi:
        theta = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
        ring = np.where(theta <= span, ring, 0.0)
    return ring.astype(np.float32)


# --------------------------------------------------------------------------- #
# lens centre
# --------------------------------------------------------------------------- #

def test_lens_centre_recovers_a_centred_deflector() -> None:
    cx, cy = estimate_lens_center(_blob(CENTRE, CENTRE))
    assert cx == pytest.approx(CENTRE, abs=1.0)
    assert cy == pytest.approx(CENTRE, abs=1.0)


def test_lens_centre_recovers_an_offset_deflector() -> None:
    cx, cy = estimate_lens_center(_blob(20.0, 35.0))
    assert cx == pytest.approx(20.0, abs=1.0)
    assert cy == pytest.approx(35.0, abs=1.0)


def test_lens_centre_of_a_blank_image_is_the_geometric_centre() -> None:
    assert estimate_lens_center(np.zeros((SIZE, SIZE), dtype=np.float32)) == (CENTRE, CENTRE)


# --------------------------------------------------------------------------- #
# arc extraction
# --------------------------------------------------------------------------- #

def test_source_arc_mask_sits_at_the_ring_radius() -> None:
    mask = extract_source_arc_mask(_arc(radius=15.0))
    assert mask.any(), "a bright ring should yield a non-empty arc mask"
    yy, xx = np.nonzero(mask)
    radii = np.hypot(xx - CENTRE, yy - CENTRE)
    assert float(np.median(radii)) == pytest.approx(15.0, abs=2.0)


def test_source_arc_mask_of_a_blank_image_is_empty() -> None:
    assert not extract_source_arc_mask(np.zeros((SIZE, SIZE), dtype=np.float32)).any()


# --------------------------------------------------------------------------- #
# angular span
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False), 2.0 * np.pi),
        (np.linspace(0.0, np.pi, 180, endpoint=False), np.pi),
        (np.linspace(0.0, 0.5 * np.pi, 90, endpoint=False), 0.5 * np.pi),
    ],
)
def test_angular_span_of_a_uniform_arc(theta: np.ndarray, expected: float) -> None:
    assert _circular_angular_span(theta) == pytest.approx(expected, abs=0.1)


def test_angular_span_edge_cases() -> None:
    assert _circular_angular_span(np.array([1.0])) == 0.0
    assert np.isnan(_circular_angular_span(np.array([])))


def test_angular_span_is_wrap_aware() -> None:
    """Angles either side of 0 form one arc, not one spanning almost the circle."""
    theta = np.concatenate([np.linspace(0.0, 0.2, 20), np.linspace(2.0 * np.pi - 0.2, 2.0 * np.pi, 20)])
    assert _circular_angular_span(theta) == pytest.approx(0.4, abs=0.1)


# --------------------------------------------------------------------------- #
# the metric itself
# --------------------------------------------------------------------------- #

def test_equal_explicit_masks_have_zero_geometry_error() -> None:
    image = _arc(radius=15.0)
    mask = extract_source_arc_mask(image)
    out = compute_lensing_arc_metrics(image, image, gt_arc_mask=mask, pred_arc_mask=mask)
    assert out["valid_gt_arc"] and out["valid_pred_arc"]
    assert out["angular_extent_abs_error"] == 0.0
    assert out["num_gt_arc_pixels"] == out["num_pred_arc_pixels"]


def test_a_complete_ring_spans_the_full_circle() -> None:
    out = compute_lensing_arc_metrics(_arc(15.0), _arc(15.0), mode="separate_masks")
    assert out["angular_extent_gt"] == pytest.approx(2.0 * np.pi, abs=0.3)


def test_a_truncated_arc_covers_fewer_pixels_than_a_complete_ring() -> None:
    """Pixel count, not angular extent.

    extract_source_arc_mask thresholds at a percentile of the whole frame, so as
    the arc shrinks the threshold falls and the measured angular extent does not
    decrease monotonically with the true span. The pixel count does.
    """
    full = compute_lensing_arc_metrics(_arc(15.0), _arc(15.0), mode="separate_masks")
    half = compute_lensing_arc_metrics(_arc(15.0, span=np.pi), _arc(15.0, span=np.pi), mode="separate_masks")
    assert half["num_gt_arc_pixels"] < full["num_gt_arc_pixels"]


def test_a_mismatched_prediction_scores_worse_than_a_matching_one() -> None:
    matching = compute_lensing_arc_metrics(_arc(15.0), _arc(15.0), mode="separate_masks")
    mismatched = compute_lensing_arc_metrics(_arc(15.0, span=0.5 * np.pi), _arc(15.0), mode="separate_masks")
    assert mismatched["angular_extent_abs_error"] > matching["angular_extent_abs_error"]


def test_unsupported_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported arc metric mode"):
        compute_lensing_arc_metrics(_arc(15.0), _arc(15.0), mode="nonsense")


# --------------------------------------------------------------------------- #
# the shard reader
# --------------------------------------------------------------------------- #

PIXEL_SCALES = [{"band": "F106", "pixel_scale": 0.11}, {"band": "F129", "pixel_scale": 0.11}]


# The deflector blob is deliberately off-centre. A centred blob and a centred
# ring are indistinguishable to a centroid test, so the reader could return the
# groups the wrong way round and still satisfy it.
DEFLECTOR_XY = (20.0, 35.0)


def _write_lens_shard(shard_path: Path, count: int = 2) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(shard_path), mode="w")
    source = np.stack([np.stack([_arc(15.0)] * 3, axis=-1) for _ in range(count)])
    deflector = np.stack([np.stack([_blob(*DEFLECTOR_XY)] * 3, axis=-1) for _ in range(count)])
    src_group = root.create_group("source_lensed_clean_psf")
    arr = src_group.create_dataset("roman", data=source, dtype="f4")
    arr.attrs["band_config"] = json.dumps(PIXEL_SCALES)
    root.create_group("deflector_clean_psf").create_dataset("roman", data=deflector, dtype="f4")


def test_shard_reader_returns_source_and_deflector(tmp_path: Path) -> None:
    rel = Path("run_000/shards/lens/lens_000000.zarr")
    _write_lens_shard(tmp_path / rel)
    cache = LensZarrCache(tmp_path, "roman", "source_lensed_clean_psf", "deflector_clean_psf")

    source, deflector = cache.source_and_deflector(f"{rel.as_posix()}:1")
    assert source.shape == (SIZE, SIZE, 3)
    assert deflector.shape == (SIZE, SIZE, 3)

    # The deflector is a blob at DEFLECTOR_XY; the source is a ring centred on
    # the frame. Each assertion fails if the two groups are returned swapped.
    cx, cy = estimate_lens_center(deflector)
    assert (cx, cy) == pytest.approx(DEFLECTOR_XY, abs=1.5)

    mask = extract_source_arc_mask(source)
    assert mask.any()
    yy, xx = np.nonzero(mask)
    assert float(np.median(np.hypot(xx - CENTRE, yy - CENTRE))) == pytest.approx(15.0, abs=2.0)


def test_shard_reader_reads_pixel_scale_from_band_config(tmp_path: Path) -> None:
    rel = Path("run_000/shards/lens/lens_000000.zarr")
    _write_lens_shard(tmp_path / rel)
    cache = LensZarrCache(tmp_path, "roman", "source_lensed_clean_psf", "deflector_clean_psf")
    assert cache.pixel_scale(f"{rel.as_posix()}:0") == pytest.approx(0.11)


def test_shard_reader_reports_no_pixel_scale_without_band_config(tmp_path: Path) -> None:
    rel = Path("run_000/shards/lens/lens_000000.zarr")
    shard = tmp_path / rel
    shard.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(shard), mode="w")
    blank = np.zeros((1, SIZE, SIZE, 3), dtype=np.float32)
    root.create_group("source_lensed_clean_psf").create_dataset("roman", data=blank, dtype="f4")
    root.create_group("deflector_clean_psf").create_dataset("roman", data=blank, dtype="f4")
    cache = LensZarrCache(tmp_path, "roman", "source_lensed_clean_psf", "deflector_clean_psf")
    assert cache.pixel_scale(f"{rel.as_posix()}:0") is None


def test_both_extractors_retain_two_separated_components() -> None:
    # Two unequal bright patches: a largest-only policy would discard the
    # smaller right-hand patch. The remaining extraction settings are unchanged.
    signal = np.zeros((SIZE, SIZE), dtype=np.float64)
    signal[26:33, 9:14] = 10.0
    signal[27:32, 45:49] = 10.0
    config = ArcMaskConfig(smooth_sigma=0.0)
    gt_mask = extract_source_arc_mask(signal, config)
    pred_mask = extract_arc_mask(signal, (CENTRE, CENTRE), config)
    for mask in (gt_mask, pred_mask):
        assert mask[26:33, 9:14].all()
        assert mask[27:32, 45:49].all()
        assert mask.sum() == 55


def test_largest_only_component_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="both GT and prediction"):
        ArcMaskConfig(pred_component_mode="largest")


def test_cli_rejects_largest_only_component_policy(monkeypatch, capsys) -> None:
    import evaluate_lensing_arc_metrics as arc

    monkeypatch.setattr(sys, "argv", ["evaluate_lensing_arc_metrics.py", "--pred-component-mode", "largest"])
    with pytest.raises(SystemExit) as error:
        arc.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
