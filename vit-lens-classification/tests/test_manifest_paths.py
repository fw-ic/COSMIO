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

"""Coverage for shard-path resolution in manifest.py.

Every other fixture writes manifests with absolute shard paths, so the
relative-path branches are unexercised by them. docs/data_schema.md states that
relative paths are preferred for portable dataset releases, which makes these the branches
a user who downloads the published dataset takes first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vit_lens_classification.manifest import (
    _resolve_manifest_shard_path,
    build_lens_entries,
    build_record_id,
)

from .helpers import write_lens_fixture


# --------------------------------------------------------------------------- #
# build_record_id
# --------------------------------------------------------------------------- #

def test_record_id_is_relative_to_the_data_root(tmp_path: Path) -> None:
    root = tmp_path / "lens"
    shard = root / "run_000" / "shards" / "lens" / "lens_000000.zarr"
    assert build_record_id(root, shard, 3) == "run_000/shards/lens/lens_000000.zarr:3"


def test_record_id_falls_back_to_the_run_prefix_outside_the_root(tmp_path: Path) -> None:
    """A shard outside the declared root is keyed from its run_ component."""
    root = tmp_path / "lens"
    shard = tmp_path / "elsewhere" / "run_004" / "shards" / "lens" / "lens_000000.zarr"
    assert build_record_id(root, shard, 7) == "run_004/shards/lens/lens_000000.zarr:7"


def test_record_id_falls_back_to_the_bare_filename(tmp_path: Path) -> None:
    root = tmp_path / "lens"
    shard = tmp_path / "elsewhere" / "lens_000000.zarr"
    assert build_record_id(root, shard, 0) == "lens_000000.zarr:0"


def test_record_id_is_posix_on_every_platform(tmp_path: Path) -> None:
    root = tmp_path / "lens"
    shard = root / "run_000" / "shards" / "lens" / "lens_000000.zarr"
    assert "\\" not in build_record_id(root, shard, 1)


# --------------------------------------------------------------------------- #
# _resolve_manifest_shard_path
# --------------------------------------------------------------------------- #

def test_relative_path_resolves_under_the_root(tmp_path: Path) -> None:
    root = tmp_path / "lens"
    rel = Path("run_000/shards/lens/lens_000000.zarr")
    (root / rel).mkdir(parents=True)
    assert _resolve_manifest_shard_path(root, rel) == root / rel


def test_run_prefixed_path_resolves_when_the_leading_components_differ(tmp_path: Path) -> None:
    """A manifest written on another machine may carry stale leading directories."""
    root = tmp_path / "lens"
    actual = root / "run_000" / "shards" / "lens" / "lens_000000.zarr"
    actual.mkdir(parents=True)
    stale = Path("some/other/prefix/run_000/shards/lens/lens_000000.zarr")
    assert _resolve_manifest_shard_path(root, stale) == actual


def test_absolute_path_is_returned_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "lens"
    shard = tmp_path / "anywhere" / "lens_000000.zarr"
    assert _resolve_manifest_shard_path(root, shard) == shard


def test_unresolvable_path_returns_a_candidate_rather_than_raising(tmp_path: Path) -> None:
    root = tmp_path / "lens"
    resolved = _resolve_manifest_shard_path(root, Path("run_009/shards/lens/missing.zarr"))
    assert not resolved.exists()
    assert "missing.zarr" in resolved.name


# --------------------------------------------------------------------------- #
# end to end, through a manifest that uses relative paths
# --------------------------------------------------------------------------- #

def _rewrite_manifest_paths_as_relative(lens_root: Path, run_name: str) -> None:
    manifest_path = lens_root / run_name / "shards" / "manifest_lens.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in payload["results"]:
        item["path"] = Path(item["path"]).relative_to(lens_root).as_posix()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def test_lens_entries_build_from_a_manifest_with_relative_paths(tmp_path: Path) -> None:
    lens_root = write_lens_fixture(tmp_path)
    _rewrite_manifest_paths_as_relative(lens_root, "run_000")

    entries = build_lens_entries(lens_root, "run_000", split="eval")

    assert entries, "a manifest with relative shard paths must still yield entries"
    assert all(entry["label"] == 1 for entry in entries)
    for entry in entries:
        assert not Path(entry["shard_path"]).is_absolute() or Path(entry["shard_path"]).exists()


def test_relative_and_absolute_manifests_agree(tmp_path: Path) -> None:
    """The two manifest styles must describe the same samples."""
    absolute_root = write_lens_fixture(tmp_path / "abs")
    relative_root = write_lens_fixture(tmp_path / "rel")
    _rewrite_manifest_paths_as_relative(relative_root, "run_000")

    from_absolute = build_lens_entries(absolute_root, "run_000", split="eval")
    from_relative = build_lens_entries(relative_root, "run_000", split="eval")

    assert len(from_absolute) == len(from_relative)
    assert [e["sample_id"] for e in from_absolute] == [e["sample_id"] for e in from_relative]
    assert [e["local_index"] for e in from_absolute] == [e["local_index"] for e in from_relative]


def test_missing_shard_in_a_manifest_is_reported(tmp_path: Path) -> None:
    lens_root = write_lens_fixture(tmp_path)
    manifest_path = lens_root / "run_000" / "shards" / "manifest_lens.json"
    manifest_path.write_text(json.dumps({"results": [], "summary": {"ok": 0}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No lens shards found"):
        build_lens_entries(lens_root, "run_000", split="eval")
