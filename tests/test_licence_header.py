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

"""Every code file must carry the Futurewei licence header, in first position.

Header coverage is the kind of thing that is correct on the day it is applied
and silently wrong three commits later, when someone adds a file. Enforcing it
here means a missing header fails the build rather than a later audit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MARKER = "Copyright (c) 2026 Futurewei Technologies, Inc."

COMMENTABLE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".html", ".css"}
COMMENTABLE_NAMES = {"requirements.txt"}


def _tracked_files() -> list[Path] | None:
    """Ask git what is in the repository.

    Walking the filesystem instead would sweep in anything a developer happens
    to leave in the tree -- a virtualenv not named .venv, a vendored checkout, a
    coverage report -- and parametrize a failing test over every file inside it.
    A venv named `env` produced thousands of failures before this change.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
    return [ROOT / name for name in names]


SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", ".env",
    "build", "dist", "site-packages", "node_modules", ".tox",
}


def _commentable_files() -> list[Path]:
    tracked = _tracked_files()
    if not tracked:
        # Either not a git checkout (an unpacked sdist), or a checkout that
        # tracks nothing here -- which is what happens when a release tarball is
        # unpacked inside somebody else's working tree, where `git ls-files`
        # exits 0 and lists nothing. Both must fall through to the walk; testing
        # `is None` alone left the file list empty and every header assertion
        # silently unparametrized.
        tracked = []
        for path in sorted(ROOT.rglob("*")):
            # Match on the path RELATIVE to ROOT. Matching absolute parts means
            # an unrelated ancestor directory -- /build, ~/env, any dist/ --
            # excludes the entire tree.
            if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
                continue
            tracked.append(path)
    found = []
    for path in tracked:
        if not path.is_file():
            continue
        if path.suffix in COMMENTABLE_SUFFIXES or path.name in COMMENTABLE_NAMES:
            found.append(path)
    return sorted(found)


FILES = _commentable_files()
IDS = [str(path.relative_to(ROOT)) for path in FILES]


def test_repository_has_commentable_files() -> None:
    assert FILES, "No commentable files discovered; the walk is misconfigured."


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_file_carries_licence_header(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert MARKER in text[:4000], f"{path.relative_to(ROOT)} is missing the Futurewei licence header."
    assert "SPDX-License-Identifier: MIT" in text[:4000]
    assert "Permission is hereby granted, free of charge" in text[:4000]
    assert "you may not use this file except in compliance" not in text.split("Permission is hereby granted", 1)[0]


def _opening_token(path: Path) -> str:
    if path.suffix == ".html":
        return "<!--"
    if path.suffix == ".css":
        return "/*"
    return "# ===="


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_header_is_in_first_position(path: Path) -> None:
    """Only a shebang or an HTML doctype may precede the header block."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines, f"{path.relative_to(ROOT)} is empty."

    offset = 0
    first = lines[0].lstrip().lower()
    if lines[0].startswith("#!") or first.startswith("<!doctype"):
        offset = 1

    assert lines[offset].startswith(_opening_token(path)), (
        f"{path.relative_to(ROOT)} does not open with the licence header; "
        f"line {offset + 1} is {lines[offset]!r}. Only a shebang or an HTML "
        "doctype may come before it."
    )
    window = "\n".join(lines[offset : offset + 6])
    assert MARKER in window, (
        f"{path.relative_to(ROOT)} opens with a comment block that is not the "
        "Futurewei header."
    )
