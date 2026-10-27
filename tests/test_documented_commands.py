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

"""Check that the commands printed in README.md are the commands that exist.

The unit suites cover library internals and never invoke an entry point, so a
documented invocation could name a flag no parser defined and still pass CI.
That is exactly what happened: the downstream track opened with
``vit_lens_build_real_roman_manifest --config <yaml>``, which exited 2, and
the asyndiff commands passed a --config path that resolved to a doubled
directory. Both were invisible to a suite that only imports modules.

These tests parse the shell blocks out of README.md and hold them against each
command's own --help output, so documentation drift fails the build.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

CONSOLE_SCRIPTS = {
    "asyndiff_build_index",
    "asyndiff_train_diffusion",
    "asyndiff_sample",
    "asyndiff_render_curves",
    "vit_lens_build_real_roman_manifest",
    "vit_lens_build_imputed_roman_manifest",
    "vit_lens_train",
    "vit_lens_evaluate",
    "vit_lens_visualize_imputed_roman",
}
REPO_SCRIPTS = {
    "evaluate_imputation_methods.py",
    "evaluate_lensing_arc_metrics.py",
}


def _shell_lines(text: str) -> list[str]:
    lines: list[str] = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def _resolve(script: str) -> str | None:
    """Locate an installed console script.

    shutil.which alone is not enough: under ``python -m pytest`` the
    environment's bin directory need not be on PATH, and a skip here would
    silently drop the only checks that exercise the entry points.
    """
    found = shutil.which(script)
    if found:
        return found
    sibling = Path(sys.executable).parent / script
    return str(sibling) if sibling.exists() else None


def _documented_commands() -> list[tuple[str, list[str], list[str]]]:
    commands: list[tuple[str, list[str], list[str]]] = []
    for line in _shell_lines(README.read_text(encoding="utf-8")):
        tokens = line.split()
        flags = sorted({token for token in tokens if token.startswith("--")})
        if tokens[0] in CONSOLE_SCRIPTS:
            commands.append((tokens[0], [tokens[0]], flags))
        elif tokens[0] in {"python", "python3"} and len(tokens) > 1 and tokens[1] in REPO_SCRIPTS:
            commands.append((tokens[1], [sys.executable, str(ROOT / tokens[1])], flags))
    return commands


DOCUMENTED = _documented_commands()
IDS = [f"{index}-{name}" for index, (name, _, _) in enumerate(DOCUMENTED)]


def test_readme_documents_at_least_one_command() -> None:
    assert DOCUMENTED, "No runnable commands parsed from README.md shell blocks."


@pytest.mark.parametrize(("name", "argv", "flags"), DOCUMENTED, ids=IDS)
def test_documented_flags_are_accepted(name: str, argv: list[str], flags: list[str]) -> None:
    if argv[0] != sys.executable:
        resolved = _resolve(argv[0])
        if resolved is None:
            pytest.skip(f"{name} is not installed; run `pip install -e` for both packages.")
        argv = [resolved, *argv[1:]]
    completed = subprocess.run(
        [*argv, "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0, f"{name} --help failed:\n{completed.stderr}"
    missing = [flag for flag in flags if flag not in completed.stdout]
    assert not missing, f"README documents {missing} for `{name}`, which its parser does not define."


@pytest.mark.parametrize(("name", "argv", "flags"), DOCUMENTED, ids=IDS)
def test_documented_config_paths_exist(name: str, argv: list[str], flags: list[str]) -> None:
    for line in _shell_lines(README.read_text(encoding="utf-8")):
        tokens = line.split()
        if not tokens or (tokens[0] != name and (len(tokens) < 2 or tokens[1] != name)):
            continue
        for index, token in enumerate(tokens[:-1]):
            if token == "--config":
                config = ROOT / tokens[index + 1]
                assert config.is_file(), f"README passes --config {tokens[index + 1]} to `{name}`, which does not exist."


def _documented_config_paths(prefix: str) -> list[str]:
    paths: list[str] = []
    for line in _shell_lines(README.read_text(encoding="utf-8")):
        tokens = line.split()
        if not tokens[0].startswith(prefix):
            continue
        for index, token in enumerate(tokens[:-1]):
            if token == "--config":
                paths.append(tokens[index + 1])
    return paths


@pytest.mark.parametrize("config_path", _documented_config_paths("asyndiff_"))
def test_documented_asyndiff_configs_resolve(config_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load each documented --config through the CLI's own resolution path.

    --help cannot catch this class of defect: the flag exists, the file exists,
    and the command still dies at runtime because the path is resolved against
    the wrong root. _load_config is private, but it is the code under test --
    calling load_with_overrides directly would bypass the bug entirely.
    """
    from argparse import Namespace

    from asyndiff import cli

    monkeypatch.chdir(ROOT)
    config = cli._load_config(Namespace(config=config_path))
    assert isinstance(config, dict) and config, f"{config_path} resolved to an empty config."


@pytest.mark.parametrize("config_path", _documented_config_paths("vit_lens_"))
def test_documented_vit_configs_resolve(config_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from vit_lens_classification.config import load_config

    monkeypatch.chdir(ROOT)
    config = load_config(config_path)
    assert isinstance(config, dict) and config, f"{config_path} resolved to an empty config."


def _all_console_scripts() -> list[str]:
    return sorted(CONSOLE_SCRIPTS)


@pytest.mark.parametrize("script", _all_console_scripts())
def test_every_console_script_is_importable(script: str) -> None:
    """--help must work for every entry point, documented or not.

    vit_lens_visualize_imputed_roman is declared in pyproject.toml but appears
    in no README block, so the documented-command tests above never reach it.
    It imports matplotlib, which its own package did not declare -- a defect
    invisible to any check driven by the README alone.
    """
    resolved = _resolve(script)
    if resolved is None:
        pytest.skip(f"{script} is not installed; run `pip install -e` for both packages.")
    completed = subprocess.run([resolved, "--help"], capture_output=True, text=True, cwd=ROOT, check=False)
    assert completed.returncode == 0, f"{script} --help failed:\n{completed.stderr}"
