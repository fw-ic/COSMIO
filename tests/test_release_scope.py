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

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
def test_only_asyndiff_imputation_model_is_released() -> None:
    package_roots = {
        pyproject.parent.name
        for pyproject in ROOT.glob("*/pyproject.toml")
    }
    assert package_roots == {"asyn-diffusion", "vit-lens-classification"}
    assert (ROOT / "asyn-diffusion").is_dir()


def test_downstream_classification_is_released() -> None:
    package = ROOT / "vit-lens-classification" / "src" / "vit_lens_classification"
    required = {
        "build_imputed_roman_manifest.py",
        "build_real_roman_manifest.py",
        "train_vit.py",
        "evaluate_vit.py",
        "model.py",
    }
    assert required <= {path.name for path in package.glob("*.py")}


def test_evaluator_has_one_method() -> None:
    tree = ast.parse((ROOT / "evaluate_imputation_methods.py").read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "ALL_METHODS"
    }
    assert assignments["ALL_METHODS"] == ("asyn-diffusion",)


def test_project_page_assets_exist() -> None:
    assert (ROOT / "docs" / "index.html").is_file()
    assert (ROOT / "docs" / "styles.css").is_file()
    assert (ROOT / "docs" / "assets" / "cosmio-teaser.png").is_file()
    assert (ROOT / "docs" / "assets" / "data-sample.png").is_file()
