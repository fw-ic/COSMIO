#!/usr/bin/env bash
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

set -euo pipefail

# Resolve every path from the script's own location so the pipeline runs from
# any working directory. Previously ROOT was the literal string
# "asyn-diffusion", which only resolved when the script was invoked from the
# repository root -- never from the directory the script itself lives in.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

ROOT="$(basename "$SCRIPT_DIR")"
CONFIG="${CONFIG:-$ROOT/configs/train.yaml}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# `-x` tests a filesystem path, so a bare command name such as `python3` always
# failed the old check and was silently replaced by `python`.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python interpreter found on PATH. Set PYTHON_BIN." >&2
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN='$PYTHON_BIN' is not on PATH and is not an executable file." >&2
  exit 1
fi

build_index="${build_index:-1}"
train_diffusion="${train_diffusion:-1}"

if [[ "$build_index" == "1" ]]; then
  "$PYTHON_BIN" -m asyndiff.cli build-index --config "$CONFIG"
fi

if [[ "$train_diffusion" == "1" ]]; then
  "$PYTHON_BIN" -m asyndiff.cli train-diffusion --config "$CONFIG"
fi
