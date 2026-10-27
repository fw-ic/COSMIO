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

"""Versioned per-record random streams for evaluation, independent of batching."""
from __future__ import annotations

import hashlib
import json
import torch

SEED_SCHEME = "sha256-record-draw-v1"


def sample_seed(seed: int, record_id: str, draw: int) -> int:
    if not record_id or draw < 0:
        raise ValueError("Sampling needs a nonempty record_id and a nonnegative draw index.")
    payload = json.dumps([SEED_SCHEME, int(seed), str(record_id), int(draw)], ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def sample_generators(seeds, device):
    if seeds is None:
        return None
    return [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]


def randn_per_sample(shape, *, device, dtype=None, generators=None):
    if generators is None:
        return torch.randn(shape, device=device, dtype=dtype)
    if len(generators) != int(shape[0]):
        raise ValueError("One random generator is required per batch record.")
    return torch.stack([
        torch.randn(tuple(shape[1:]), device=device, dtype=dtype, generator=generator)
        for generator in generators
    ])


def randn_like_per_sample(tensor, generators=None):
    return randn_per_sample(tensor.shape, device=tensor.device, dtype=tensor.dtype, generators=generators)
