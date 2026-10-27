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

"""Numerical validation at data and evaluation boundaries."""
from __future__ import annotations

import numpy as np
import torch


def require_finite(value, context: str):
    if isinstance(value, torch.Tensor):
        valid = value.numel() > 0 and bool(torch.isfinite(value).all())
    else:
        array = np.asarray(value)
        valid = array.size > 0 and bool(np.isfinite(array).all())
    if not valid:
        raise ValueError(f"{context}: empty or nonfinite numerical values.")
    return value


def require_finite_batch(tensor, record_ids, context: str):
    if tensor.shape[0] != len(record_ids):
        raise ValueError(f"{context}: record/tensor batch mismatch.")
    if tensor.numel() == 0:
        raise ValueError(f"{context}: empty numerical batch.")
    valid = torch.isfinite(tensor).flatten(1).all(1)
    for index in (~valid).nonzero(as_tuple=False).flatten().tolist():
        raise ValueError(f"{context}: record_id={record_ids[index]} contains nonfinite values.")
