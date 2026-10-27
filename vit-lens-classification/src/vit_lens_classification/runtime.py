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

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vit_lens_classification.metrics import binary_classification_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None or str(device_name).strip().lower() in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device_name))


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    return output_path


def _evaluation_subset(loader: torch.utils.data.DataLoader, max_batches: int | None):
    if max_batches is None:
        return loader, None
    if int(max_batches) <= 0 or not loader.batch_size:
        raise ValueError("max_eval_batches and batch_size must be positive.")
    entries = getattr(loader.dataset, "entries", None)
    if entries is None:
        raise ValueError("Limited classifier evaluation requires dataset.entries with labels and record_id.")
    groups = {label: sorted((i for i, item in enumerate(entries) if int(item["label"]) == label),
                           key=lambda i: str(entries[i]["record_id"])) for label in (0, 1)}
    if not all(groups.values()) or any(int(item["label"]) not in groups for item in entries):
        raise ValueError("Classifier evaluation requires both labels 0 and 1.")
    capacity = min(len(entries), int(max_batches) * int(loader.batch_size))
    if capacity < 2:
        raise ValueError("Limited classifier evaluation must allow at least two examples, one per class.")
    n0 = min(len(groups[0]), max(1, round(capacity * len(groups[0]) / len(entries))))
    n0 = max(n0, capacity - len(groups[1]))
    n0 = min(n0, capacity - 1)
    rng = np.random.default_rng(1337)
    indices = sorted(int(i) for label, count in ((0, n0), (1, capacity - n0))
                     for i in rng.choice(groups[label], size=count, replace=False))
    subset_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(loader.dataset, indices), batch_size=loader.batch_size,
        shuffle=False, num_workers=loader.num_workers, collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory, drop_last=False,
    )
    report = {"selection": "fixed-stratified-v1", "seed": 1337,
              "available_count": len(entries), "evaluated_count": len(indices),
              "class_counts": {"0": n0, "1": capacity - n0},
              "record_ids": [str(entries[i]["record_id"]) for i in indices]}
    return subset_loader, report


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loader, subset_report = _evaluation_subset(loader, max_batches)
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        logits = model(batch["image"].to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        if not np.isfinite(probs).all():
            raise ValueError("Classifier produced nonfinite evaluation probabilities.")
        batch_labels = batch["label"].detach().cpu().numpy().astype(np.int64)
        probabilities.append(probs)
        labels.append(batch_labels)
        for record_id, sample_id, label, prob in zip(
            batch["record_id"],
            batch["sample_id"],
            batch_labels.tolist(),
            probs.tolist(),
        ):
            rows.append(
                {
                    "record_id": str(record_id),
                    "sample_id": str(sample_id),
                    "label": int(label),
                    "prob_lens": float(prob),
                    "pred_label": int(prob >= float(threshold)),
                }
            )

    if not probabilities:
        raise ValueError("Classifier evaluation produced no examples.")

    all_probs = np.concatenate(probabilities, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    if set(all_labels.tolist()) != {0, 1}:
        raise ValueError("Classifier evaluation requires both labels 0 and 1 in the scored examples.")
    metrics = binary_classification_metrics(all_labels, all_probs, threshold=threshold)
    if subset_report is not None:
        metrics["evaluation_subset"] = subset_report
    return metrics, rows


def checkpoint_metric_key(metrics: dict[str, Any]) -> float:
    value = metrics.get("auroc", float("nan"))
    if value is None or not math.isfinite(float(value)):
        raise ValueError("Cannot select a classifier checkpoint without finite AUROC; accuracy is not a substitute.")
    return float(value)
