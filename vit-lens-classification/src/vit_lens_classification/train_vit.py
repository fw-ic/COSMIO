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
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vit_lens_classification.config import load_config
from vit_lens_classification.data import create_dataloader
from vit_lens_classification.model import build_model
from vit_lens_classification.runtime import (
    checkpoint_metric_key,
    resolve_device,
    run_evaluation,
    save_json,
    save_jsonl,
    seed_everything,
)


def _image_size_from_config(config: dict[str, Any]) -> int:
    data_image_size = config["data"].get("image_size")
    model_image_size = config["model"].get("image_size")
    if data_image_size is None and model_image_size is None:
        raise ValueError("Config must define data.image_size or model.image_size.")
    image_size = int(model_image_size if model_image_size is not None else data_image_size)
    config["data"]["image_size"] = image_size
    config["model"]["image_size"] = image_size
    return image_size


def train(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("epochs", "max_train_steps", "max_eval_batches"):
        value = config["train"].get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"{key} must be positive, got {value}.")
    seed_everything(int(config["train"].get("seed", 1337)))
    image_size = _image_size_from_config(config)
    output_dir = Path(config["experiment"]["output_root"]) / str(config["experiment"]["name"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["train"].get("device", "auto"))
    train_loader = create_dataloader(
        manifest_path=config["data"]["manifest_path"],
        split="train",
        image_size=image_size,
        batch_size=int(config["train"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
        balanced_sampling=bool(config["train"].get("balanced_sampling", True)),
        training=True,
    )
    eval_loader = create_dataloader(
        manifest_path=config["data"]["manifest_path"],
        split=str(config["evaluation"].get("split", "eval")),
        image_size=image_size,
        batch_size=int(config["train"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
        balanced_sampling=False,
    )

    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 1.0e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1.0e-4)),
    )
    criterion = nn.CrossEntropyLoss()
    max_train_steps = config["train"].get("max_train_steps")
    max_eval_batches = config["train"].get("max_eval_batches")
    checkpoint_every_epochs = int(config["train"].get("checkpoint_every_epochs", 0) or 0)

    history: list[dict[str, Any]] = []
    periodic_checkpoints: list[str] = []
    best_metric = float("-inf")
    best_checkpoint_written = False
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch_index, batch in enumerate(train_loader):
            if max_train_steps is not None and batch_index >= int(max_train_steps):
                break
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device))
            labels = batch["label"].to(device)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite classifier training loss at epoch={epoch}, batch={batch_index}.")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1

        eval_metrics, _eval_rows = run_evaluation(
            model=model,
            loader=eval_loader,
            device=device,
            threshold=float(config["evaluation"].get("threshold", 0.5)),
            max_batches=int(max_eval_batches) if max_eval_batches is not None else None,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(epoch_loss / max(steps, 1)),
            "train_steps": int(steps),
            "eval_metrics": eval_metrics,
        }
        history.append(row)

        checkpoint_payload = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "eval_metrics": eval_metrics,
        }
        torch.save(checkpoint_payload, last_checkpoint_path)
        if checkpoint_every_epochs > 0 and epoch % checkpoint_every_epochs == 0:
            periodic_checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(checkpoint_payload, periodic_checkpoint_path)
            periodic_checkpoints.append(str(periodic_checkpoint_path))
        metric_value = checkpoint_metric_key(eval_metrics)
        # A NaN metric — which is what an empty or single-class eval split yields —
        # fails every comparison, so best.pt is silently never written.
        if math.isfinite(metric_value) and metric_value >= best_metric:
            best_metric = metric_value
            best_checkpoint_written = True
            torch.save(checkpoint_payload, best_checkpoint_path)

    history_path = output_dir / "history.jsonl"
    save_jsonl(history_path, history)

    summary = {
        "output_dir": str(output_dir),
        "history_path": str(history_path),
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_written else None,
        "last_checkpoint": str(last_checkpoint_path),
        "periodic_checkpoints": periodic_checkpoints,
        "checkpoint_every_epochs": int(checkpoint_every_epochs),
        "best_metric": float(best_metric) if best_checkpoint_written else None,
        "selection_metric": "auroc",
        "status": "completed",
        "epochs": int(config["train"]["epochs"]),
    }
    save_json(output_dir / "train_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Roman-only ViT lens-vs-nonlens classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="Optional runtime device override, for example `cuda`, `cuda:0`, or `cpu`.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.setdefault("train", {})
        config["train"]["device"] = str(args.device)
    summary = train(config)
    print(summary["best_checkpoint"])


if __name__ == "__main__":
    main()

