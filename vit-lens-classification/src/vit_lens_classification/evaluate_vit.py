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
from pathlib import Path
from typing import Any

import torch

from vit_lens_classification.config import load_config
from vit_lens_classification.data import create_dataloader
from vit_lens_classification.model import build_model
from vit_lens_classification.runtime import resolve_device, run_evaluation, save_json, save_jsonl


def _image_size_from_config(config: dict[str, Any]) -> int:
    data_image_size = config["data"].get("image_size")
    model_image_size = config["model"].get("image_size")
    if data_image_size is None and model_image_size is None:
        raise ValueError("Config must define data.image_size or model.image_size.")
    image_size = int(model_image_size if model_image_size is not None else data_image_size)
    config["data"]["image_size"] = image_size
    config["model"]["image_size"] = image_size
    return image_size


def evaluate(config: dict[str, Any], checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    image_size = _image_size_from_config(config)
    output_dir = Path(config["experiment"]["output_root"]) / str(config["experiment"]["name"])
    device = resolve_device(config["train"].get("device", "auto"))
    loader = create_dataloader(
        manifest_path=config["data"]["manifest_path"],
        split=str(config["evaluation"].get("split", "eval")),
        image_size=image_size,
        batch_size=int(config["train"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
        balanced_sampling=False,
    )
    model = build_model(config["model"]).to(device)
    resolved_checkpoint = Path(checkpoint_path or output_dir / "checkpoints" / "best.pt")
    state = torch.load(resolved_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    metrics, rows = run_evaluation(
        model=model,
        loader=loader,
        device=device,
        threshold=float(config["evaluation"].get("threshold", 0.5)),
        max_batches=int(config["train"].get("max_eval_batches")) if config["train"].get("max_eval_batches") is not None else None,
    )
    summary = {
        "checkpoint": str(resolved_checkpoint),
        "metrics": metrics,
        "prediction_count": len(rows),
    }
    save_json(output_dir / "eval_summary.json", summary)
    save_jsonl(output_dir / "eval_predictions.jsonl", rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Roman-only ViT lens-vs-nonlens classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, help="Optional runtime device override, for example `cuda`, `cuda:0`, or `cpu`.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.setdefault("train", {})
        config["train"]["device"] = str(args.device)
    summary = evaluate(config, checkpoint_path=args.checkpoint)
    print(summary["checkpoint"])


if __name__ == "__main__":
    main()

