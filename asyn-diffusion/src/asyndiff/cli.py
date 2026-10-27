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

from asyndiff.config import load_with_overrides
from asyndiff.data.index import build_and_store_artifacts
from asyndiff.ops.visualization import render_training_curves
from asyndiff.training import sample_prediction, train_asyn_diffusion


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/train.yaml")
    return parser


def _load_config(args: argparse.Namespace) -> dict:
    # Delegate resolution to load_with_overrides, which searches the working
    # directory, the package root, and the repository root in that order.
    # Pre-joining a root here would make the path absolute and defeat that
    # search, so a relative --config was resolved against one root only.
    return load_with_overrides(args.config)


def build_index_main() -> None:
    parser = _common_parser("Build sample index, channel stats, and visualization subsets.")
    args = parser.parse_args()
    build_and_store_artifacts(_load_config(args))


def train_diffusion_main() -> None:
    parser = _common_parser("Train the asyn-diffusion baseline.")
    args = parser.parse_args()
    train_asyn_diffusion(_load_config(args))


def sample_main() -> None:
    parser = _common_parser("Sample missing-modality reconstructions from a trained asyn-diffusion checkpoint.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument(
        "--known",
        default=None,
        help="Comma-separated known modalities (one of: lsst,euclid | lsst | euclid | lsst,roman).",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = _load_config(args)
    split = args.split or "val"
    index = int(args.index if args.index is not None else 0)
    known = args.known.split(",") if args.known is not None else ["lsst", "euclid"]
    checkpoint = Path(args.checkpoint) if args.checkpoint is not None else None
    output_path = Path(args.output) if args.output is not None else None
    print(
        sample_prediction(
            config,
            split=split,
            sample_index=index,
            known_modalities=known,
            checkpoint_path=checkpoint,
            output_path=output_path,
        )
    )


def render_curves_main() -> None:
    parser = argparse.ArgumentParser(description="Render PNG curves from a metrics JSONL file.")
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render_training_curves(Path(args.history), Path(args.output))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index")
    build.add_argument("--config", default="configs/train.yaml")

    train = subparsers.add_parser("train-diffusion")
    train.add_argument("--config", default="configs/train.yaml")

    sample = subparsers.add_parser("sample")
    sample.add_argument("--config", default="configs/train.yaml")
    sample.add_argument("--split", default=None)
    sample.add_argument("--index", type=int, default=None)
    sample.add_argument("--known", default=None)
    sample.add_argument("--checkpoint", default=None)
    sample.add_argument("--output", default=None)

    curves = subparsers.add_parser("render-curves")
    curves.add_argument("--history", required=True)
    curves.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "build-index":
        build_and_store_artifacts(_load_config(args))
        return
    if args.command == "train-diffusion":
        train_asyn_diffusion(_load_config(args))
        return
    if args.command == "sample":
        config = _load_config(args)
        split = args.split or "val"
        index = int(args.index if args.index is not None else 0)
        known = args.known.split(",") if args.known is not None else ["lsst", "euclid"]
        checkpoint = Path(args.checkpoint) if args.checkpoint is not None else None
        output_path = Path(args.output) if args.output is not None else None
        print(
            sample_prediction(
                config,
                split=split,
                sample_index=index,
                known_modalities=known,
                checkpoint_path=checkpoint,
                output_path=output_path,
            )
        )
        return
    if args.command == "render-curves":
        render_training_curves(Path(args.history), Path(args.output))


if __name__ == "__main__":
    main()

