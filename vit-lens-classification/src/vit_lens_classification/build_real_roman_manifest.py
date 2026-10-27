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

from vit_lens_classification.manifest import (
    build_lens_entries,
    build_manifest_payload,
    validate_classifier_splits,
    build_nonlens_entries,
    build_nonlens_run_entries,
    write_json,
)


DEFAULT_NONLENS_HALF_EXAMPLES = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Roman-only real-lens vs nonlens classifier manifest.")
    parser.add_argument(
        "--nonlens-index-path",
        default="asyn-diffusion/artifacts/sample_index.json",
        help="Legacy nonlens split index used when --nonlens-source index.",
    )
    parser.add_argument(
        "--nonlens-source",
        choices=("run", "index"),
        default="run",
        help="Source for real nonlens negatives. Default: run_007 first/second 8000 split.",
    )
    parser.add_argument(
        "--nonlens-root",
        default="data/regular",
        help="Normalized nonlens dataset root used when --nonlens-source run.",
    )
    parser.add_argument("--nonlens-run", default="run_007")
    parser.add_argument("--train-negative-start", type=int, default=0)
    parser.add_argument("--train-negative-count", type=int, default=DEFAULT_NONLENS_HALF_EXAMPLES)
    parser.add_argument("--eval-negative-start", type=int, default=DEFAULT_NONLENS_HALF_EXAMPLES)
    parser.add_argument("--eval-negative-count", type=int, default=DEFAULT_NONLENS_HALF_EXAMPLES)
    parser.add_argument(
        "--lens-root",
        default="data/lens",
        help="Normalized lens dataset root.",
    )
    parser.add_argument(
        "--normalization-metadata",
        default="data/lens/normalization_metadata.json",
        help="Normalization metadata path recorded in the output manifest.",
    )
    parser.add_argument("--train-run", default="run_000")
    parser.add_argument("--eval-run", default="run_001")
    parser.add_argument("--target-group", default="images")
    parser.add_argument(
        "--output",
        default="vit-lens-classification/artifacts/real_roman_manifest.json",
    )
    return parser.parse_args()


def build_manifest(args: argparse.Namespace) -> Path:
    entries = []
    split_reports: list = []
    nonlens_source = str(getattr(args, "nonlens_source", "run"))
    if nonlens_source == "index":
        entries.extend(build_nonlens_entries(args.nonlens_index_path, reports=split_reports))
    else:
        entries.extend(
            build_nonlens_run_entries(
                nonlens_root=args.nonlens_root,
                run_names=[str(args.nonlens_run)],
                split="train",
                start=int(getattr(args, "train_negative_start", 0)),
                count=getattr(args, "train_negative_count", DEFAULT_NONLENS_HALF_EXAMPLES),
            )
        )
        entries.extend(
            build_nonlens_run_entries(
                nonlens_root=args.nonlens_root,
                run_names=[str(args.nonlens_run)],
                split="eval",
                start=int(getattr(args, "eval_negative_start", DEFAULT_NONLENS_HALF_EXAMPLES)),
                count=getattr(args, "eval_negative_count", DEFAULT_NONLENS_HALF_EXAMPLES),
            )
        )
    entries.extend(
        build_lens_entries(
            args.lens_root,
            run_name=str(args.train_run),
            split="train",
            target_group=str(args.target_group),
            source_kind="lens_normalized",
        )
    )
    entries.extend(
        build_lens_entries(
            args.lens_root,
            run_name=str(args.eval_run),
            split="eval",
            target_group=str(args.target_group),
            source_kind="lens_normalized",
        )
    )
    validate_classifier_splits(entries)
    payload = build_manifest_payload(
        track="real_roman",
        normalization_metadata_path=args.normalization_metadata,
        entries=entries,
        extra={
            "source_split_reports": split_reports,
            "lens_root": str(args.lens_root),
            "train_run": str(args.train_run),
            "eval_run": str(args.eval_run),
            "target_group": str(args.target_group),
            "nonlens_index_path": str(args.nonlens_index_path),
            "nonlens_source": nonlens_source,
            "nonlens_root": str(getattr(args, "nonlens_root", None)),
            "nonlens_run": str(getattr(args, "nonlens_run", None)),
            "train_negative_start": int(getattr(args, "train_negative_start", 0)),
            "train_negative_count": getattr(args, "train_negative_count", DEFAULT_NONLENS_HALF_EXAMPLES),
            "eval_negative_start": int(getattr(args, "eval_negative_start", DEFAULT_NONLENS_HALF_EXAMPLES)),
            "eval_negative_count": getattr(args, "eval_negative_count", DEFAULT_NONLENS_HALF_EXAMPLES),
        },
    )
    return write_json(args.output, payload)


def main() -> None:
    output_path = build_manifest(parse_args())
    print(output_path)


if __name__ == "__main__":
    main()

