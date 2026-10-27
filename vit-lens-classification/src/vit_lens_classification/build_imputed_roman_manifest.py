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
    build_imputed_entries,
    build_manifest_payload,
    validate_classifier_splits,
    build_nonlens_entries_for_splits,
    build_nonlens_run_entries,
    build_prediction_entries,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Roman-only imputed-lens vs nonlens classifier manifest.")
    parser.add_argument(
        "--nonlens-index-path",
        default="asyn-diffusion/artifacts/sample_index.json",
        help="Nonlens split index used to source train negatives and eval negatives.",
    )
    parser.add_argument(
        "--train-negative-source",
        choices=("real", "imputed", "mixed"),
        default="real",
        help="Negative train source. Default: real nonlens train split from --nonlens-index-path.",
    )
    parser.add_argument(
        "--train-negative-results-json",
        default=None,
        help="Evaluation JSON with saved imputed nonlens Roman predictions. Required when --train-negative-source imputed.",
    )
    parser.add_argument(
        "--train-negative-start",
        type=int,
        default=0,
        help="Start offset inside --train-negative-results-json examples.",
    )
    parser.add_argument(
        "--train-negative-count",
        type=int,
        default=None,
        help="Optional number of examples from --train-negative-results-json to use for train negatives.",
    )
    parser.add_argument(
        "--train-negative-real-root",
        default=None,
        help="Normalized nonlens root used for real train negatives when --train-negative-source mixed.",
    )
    parser.add_argument(
        "--train-negative-real-runs",
        default=None,
        help="Comma-separated normalized nonlens runs used for real train negatives when --train-negative-source mixed.",
    )
    parser.add_argument(
        "--train-results-json",
        required=True,
        help="Evaluation JSON produced from lens run_000 with --save-pred-sample-normalized.",
    )
    parser.add_argument(
        "--eval-results-json",
        default=None,
        help="Optional evaluation JSON for imputed eval positives. Only used when --eval-source imputed.",
    )
    parser.add_argument(
        "--eval-negative-source",
        choices=("index", "imputed", "real"),
        default="index",
        help="Negative eval source. Default: nonlens val split from --nonlens-index-path.",
    )
    parser.add_argument(
        "--eval-negative-results-json",
        default=None,
        help="Evaluation JSON with saved imputed nonlens Roman predictions for eval negatives.",
    )
    parser.add_argument(
        "--eval-negative-start",
        type=int,
        default=0,
        help="Start offset inside --eval-negative-results-json examples.",
    )
    parser.add_argument(
        "--eval-negative-count",
        type=int,
        default=None,
        help="Optional number of examples from --eval-negative-results-json to use for eval negatives.",
    )
    parser.add_argument(
        "--eval-negative-real-root",
        default=None,
        help="Normalized nonlens root used when --eval-negative-source real.",
    )
    parser.add_argument(
        "--eval-negative-real-runs",
        default=None,
        help="Comma-separated normalized nonlens runs used when --eval-negative-source real.",
    )
    parser.add_argument(
        "--normalization-metadata",
        default="data/regular/normalization_metadata.json",
        help="Roman normalization metadata path recorded in the manifest for consistency.",
    )
    parser.add_argument(
        "--eval-source",
        choices=("real", "imputed"),
        default="real",
        help="Positive eval source. Default: real normalized lens data.",
    )
    parser.add_argument(
        "--eval-lens-root",
        default="data/lens",
        help="Normalized lens dataset root used when --eval-source real.",
    )
    parser.add_argument("--eval-run", default="run_001", help="Normalized real-lens eval run used when --eval-source real.")
    parser.add_argument("--method", default=None, help="Optional method name if the evaluation JSON contains multiple methods.")
    parser.add_argument("--target-modality", default="roman")
    parser.add_argument(
        "--output",
        required=True,
        help="Output manifest path; use data.manifest_path from the intended classifier config.",
    )
    return parser.parse_args()


def build_manifest(args: argparse.Namespace) -> Path:
    entries = []
    split_reports: list = []
    train_negative_source = str(getattr(args, "train_negative_source", "real"))
    if train_negative_source == "real":
        entries.extend(
            build_nonlens_entries_for_splits(
                args.nonlens_index_path,
                split_map={"train": "train"},
                reports=split_reports,
            )
        )
    elif train_negative_source == "imputed":
        if args.train_negative_results_json is None:
            raise ValueError("--train-negative-results-json is required when --train-negative-source imputed.")
        entries.extend(
            build_prediction_entries(
                evaluation_results_path=args.train_negative_results_json,
                split="train",
                label=0,
                method_name=args.method,
                target_modality=args.target_modality,
                start=int(getattr(args, "train_negative_start", 0)),
                count=getattr(args, "train_negative_count", None),
            )
        )
    else:
        if args.train_negative_results_json is None:
            raise ValueError("--train-negative-results-json is required when --train-negative-source mixed.")
        if args.train_negative_real_root is None:
            raise ValueError("--train-negative-real-root is required when --train-negative-source mixed.")
        if args.train_negative_real_runs is None:
            raise ValueError("--train-negative-real-runs is required when --train-negative-source mixed.")
        real_runs = [item.strip() for item in str(args.train_negative_real_runs).split(",") if item.strip()]
        if not real_runs:
            raise ValueError("--train-negative-real-runs must include at least one run name.")
        entries.extend(
            build_prediction_entries(
                evaluation_results_path=args.train_negative_results_json,
                split="train",
                label=0,
                method_name=args.method,
                target_modality=args.target_modality,
                start=int(getattr(args, "train_negative_start", 0)),
                count=getattr(args, "train_negative_count", None),
            )
        )
        entries.extend(
            build_nonlens_run_entries(
                nonlens_root=args.train_negative_real_root,
                run_names=real_runs,
                split="train",
            )
        )
    eval_negative_source = str(getattr(args, "eval_negative_source", "index"))
    if eval_negative_source == "index":
        entries.extend(
            build_nonlens_entries_for_splits(
                args.nonlens_index_path,
                split_map={"val": "eval"},
                reports=split_reports,
            )
        )
    elif eval_negative_source == "imputed":
        eval_negative_results_json = getattr(args, "eval_negative_results_json", None)
        if eval_negative_results_json is None:
            raise ValueError("--eval-negative-results-json is required when --eval-negative-source imputed.")
        entries.extend(
            build_prediction_entries(
                evaluation_results_path=eval_negative_results_json,
                split="eval",
                label=0,
                method_name=args.method,
                target_modality=args.target_modality,
                start=int(getattr(args, "eval_negative_start", 0)),
                count=getattr(args, "eval_negative_count", None),
            )
        )
    else:
        if args.eval_negative_real_root is None:
            raise ValueError("--eval-negative-real-root is required when --eval-negative-source real.")
        if args.eval_negative_real_runs is None:
            raise ValueError("--eval-negative-real-runs is required when --eval-negative-source real.")
        eval_real_runs = [item.strip() for item in str(args.eval_negative_real_runs).split(",") if item.strip()]
        if not eval_real_runs:
            raise ValueError("--eval-negative-real-runs must include at least one run name.")
        entries.extend(
            build_nonlens_run_entries(
                nonlens_root=args.eval_negative_real_root,
                run_names=eval_real_runs,
                split="eval",
                start=int(getattr(args, "eval_negative_start", 0)),
                count=getattr(args, "eval_negative_count", None),
            )
        )
    entries.extend(
        build_imputed_entries(
            evaluation_results_path=args.train_results_json,
            split="train",
            method_name=args.method,
            target_modality=args.target_modality,
        )
    )
    if str(args.eval_source) == "real":
        entries.extend(
            build_lens_entries(
                args.eval_lens_root,
                run_name=str(args.eval_run),
                split="eval",
                target_group="images",
                source_kind="lens_normalized",
            )
        )
    else:
        if args.eval_results_json is None:
            raise ValueError("--eval-results-json is required when --eval-source imputed.")
        entries.extend(
            build_imputed_entries(
                evaluation_results_path=args.eval_results_json,
                split="eval",
                method_name=args.method,
                target_modality=args.target_modality,
            )
        )
    validate_classifier_splits(entries)
    payload = build_manifest_payload(
        track="imputed_roman",
        normalization_metadata_path=args.normalization_metadata,
        entries=entries,
        extra={
            "source_split_reports": split_reports,
            "train_results_json": str(args.train_results_json),
            "train_negative_source": train_negative_source,
            "train_negative_results_json": (
                str(getattr(args, "train_negative_results_json"))
                if getattr(args, "train_negative_results_json", None) is not None
                else None
            ),
            "train_negative_start": int(getattr(args, "train_negative_start", 0)),
            "train_negative_count": getattr(args, "train_negative_count", None),
            "train_negative_real_root": (
                str(getattr(args, "train_negative_real_root"))
                if getattr(args, "train_negative_real_root", None) is not None
                else None
            ),
            "train_negative_real_runs": (
                [item.strip() for item in str(getattr(args, "train_negative_real_runs")).split(",") if item.strip()]
                if getattr(args, "train_negative_real_runs", None) is not None
                else None
            ),
            "eval_results_json": str(args.eval_results_json) if args.eval_results_json is not None else None,
            "eval_negative_source": eval_negative_source,
            "eval_negative_results_json": (
                str(getattr(args, "eval_negative_results_json"))
                if getattr(args, "eval_negative_results_json", None) is not None
                else None
            ),
            "eval_negative_start": int(getattr(args, "eval_negative_start", 0)),
            "eval_negative_count": getattr(args, "eval_negative_count", None),
            "eval_negative_real_root": (
                str(getattr(args, "eval_negative_real_root"))
                if getattr(args, "eval_negative_real_root", None) is not None
                else None
            ),
            "eval_negative_real_runs": (
                [item.strip() for item in str(getattr(args, "eval_negative_real_runs")).split(",") if item.strip()]
                if getattr(args, "eval_negative_real_runs", None) is not None
                else None
            ),
            "eval_source": str(args.eval_source),
            "eval_lens_root": str(args.eval_lens_root),
            "eval_run": str(args.eval_run),
            "method": str(args.method) if args.method is not None else None,
            "target_modality": str(args.target_modality),
            "nonlens_index_path": str(args.nonlens_index_path),
        },
    )
    return write_json(args.output, payload)


def main() -> None:
    output_path = build_manifest(parse_args())
    print(output_path)


if __name__ == "__main__":
    main()

