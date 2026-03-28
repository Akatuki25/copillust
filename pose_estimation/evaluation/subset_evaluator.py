"""Subset-aware evaluator implementing BaseEvaluator.

Computes all metrics from docs/task.md, broken down by metadata subsets:
1. Overall AP / PCK
2. Chibi subset visible-joint PCK
3. Lineart subset visible-joint PCK
4. Partial-body subset visible-joint PCK
5. No-prediction / collapse rate
6. Left-right confusion rate
7. Out-of-frame misclassification rate

Usage:
    from pose_estimation.evaluation.subset_evaluator import SubsetEvaluator

    evaluator = SubsetEvaluator()
    results = evaluator.evaluate(predictions, ground_truth)
    results = evaluator.evaluate(predictions, ground_truth, subset_filter={"body_type": "chibi"})
    full_report = evaluator.full_report(predictions, ground_truth)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pose_estimation.core.base_evaluator import BaseEvaluator
from pose_estimation.core.types import PoseResult
from pose_estimation.evaluation.metrics import (
    collapse_rate,
    left_right_confusion_rate,
    oks_ap,
    out_of_frame_misclassification,
    pck,
)

# Standard subsets to evaluate, based on docs/task.md.
DEFAULT_SUBSETS: list[dict[str, Any]] = [
    {"name": "overall", "filter": None},
    {"name": "chibi", "filter": {"body_type": "chibi"}},
    {"name": "lineart", "filter": {"render_type": "lineart"}},
    {"name": "partial_body", "filter": {"frame_type": "truncated_bottom"}},
    {"name": "drawing", "filter": {"render_type": "drawing"}},
]


class SubsetEvaluator(BaseEvaluator):
    """Evaluate pose estimation with per-subset breakdown."""

    def __init__(self, subsets: list[dict[str, Any]] | None = None) -> None:
        self.subsets = subsets or DEFAULT_SUBSETS

    def evaluate(
        self,
        predictions: list[PoseResult],
        ground_truth: list[PoseResult],
        subset_filter: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Compute all metrics, optionally filtered to a subset.

        Returns a flat dict of metric_name → value.
        """
        if subset_filter:
            predictions, ground_truth = self.filter_by_metadata(
                predictions, ground_truth, subset_filter
            )

        if not predictions or not ground_truth:
            return {"n_samples": 0}

        results: dict[str, float] = {"n_samples": float(len(predictions))}

        # PCK.
        pck_results = pck(predictions, ground_truth)
        results.update(pck_results)

        # OKS AP.
        ap_results = oks_ap(predictions, ground_truth)
        results.update(ap_results)

        # Collapse rate.
        collapse_results = collapse_rate(predictions)
        results.update(collapse_results)

        # Left-right confusion.
        lr_results = left_right_confusion_rate(predictions, ground_truth)
        results.update(lr_results)

        # Out-of-frame misclassification.
        oof_results = out_of_frame_misclassification(predictions, ground_truth)
        results.update(oof_results)

        return results

    def full_report(
        self,
        predictions: list[PoseResult],
        ground_truth: list[PoseResult],
    ) -> pd.DataFrame:
        """Compute metrics for all subsets and return as a DataFrame.

        Returns:
            DataFrame with one row per subset and metric columns.
        """
        rows: list[dict[str, Any]] = []

        for subset_cfg in self.subsets:
            name = subset_cfg["name"]
            filt = subset_cfg["filter"]
            metrics = self.evaluate(predictions, ground_truth, filt)
            metrics["subset"] = name
            rows.append(metrics)

        df = pd.DataFrame(rows)
        # Reorder columns.
        cols = ["subset", "n_samples"]
        cols += [c for c in df.columns if c not in cols]
        return df[cols]

    def print_report(
        self,
        predictions: list[PoseResult],
        ground_truth: list[PoseResult],
    ) -> None:
        """Print a formatted evaluation report to stdout."""
        df = self.full_report(predictions, ground_truth)
        print("\n" + "=" * 80)
        print("POSE ESTIMATION EVALUATION REPORT")
        print("=" * 80)
        print(df.to_string(index=False, float_format="{:.4f}".format))
        print("=" * 80)
