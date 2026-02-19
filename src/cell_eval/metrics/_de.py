"""DE metrics module."""

from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from .._types import DEComparison, DESortBy


def de_overlap_metric(
    data: DEComparison,
    k: int | None,
    metric: Literal["precision", "overlap"] = "overlap",
    fdr_threshold: float = 0.05,
    sort_by: DESortBy = DESortBy.ABS_FOLD_CHANGE,
) -> dict[str, float]:
    """Compute overlap between real and predicted DE genes.

    Note: use `k` argument for measuring recall and use `topk` argument for measuring precision.

    """
    return data.compute_overlap(
        k=k,
        metric=metric,
        fdr_threshold=fdr_threshold,
        sort_by=sort_by,
    )


class DESpearmanSignificant:
    """Compute Spearman correlation on number of significant DE genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> float:
        """Compute correlation between number of significant genes in real and predicted DE."""

        filt_real = (
            data.real.filter_to_significant(fdr_threshold=self.fdr_threshold)
            .group_by(data.real.target_col)
            .len()
        )
        filt_pred = (
            data.pred.filter_to_significant(fdr_threshold=self.fdr_threshold)
            .group_by(data.pred.target_col)
            .len()
        )

        merged = filt_real.join(
            filt_pred,
            left_on=data.real.target_col,
            right_on=data.pred.target_col,
            suffix="_pred",
            how="left",
            coalesce=True,
        ).fill_null(0)

        # No significant genes in either real or predicted DE. Set to 1.0 since perfect
        # agreement but will fail spearman test
        if merged.shape[0] == 0:
            return 1.0

        return float(
            merged.select(
                pl.corr(
                    pl.col("len"),
                    pl.col("len_pred"),
                    method="spearman",
                ).alias("spearman_corr_nsig")
            )
            .to_numpy()
            .flatten()[0]
        )


class DEDirectionMatch:
    """Compute agreement in direction of DE gene changes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> dict[str, float]:
        """Compute directional agreement between real and predicted DE genes."""
        matches = {}

        merged = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold).join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )
        for row in (
            merged.with_columns(
                direction_match=pl.col(data.real.log2_fold_change_col).sign()
                == pl.col(f"{data.real.log2_fold_change_col}_pred").sign()
            )
            .group_by(
                data.real.target_col,
            )
            .agg(pl.mean("direction_match"))
            .iter_rows()
        ):
            matches.update({row[0]: row[1]})
        return matches


class DESpearmanLFC:
    """Compute Spearman correlation on log fold changes of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> dict[str, float]:
        """Compute correlation between log fold changes of significant genes."""
        correlations = {}

        merged = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold).join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )

        for row in (
            merged.group_by(
                data.real.target_col,
            )
            .agg(
                pl.corr(
                    pl.col(data.real.fold_change_col).cast(pl.Float64),
                    pl.col(f"{data.real.fold_change_col}_pred").cast(pl.Float64),
                    method="spearman",
                ).alias("spearman_corr"),
            )
            .iter_rows()
        ):
            correlations.update({row[0]: row[1]})

        return correlations


class DESigGenesRecall:
    """Compute recall of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> dict[str, float]:
        """Compute recall of significant genes between real and predicted DE."""

        filt_real = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold)
        filt_pred = data.pred.filter_to_significant(fdr_threshold=self.fdr_threshold)

        recall_frame = (
            filt_real.join(
                filt_pred,
                on=[data.real.target_col, data.real.feature_col],
                how="inner",
                coalesce=True,
            )
            .group_by(data.real.target_col)
            .len()
            .join(
                filt_real.group_by(data.real.target_col).len(),
                on=data.real.target_col,
                how="full",
                suffix="_expected",
                coalesce=True,
            )
            .fill_null(0)
            .with_columns(recall=pl.col("len") / pl.col("len_expected"))
            .select([data.real.target_col, "recall"])
        )

        return {row[0]: row[1] for row in recall_frame.iter_rows()}


class DENsigCounts:
    """Compute counts of significant genes."""

    def __init__(self, fdr_threshold: float = 0.05) -> None:
        self.fdr_threshold = fdr_threshold

    def __call__(self, data: DEComparison) -> dict[str, dict[str, int]]:
        """Compute counts of significant genes in real and predicted DE."""
        counts = {}

        for pert in data.iter_perturbations():
            real_sig = data.real.get_significant_genes(pert, self.fdr_threshold)
            pred_sig = data.pred.get_significant_genes(pert, self.fdr_threshold)

            counts[pert] = {
                "real": int(real_sig.size),
                "pred": int(pred_sig.size),
            }

        return counts


def compute_pr_auc(data: DEComparison) -> dict[str, float]:
    """Compute precision-recall AUC per perturbation for significant recovery."""
    return compute_generic_auc(data, method="pr")


def compute_roc_auc(data: DEComparison) -> dict[str, float]:
    """Compute ROC AUC per perturbation for significant recovery."""
    return compute_generic_auc(data, method="roc")


def compute_pr_curve(
    data: DEComparison,
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Compute precision-recall curve arrays per perturbation.

    Returns a dictionary keyed by perturbation, where each value contains:
      - precision: precision values from sklearn.metrics.precision_recall_curve
      - recall: recall values from sklearn.metrics.precision_recall_curve
      - thresholds: threshold values from sklearn.metrics.precision_recall_curve

    Perturbations that are not evaluable under the same constraints used by
    ``compute_pr_auc`` (no rows or only one class in labels) are returned with
    empty arrays for all three fields.
    """
    target_col = data.real.target_col
    feature_col = data.real.feature_col
    real_fdr_col = data.real.fdr_col
    pred_fdr_col = data.pred.fdr_col

    labeled_real = data.real.data.with_columns(
        (pl.col(real_fdr_col) < 0.05).cast(pl.Float32).alias("label")
    ).select([target_col, feature_col, "label"])

    merged = (
        data.pred.data.select([target_col, feature_col, pred_fdr_col])
        .join(
            labeled_real,
            on=[target_col, feature_col],
            how="inner",
            coalesce=True,
        )
        .drop_nulls(["label"])
        .with_columns((-pl.col(pred_fdr_col).replace(0, 1e-10).log10()).alias("nlp"))
        .drop_nulls(["nlp"])
    )

    empty = np.array([], dtype=np.float64)
    curves: dict[str, dict[str, NDArray[np.float64]]] = {}
    for pert in data.iter_perturbations():
        pert_data = merged.filter(pl.col(target_col) == pert)
        if pert_data.shape[0] == 0:
            curves[pert] = {
                "precision": empty.copy(),
                "recall": empty.copy(),
                "thresholds": empty.copy(),
            }
            continue

        labels = pert_data["label"].to_numpy()
        scores = pert_data["nlp"].to_numpy()
        if not (0 < labels.sum() < len(labels)):
            curves[pert] = {
                "precision": empty.copy(),
                "recall": empty.copy(),
                "thresholds": empty.copy(),
            }
            continue

        precision, recall, thresholds = precision_recall_curve(labels, scores)
        curves[pert] = {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }

    return curves


def compute_generic_auc(
    data: DEComparison,
    method: Literal["pr", "roc"] = "pr",
) -> dict[str, float]:
    """Compute AUC values for significant recovery per perturbation."""

    target_col = data.real.target_col
    feature_col = data.real.feature_col
    real_fdr_col = data.real.fdr_col
    pred_fdr_col = data.pred.fdr_col

    labeled_real = data.real.data.with_columns(
        (pl.col(real_fdr_col) < 0.05).cast(pl.Float32).alias("label")
    ).select([target_col, feature_col, "label"])

    merged = (
        data.pred.data.select([target_col, feature_col, pred_fdr_col])
        .join(
            labeled_real,
            on=[target_col, feature_col],
            how="inner",
            coalesce=True,
        )
        .drop_nulls(["label"])
        .with_columns((-pl.col(pred_fdr_col).replace(0, 1e-10).log10()).alias("nlp"))
        .drop_nulls(["nlp"])
    )

    results: dict[str, float] = {}
    for pert in data.iter_perturbations():
        pert_data = merged.filter(pl.col(target_col) == pert)
        if pert_data.shape[0] == 0:
            results[pert] = float("nan")
            continue

        labels = pert_data["label"].to_numpy()
        scores = pert_data["nlp"].to_numpy()

        if not (0 < labels.sum() < len(labels)):
            results[pert] = float("nan")
            continue

        match method:
            case "pr":
                results[pert] = float(average_precision_score(labels, scores))
            case "roc":
                fpr, tpr, _ = roc_curve(labels, scores)
                results[pert] = float(auc(fpr, tpr))
            case _:
                raise ValueError(f"Invalid AUC method: {method}")

    return results
