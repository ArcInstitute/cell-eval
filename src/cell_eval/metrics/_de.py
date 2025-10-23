"""DE metrics module."""

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .._types import DEComparison, DESortBy


@dataclass(frozen=True)
class _RandomGeneRecord:
    gene: str
    rank: float
    fdr: float


def _spearman_from_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman correlation for the provided vectors or NaN if undefined."""
    if x.size < 2 or y.size < 2:
        return float("nan")
    corr_df = pl.DataFrame({"x": x, "y": y}).select(
        pl.corr(pl.col("x"), pl.col("y"), method="spearman").alias("spearman_corr")
    )
    value = corr_df.row(0)[0]
    return float(value) if value is not None else float("nan")


def _choose_grouped_subset(
    records: list[_RandomGeneRecord],
    rng: random.Random,
    min_genes: int,
    max_genes: int,
    fdr_threshold: float,
    max_de: int,
) -> list[_RandomGeneRecord]:
    if len(records) < min_genes:
        return []

    de_pool = [record for record in records if record.fdr <= fdr_threshold]
    non_de_pool = [record for record in records if record.fdr > fdr_threshold]

    if not de_pool:
        return []

    max_total = min(len(records), max_genes)
    feasible_counts: list[int] = []
    for count in range(min_genes, max_total + 1):
        max_de_possible = min(max_de, count, len(de_pool))
        min_de_required = max(1, count - len(non_de_pool))
        if min_de_required <= max_de_possible:
            feasible_counts.append(count)

    if not feasible_counts:
        return []

    selected_count = rng.choice(feasible_counts)
    max_de_possible = min(max_de, selected_count, len(de_pool))
    min_de_required = max(1, selected_count - len(non_de_pool))
    if min_de_required > max_de_possible:
        return []

    if max_de_possible == min_de_required:
        de_count = min_de_required
    else:
        de_count = rng.randint(min_de_required, max_de_possible)
    non_de_count = selected_count - de_count

    selected: list[_RandomGeneRecord] = []

    if len(de_pool) <= de_count:
        selected_de = list(de_pool)
    else:
        selected_de = rng.sample(de_pool, de_count)
    selected.extend(selected_de)

    if non_de_count > 0:
        if len(non_de_pool) <= non_de_count:
            selected_non = list(non_de_pool)[:non_de_count]
        else:
            selected_non = rng.sample(non_de_pool, non_de_count)
        selected.extend(selected_non)

    if len(selected) < selected_count:
        excluded = {(record.gene, record.rank) for record in selected}
        remaining = [
            record for record in records if (record.gene, record.rank) not in excluded
        ]
        remaining.sort(key=lambda record: (record.rank, record.gene))
        needed = selected_count - len(selected)
        selected.extend(remaining[:needed])

    selected.sort(key=lambda record: (record.rank, record.gene))
    return selected


def de_overlap_metric(
    data: DEComparison,
    k: int | None,
    metric: Literal["precision", "overlap", "jaccard"] = "overlap",
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


def random_jaccard_metric(
    data: DEComparison,
    seed: int = 42,
    min_genes: int = 10,
    max_genes: int = 20,
    fdr_threshold: float = 0.05,
    max_de: int = 15,
    num_samples: int = 1,
) -> dict[str, float]:
    """Compute Jaccard overlap on random subsets of genes per perturbation.

    The metric samples `num_samples` subsets per perturbation (each between
    `min_genes` and `max_genes` genes with between 1 and `max_de` DE genes) and
    returns the average Jaccard score across those samples.
    """
    if min_genes < 1:
        raise ValueError("min_genes must be at least 1")
    if max_genes < min_genes:
        raise ValueError("max_genes must be greater than or equal to min_genes")
    if max_de < 1:
        raise ValueError("max_de must be at least 1")
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    rng = random.Random(seed)
    results: dict[str, float] = {}
    target_col = data.real.target_col
    feature_col = data.real.feature_col
    fdr_col = data.real.fdr_col
    sort_col = data.real.abs_log2_fold_change_col

    for perturbation in data.iter_perturbations():
        real_subset = (
            data.real.data.filter(pl.col(target_col) == perturbation)
            .sort(sort_col, descending=True)
            .select([feature_col, fdr_col])
        )

        if real_subset.height < min_genes:
            results[perturbation] = 0.0
            continue

        real_genes = real_subset.get_column(feature_col).to_list()
        real_fdrs = real_subset.get_column(fdr_col).to_list()
        records = [
            _RandomGeneRecord(gene=gene, rank=float(idx), fdr=float(fdr))
            for idx, (gene, fdr) in enumerate(zip(real_genes, real_fdrs))
        ]

        pred_subset = (
            data.pred.data.filter(pl.col(target_col) == perturbation)
            .filter(pl.col(feature_col).is_in(real_genes))
            .select([feature_col, data.pred.fdr_col])
        )
        pred_fdr_map = {row[0]: float(row[1]) for row in pred_subset.iter_rows()}

        total_score = 0.0
        for _ in range(num_samples):
            sampled_records = _choose_grouped_subset(
                records=records,
                rng=rng,
                min_genes=min_genes,
                max_genes=max_genes,
                fdr_threshold=fdr_threshold,
                max_de=max_de,
            )

            if not sampled_records:
                score = 0.0
            else:
                sampled_genes = [record.gene for record in sampled_records]
                real_de_genes = {
                    record.gene
                    for record in sampled_records
                    if record.fdr <= fdr_threshold
                }
                pred_de_genes = {
                    gene
                    for gene in sampled_genes
                    if pred_fdr_map.get(gene, float("inf")) <= fdr_threshold
                }
                union = real_de_genes | pred_de_genes
                score = (
                    0.0 if not union else len(real_de_genes & pred_de_genes) / len(union)
                )

            total_score += score

        results[perturbation] = float(total_score / num_samples)

    return results


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

        merged = data.real.filter_to_significant(fdr_threshold=0.05).join(
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
        correlations: dict[str, float] = {}

        merged = data.real.filter_to_significant(fdr_threshold=self.fdr_threshold).join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )

        if merged.height == 0:
            return correlations

        real_fc_col = data.real.fold_change_col
        pred_fc_col = f"{real_fc_col}_pred"
        real_log_fc_col = data.real.log2_fold_change_col
        pred_log_fc_col = f"{real_log_fc_col}_pred"

        for pert in data.iter_perturbations():
            pert_frame = merged.filter(pl.col(data.real.target_col) == pert)
            if pert_frame.height == 0:
                continue

            real_vals = pert_frame[real_fc_col].to_numpy().astype("float64")
            pred_vals = pert_frame[pred_fc_col].to_numpy().astype("float64")
            correlation = _spearman_from_numpy(real_vals, pred_vals)

            if np.isnan(correlation):
                real_log_vals = pert_frame[real_log_fc_col].to_numpy().astype("float64")
                pred_log_vals = pert_frame[pred_log_fc_col].to_numpy().astype("float64")
                correlation = _spearman_from_numpy(real_log_vals, pred_log_vals)

            if np.isnan(correlation):
                correlation = 0.0

            correlations[pert] = correlation

        return correlations


class DESpearmanLFCBinned:
    """Compute Spearman correlation on binned log fold changes across perturbations."""

    def __init__(self, fdr_threshold: float = 0.05, n_bins: int = 4) -> None:
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        self.fdr_threshold = fdr_threshold
        self.n_bins = n_bins

    def __call__(self, data: DEComparison) -> dict[str, float]:
        """Compute Spearman correlation using shared log fold change bins."""
        real_significant = data.real.filter_to_significant(
            fdr_threshold=self.fdr_threshold
        )
        if real_significant.height == 0:
            return {}

        merged = real_significant.join(
            data.pred.data,
            on=[data.real.target_col, data.real.feature_col],
            suffix="_pred",
            how="inner",
        )

        if merged.height == 0:
            return {}

        log_fc_col = data.real.log2_fold_change_col
        log_fc_pred_col = f"{log_fc_col}_pred"

        all_log_fold_changes = real_significant.select(log_fc_col).to_numpy().astype(
            "float64"
        )
        flat_log_fold_changes = all_log_fold_changes.flatten()

        if flat_log_fold_changes.size == 0:
            return {}

        quantile_edges = np.quantile(
            flat_log_fold_changes,
            np.linspace(0.0, 1.0, self.n_bins + 1),
        )

        eps = np.finfo(np.float64).eps
        for idx in range(1, quantile_edges.size):
            if quantile_edges[idx] <= quantile_edges[idx - 1]:
                quantile_edges[idx] = quantile_edges[idx - 1] + eps

        quantile_edges[0] = -np.inf
        quantile_edges[-1] = np.inf
        interior_edges = quantile_edges[1:-1]

        merged = merged.with_columns(
            pl.col(log_fc_col)
            .map_elements(
                lambda value: int(np.digitize(value, interior_edges, right=False)),
                return_dtype=pl.Int32,
            )
            .alias("real_bin"),
            pl.col(log_fc_pred_col)
            .map_elements(
                lambda value: int(np.digitize(value, interior_edges, right=False)),
                return_dtype=pl.Int32,
            )
            .alias("pred_bin"),
        )

        results: dict[str, float] = {}
        for perturbation in data.iter_perturbations():
            pert_frame = merged.filter(pl.col(data.real.target_col) == perturbation)
            if pert_frame.height == 0:
                continue

            real_bins = pert_frame["real_bin"].to_numpy().astype("float64")
            pred_bins = pert_frame["pred_bin"].to_numpy().astype("float64")
            correlation = _spearman_from_numpy(real_bins, pred_bins)

            if np.isnan(correlation):
                real_vals = pert_frame[log_fc_col].to_numpy().astype("float64")
                pred_vals = pert_frame[log_fc_pred_col].to_numpy().astype("float64")
                correlation = _spearman_from_numpy(real_vals, pred_vals)

            if np.isnan(correlation):
                correlation = 0.0

            results[perturbation] = correlation

        return results


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
                precision, recall, _ = precision_recall_curve(labels, scores)
                results[pert] = float(auc(recall, precision))
            case "roc":
                fpr, tpr, _ = roc_curve(labels, scores)
                results[pert] = float(auc(fpr, tpr))
            case _:
                raise ValueError(f"Invalid AUC method: {method}")

    return results
