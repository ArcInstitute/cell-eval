import random

import numpy as np
import polars as pl
import pytest

from cell_eval._types import DESortBy, initialize_de_comparison
from cell_eval.metrics import (
    DESpearmanLFC,
    DESpearmanLFCBinned,
    de_overlap_metric,
    random_jaccard_metric,
    metrics_registry,
)
from cell_eval.metrics._de import _RandomGeneRecord, _choose_grouped_subset


def _make_de_frame(features: list[str]) -> pl.DataFrame:
    n = len(features)
    return pl.DataFrame(
        {
            "target": ["pert"] * n,
            "feature": features,
            "fold_change": [float(n - idx) for idx in range(n)],
            "p_value": [0.001] * n,
            "fdr": [0.001] * n,
        }
    )


def test_de_jaccard_overlap():
    real = _make_de_frame(["gene_a", "gene_b", "gene_c"])
    pred = _make_de_frame(["gene_a", "gene_d", "gene_e"])
    comparison = initialize_de_comparison(real, pred)

    scores = comparison.compute_overlap(
        k=None, metric="jaccard", sort_by=DESortBy.ABS_FOLD_CHANGE
    )
    assert pytest.approx(0.2) == scores["pert"]

    scores_topk = comparison.compute_overlap(
        k=2, metric="jaccard", sort_by=DESortBy.ABS_FOLD_CHANGE
    )
    assert pytest.approx(1 / 3) == scores_topk["pert"]

    metric_scores = de_overlap_metric(
        comparison, k=2, metric="jaccard", sort_by=DESortBy.ABS_FOLD_CHANGE
    )
    assert metric_scores == scores_topk


def test_jaccard_metric_registered():
    assert "jaccard_at_N" in metrics_registry.list_metrics()


def test_spearman_binned_handles_constant_bins():
    features = [f"gene_{idx}" for idx in range(5)]
    real_fold_change = [1.0, 2.0, 4.0, 8.0, 16.0]

    real = pl.DataFrame(
        {
            "target": ["pert"] * len(features),
            "feature": features,
            "fold_change": real_fold_change,
            "p_value": [0.001] * len(features),
            "fdr": [0.001] * len(features),
        }
    )

    pred = pl.DataFrame(
        {
            "target": ["pert"] * len(features),
            "feature": features,
            "fold_change": [1.0] * len(features),
            "p_value": [0.001] * len(features),
            "fdr": [0.001] * len(features),
        }
    )

    comparison = initialize_de_comparison(real, pred)

    metric = DESpearmanLFCBinned(fdr_threshold=0.05, n_bins=4)
    score = metric(comparison)["pert"]

    assert not np.isnan(score)
    assert score == 0.0


def test_spearman_lfc_handles_constant_values():
    features = [f"gene_{idx}" for idx in range(5)]
    real_fold_change = [1.0, 2.0, 4.0, 8.0, 16.0]

    real = pl.DataFrame(
        {
            "target": ["pert"] * len(features),
            "feature": features,
            "fold_change": real_fold_change,
            "p_value": [0.001] * len(features),
            "fdr": [0.001] * len(features),
        }
    )

    pred = pl.DataFrame(
        {
            "target": ["pert"] * len(features),
            "feature": features,
            "fold_change": [1.0] * len(features),
            "p_value": [0.001] * len(features),
            "fdr": [0.001] * len(features),
        }
    )

    comparison = initialize_de_comparison(real, pred)

    metric = DESpearmanLFC(fdr_threshold=0.05)
    score = metric(comparison)["pert"]

    assert not np.isnan(score)
    assert score == 0.0


def test_random_jaccard_matches_expected_sampling():
    features = [f"gene_{idx}" for idx in range(30)]
    real_fold_change = np.linspace(0.5, 3.0, len(features)).tolist()
    real_fdr = [0.01 if idx < 12 else 0.2 for idx in range(len(features))]

    real = pl.DataFrame(
        {
            "target": ["pert"] * len(features),
            "feature": features,
            "fold_change": real_fold_change,
            "p_value": [0.001] * len(features),
            "fdr": real_fdr,
        }
    )

    pred_fdr = [0.01 if idx in {1, 3, 5, 7, 9, 11} else 0.2 for idx in range(len(features))]
    pred = pl.DataFrame(
        {
            "target": ["pert"] * len(features),
            "feature": features,
            "fold_change": real_fold_change,
            "p_value": [0.001] * len(features),
            "fdr": pred_fdr,
        }
    )

    comparison = initialize_de_comparison(real, pred)

    scores = random_jaccard_metric(
        comparison,
        seed=123,
        min_genes=10,
        max_genes=20,
        fdr_threshold=0.05,
        max_de=15,
    )
    score = scores["pert"]

    target_col = comparison.real.target_col
    feature_col = comparison.real.feature_col
    fdr_col = comparison.real.fdr_col
    abs_col = comparison.real.abs_log2_fold_change_col

    real_subset = (
        comparison.real.data.filter(pl.col(target_col) == "pert")
        .sort(abs_col, descending=True)
        .select([feature_col, fdr_col])
    )
    records = [
        _RandomGeneRecord(gene=row[0], rank=float(idx), fdr=float(row[1]))
        for idx, row in enumerate(real_subset.iter_rows())
    ]

    rng = random.Random(123)
    sampled = _choose_grouped_subset(
        records=records,
        rng=rng,
        min_genes=10,
        max_genes=20,
        fdr_threshold=0.05,
        max_de=15,
    )

    assert sampled
    assert 10 <= len(sampled) <= 20
    real_de = {record.gene for record in sampled if record.fdr <= 0.05}
    assert 1 <= len(real_de) <= 15

    sampled_genes = [record.gene for record in sampled]
    pred_subset = (
        comparison.pred.data.filter(pl.col(target_col) == "pert")
        .filter(pl.col(feature_col).is_in(sampled_genes))
        .select([feature_col, comparison.pred.fdr_col])
    )
    pred_map = {row[0]: float(row[1]) for row in pred_subset.iter_rows()}
    pred_de = {
        gene for gene in sampled_genes if pred_map.get(gene, float("inf")) <= 0.05
    }

    union = real_de | pred_de
    expected = 0.0 if not union else len(real_de & pred_de) / len(union)

    assert pytest.approx(expected) == score
