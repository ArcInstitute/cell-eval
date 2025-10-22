import numpy as np
import polars as pl
import pytest

from cell_eval._types import DESortBy, initialize_de_comparison
from cell_eval.metrics import DESpearmanLFCBinned, de_overlap_metric, metrics_registry


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
