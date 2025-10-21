import polars as pl
import pytest

from cell_eval._types import DESortBy, initialize_de_comparison
from cell_eval.metrics import de_overlap_metric, metrics_registry


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
