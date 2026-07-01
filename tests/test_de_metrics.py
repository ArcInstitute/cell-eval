import numpy as np
import polars as pl
import pytest

from cell_eval._pipeline import MetricPipeline
from cell_eval._types import DEComparison, DEResults
from cell_eval.metrics._de import DESpearmanLFC


def _directional_de_comparison() -> DEComparison:
    real_df = pl.DataFrame(
        {
            "target": ["pert1"] * 6,
            "feature": ["gene1", "gene2", "gene3", "gene4", "gene5", "gene6"],
            "log2_fold_change": [-2.0, -1.0, 1.0, 2.0, 0.2, -0.2],
            "p_value": [0.01, 0.01, 0.01, 0.01, 0.5, 0.5],
            "fdr": [0.01, 0.01, 0.01, 0.01, 0.5, 0.5],
        }
    )
    pred_df = pl.DataFrame(
        {
            "target": ["pert1"] * 6,
            "feature": ["gene1", "gene2", "gene3", "gene4", "gene5", "gene6"],
            "log2_fold_change": [-1.0, -2.0, 2.0, 1.0, 0.1, -0.1],
            "p_value": [0.01, 0.01, 0.01, 0.01, 0.5, 0.5],
            "fdr": [0.01, 0.01, 0.01, 0.01, 0.5, 0.5],
        }
    )
    return DEComparison(
        real=DEResults(real_df, name="real"),
        pred=DEResults(pred_df, name="pred"),
    )


def test_de_spearman_lfc_direction_filters() -> None:
    comparison = _directional_de_comparison()

    default = DESpearmanLFC(fdr_threshold=0.05)(comparison)
    all_direction = DESpearmanLFC(
        fdr_threshold=0.05, lfc_direction="all"
    )(comparison)
    positive = DESpearmanLFC(fdr_threshold=0.05, lfc_direction="pos")(comparison)
    negative = DESpearmanLFC(fdr_threshold=0.05, lfc_direction="neg")(comparison)

    assert np.isclose(default["pert1"], 0.6)
    assert np.isclose(all_direction["pert1"], default["pert1"])
    assert np.isclose(positive["pert1"], -1.0)
    assert np.isclose(negative["pert1"], -1.0)


def test_de_spearman_lfc_direction_filters_by_real_lfc() -> None:
    real_df = pl.DataFrame(
        {
            "target": ["pert1", "pert1"],
            "feature": ["gene1", "gene2"],
            "log2_fold_change": [1.0, 2.0],
            "p_value": [0.01, 0.01],
            "fdr": [0.01, 0.01],
        }
    )
    pred_df = real_df.with_columns(
        pl.Series("log2_fold_change", [-1.0, -2.0])
    )
    comparison = DEComparison(
        real=DEResults(real_df, name="real"),
        pred=DEResults(pred_df, name="pred"),
    )

    positive = DESpearmanLFC(fdr_threshold=0.05, lfc_direction="pos")(comparison)

    assert np.isclose(positive["pert1"], -1.0)


def test_de_spearman_lfc_invalid_direction() -> None:
    with pytest.raises(ValueError, match="Invalid LFC direction"):
        DESpearmanLFC(lfc_direction="up")  # type: ignore[arg-type]


def test_full_profile_includes_all_lfc_spearman_metrics_by_default() -> None:
    pipeline = MetricPipeline(profile="full", break_on_error=True)

    pipeline.compute_de_metrics(_directional_de_comparison())
    result_columns = set(pipeline.get_results().columns)

    assert {
        "de_spearman_lfc_sig",
        "de_spearman_pos_lfc_sig",
        "de_spearman_neg_lfc_sig",
    }.issubset(result_columns)
