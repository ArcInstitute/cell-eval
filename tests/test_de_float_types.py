import polars as pl

from cell_eval._types import DEComparison, DEResults
from cell_eval.metrics._de import DESpearmanLFC


def test_de_spearman_lfc_mixed_float_types() -> None:
    """Regression test: DESpearmanLFC handles mixed Float32/Float64 columns."""
    real_df = pl.DataFrame(
        {
            "target": ["pert1", "pert1", "pert2", "pert2"],
            "feature": ["gene1", "gene2", "gene1", "gene2"],
            "log2_fold_change": [1.5, 2.0, 0.5, 1.2],
            "p_value": [0.01, 0.02, 0.03, 0.04],
            "fdr": [0.01, 0.02, 0.03, 0.04],
        }
    )

    pred_df = real_df.with_columns(pl.col("log2_fold_change").cast(pl.Float64))

    comparison = DEComparison(
        real=DEResults(real_df, name="real"),
        pred=DEResults(pred_df, name="pred"),
    )

    result = DESpearmanLFC(fdr_threshold=0.05)(comparison)

    assert isinstance(result, dict)
    assert all(isinstance(value, (int, float)) for value in result.values())


def test_de_results_preserves_negative_log2_fold_change() -> None:
    """`abs_log2_fold_change` must be |log2_fold_change|; negatives must not be zeroed.

    Pre-fix, `__post_init__` applied `log(base=2).fill_nan(0.0)` to the values, which
    silently zeroed every downregulated gene.
    """
    df = pl.DataFrame(
        {
            "target": ["p", "p", "p", "p"],
            "feature": ["g1", "g2", "g3", "g4"],
            "log2_fold_change": [-1.0, 0.0, 1.0, 2.0],
            "p_value": [0.01, 0.01, 0.01, 0.01],
            "fdr": [0.01, 0.01, 0.01, 0.01],
        }
    )
    de = DEResults(df, name="real")
    lfc = de.data["log2_fold_change"].cast(pl.Float64).to_list()
    abs_lfc = de.data["abs_log2_fold_change"].cast(pl.Float64).to_list()
    assert lfc == [-1.0, 0.0, 1.0, 2.0]
    assert abs_lfc == [1.0, 0.0, 1.0, 2.0]
