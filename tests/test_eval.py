import os
import shutil
from typing import Literal

import numpy as np
import pytest

from cell_eval import MetricsEvaluator
from cell_eval.data import (
    CONTROL_VAR,
    PERT_COL,
    build_random_anndata,
    downsample_cells,
)

OUTDIR = "TEST_OUTPUT_DIRECTORY"
KNOWN_PROFILES: list[Literal["full", "vcc", "minimal", "de", "anndata"]] = [
    "full",
    "vcc",
    "minimal",
    "de",
    "anndata",
]


def test_broken_adata_mismatched_var_size():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()

    # Randomly subset genes on pred
    var_mask = np.random.random(adata_real.shape[1]) < 0.8
    adata_pred = adata_pred[:, var_mask]

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_mismatched_var_ordering():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()

    # Randomly subset genes on pred
    indices = np.arange(adata_real.shape[1])
    np.random.shuffle(indices)
    adata_pred = adata_pred[:, indices]

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_not_normlog():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_broken_adata_not_normlog_skip_check():
    adata_real = build_random_anndata(normlog=False)
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        allow_discrete=True,
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_broken_adata_invalid_pred_scale():
    """Test that predicted data with invalid scale is rejected."""
    adata_real = build_random_anndata(normlog=True)
    adata_pred = adata_real.copy()

    # Create invalid predicted data: mix of raw counts and log1p
    adata_pred.X = np.random.uniform(
        0,
        5000,
        size=adata_pred.X.shape,  # type: ignore
    )

    with pytest.raises(ValueError, match="Invalid scale.*exceeds log1p threshold"):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_missing_pertcol_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_real
    adata_real.obs.drop(columns=[PERT_COL], inplace=True)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_missing_pertcol_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_pred
    adata_pred.obs.drop(columns=[PERT_COL], inplace=True)

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_missing_control_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_real
    adata_real = adata_real[adata_real.obs[PERT_COL] != CONTROL_VAR].copy()

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_broken_adata_missing_control_in_pred():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_pred
    adata_pred = adata_pred[adata_pred.obs[PERT_COL] != CONTROL_VAR].copy()

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
        )


def test_unknown_alternative_de_metric():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove control_pert from adata_pred
    adata_pred = adata_pred[adata_pred.obs[PERT_COL] != CONTROL_VAR].copy()

    with pytest.raises(Exception):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert=CONTROL_VAR,
            pert_col=PERT_COL,
            outdir=OUTDIR,
            de_method="unknown",
        ).compute()


def test_eval_simple():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_eval_simple_profiles():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
    )
    for profile in KNOWN_PROFILES:
        evaluator.compute(
            profile=profile,
            break_on_error=True,
        )

    with pytest.raises(ValueError):
        evaluator.compute(
            profile="unknown",  # type: ignore
            break_on_error=True,
        )


def test_eval_missing_celltype_col():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)

    adata_real.obs.drop(columns="celltype", inplace=True)
    adata_pred.obs.drop(columns="celltype", inplace=True)

    assert "celltype" not in adata_real.obs.columns
    assert "celltype" not in adata_pred.obs.columns

    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_eval_pdex_kwargs():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
        pdex_kwargs={
            "exp_post_agg": True,
        },
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_eval_pdex_kwargs_duplicated():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert="control",
        pert_col="perturbation",
        pdex_kwargs={
            "exp_post_agg": True,
            "num_workers": 4,
        },
    )
    evaluator.compute(
        break_on_error=True,
    )


def validate_expected_files(
    outdir: str, prefix: str | None = None, remove: bool = True
):
    base_real_de = "real_de.csv" if not prefix else f"{prefix}_real_de.csv"
    base_pred_de = "pred_de.csv" if not prefix else f"{prefix}_pred_de.csv"
    base_results = "results.csv" if not prefix else f"{prefix}_results.csv"
    assert os.path.exists(f"{outdir}/{base_real_de}"), (
        "Expected file for real DE results missing"
    )
    assert os.path.exists(f"{outdir}/{base_pred_de}"), (
        "Expected file for predicted DE results missing"
    )
    assert os.path.exists(f"{outdir}/{base_results}"), (
        "Expected file for results missing"
    )
    if remove:
        shutil.rmtree(outdir)


def test_eval():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)


def test_eval_prefix():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        prefix="arbitrary",
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR, prefix="arbitrary")


def test_minimal_eval():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        profile="minimal",
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)


def test_eval_sparse():
    adata_real = build_random_anndata(as_sparse=True)
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)


def test_eval_downsampled_cells():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)


def test_eval_alt_metric():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        de_method="anderson",
    )
    evaluator.compute(
        break_on_error=True,
    )
    validate_expected_files(OUTDIR)
