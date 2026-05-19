import os
import shutil
from typing import Literal, cast

import numpy as np
import pandas as pd
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
        is_log1p=False,
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
        is_log1p=False,
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_eval_trusts_declared_log1p_without_scale_scan():
    """MetricsEvaluator should not scan or transform .X when is_log1p is set."""
    adata_real = build_random_anndata(normlog=True)
    adata_pred = adata_real.copy()

    # Values above the old log1p threshold used to fail during scale guessing.
    adata_pred.X = np.random.uniform(
        0,
        5000,
        size=adata_pred.X.shape,  # type: ignore
    )
    before = np.asarray(adata_pred.X).copy()

    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        skip_de=True,
        is_log1p=True,
    )

    np.testing.assert_array_equal(evaluator.anndata_pair.pred.X, before)


def test_broken_adata_missing_pertcol_in_real():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()

    # Remove pert_col from adata_real
    cast(pd.DataFrame, adata_real.obs).drop(columns=[PERT_COL], inplace=True)

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
    cast(pd.DataFrame, adata_pred.obs).drop(columns=[PERT_COL], inplace=True)

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
            de_method="unknown",  # ty: ignore[unknown-argument]
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

    cast(pd.DataFrame, adata_real.obs).drop(columns="celltype", inplace=True)
    cast(pd.DataFrame, adata_pred.obs).drop(columns="celltype", inplace=True)

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
            "geometric_mean": False,
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
            "geometric_mean": False,
            "threads": 4,
        },
    )
    evaluator.compute(
        break_on_error=True,
    )


def test_eval_pdex_kwargs_is_log1p_conflict():
    adata_real = build_random_anndata()
    adata_pred = downsample_cells(adata_real, fraction=0.5)
    with pytest.raises(ValueError, match="Conflicting log1p configuration"):
        MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            control_pert="control",
            pert_col="perturbation",
            is_log1p=True,
            pdex_kwargs={
                "is_log1p": False,
            },
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


def _assert_results_close(a, b) -> None:
    """Assert two per-perturbation result frames are numerically identical."""
    a = a.sort("perturbation")
    b = b.sort("perturbation")
    assert sorted(a.columns) == sorted(b.columns)
    assert a["perturbation"].to_list() == b["perturbation"].to_list()
    cols = [c for c in a.columns if c != "perturbation"]
    num_a = a.select(cols).to_numpy().astype(float)
    num_b = b.select(cols).to_numpy().astype(float)
    assert np.allclose(num_a, num_b, equal_nan=True)


def test_ceiling_bootstrap_halves_membership():
    """Each half must carry every perturbation + control with the real counts."""
    adata_real = build_random_anndata()
    evaluator = MetricsEvaluator(
        adata_pred=adata_real.copy(),
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        skip_de=True,  # membership only depends on the bootstrap, skip pdex
    )

    half_real, half_pred = evaluator._bootstrap_halves(seed=0)

    real_counts = evaluator.anndata_pair.real.obs[PERT_COL].value_counts().to_dict()
    assert CONTROL_VAR in real_counts
    for half in (half_real, half_pred):
        half_counts = half.obs[PERT_COL].value_counts().to_dict()
        # same set of perturbations (incl. control) and same per-pert membership
        assert half_counts == real_counts
        # sampling with replacement must yield unique obs names
        assert half.obs_names.is_unique

    shutil.rmtree(OUTDIR)


def test_eval_ceiling():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    results, agg_results = evaluator.compute_ceiling(break_on_error=True)
    assert results.height > 0
    assert agg_results.height > 0
    assert os.path.exists(f"{OUTDIR}/ceiling_results.csv")
    assert os.path.exists(f"{OUTDIR}/agg_ceiling_results.csv")
    shutil.rmtree(OUTDIR)


def test_eval_ceiling_prefix():
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
    evaluator.compute_ceiling(break_on_error=True)
    assert os.path.exists(f"{OUTDIR}/arbitrary_ceiling_results.csv")
    assert os.path.exists(f"{OUTDIR}/arbitrary_agg_ceiling_results.csv")
    shutil.rmtree(OUTDIR)


def test_eval_ceiling_profiles():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    for profile in KNOWN_PROFILES:
        evaluator.compute_ceiling(
            profile=profile,
            break_on_error=True,
            write_csv=False,
        )
    shutil.rmtree(OUTDIR)


def test_eval_ceiling_pds_skips_de():
    """When the evaluator skips DE, the ceiling must too (no pdex, no DE files)."""
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        skip_de=True,
    )
    assert evaluator.de_comparison is None
    results, _ = evaluator.compute_ceiling(
        profile="pds",
        break_on_error=True,
        write_csv=False,
    )
    assert results.height > 0
    shutil.rmtree(OUTDIR)


def test_eval_ceiling_reproducible():
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
        num_threads=1,  # deterministic reductions
    )
    r1, _ = evaluator.compute_ceiling(seed=7, write_csv=False, break_on_error=True)
    r2, _ = evaluator.compute_ceiling(seed=7, write_csv=False, break_on_error=True)
    _assert_results_close(r1, r2)
    shutil.rmtree(OUTDIR)


def test_eval_ceiling_does_not_clobber_de():
    """The in-memory ceiling DE must not overwrite the main run's DE artifacts."""
    adata_real = build_random_anndata()
    adata_pred = adata_real.copy()
    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
        outdir=OUTDIR,
    )
    evaluator.compute(break_on_error=True)

    real_de_path = f"{OUTDIR}/real_de.csv"
    pred_de_path = f"{OUTDIR}/pred_de.csv"
    with open(real_de_path, "rb") as fh:
        before_real = fh.read()
    with open(pred_de_path, "rb") as fh:
        before_pred = fh.read()

    evaluator.compute_ceiling(seed=0, break_on_error=True)

    with open(real_de_path, "rb") as fh:
        assert fh.read() == before_real
    with open(pred_de_path, "rb") as fh:
        assert fh.read() == before_pred

    # ceiling outputs exist, but no ceiling DE artifacts are written
    assert os.path.exists(f"{OUTDIR}/ceiling_results.csv")
    assert os.path.exists(f"{OUTDIR}/agg_ceiling_results.csv")
    assert not os.path.exists(f"{OUTDIR}/ceiling_real_de.csv")
    assert not os.path.exists(f"{OUTDIR}/ceiling_pred_de.csv")
    shutil.rmtree(OUTDIR)
