from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cell_eval import MetricPipeline, MetricsEvaluator, metrics_registry
from cell_eval._types import PerturbationAnndataPair


@pytest.fixture
def pair(monkeypatch):
    # Restore the registry after each test, including on the buggy implementation.
    metric = metrics_registry.get_metric("discrimination_score_l1")
    assert metric.kwargs is not None
    monkeypatch.setattr(metric, "kwargs", metric.kwargs.copy())
    real = ad.AnnData(
        X=np.array([[0.0, 0.0, 0.0], [10.0, 1.0, 1.0], [1.0, 8.0, 2.0]]),
        obs=pd.DataFrame(
            {"perturbation": ["control", "A", "B"]}, index=["c", "a", "b"]
        ),
        var=pd.DataFrame(index=["A", "B", "C"]),
    )
    pred = real.copy()
    pred.X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 8.0, 2.0]])
    return PerturbationAnndataPair(
        real=real, pred=pred, pert_col="perturbation", control_pert="control"
    )


@pytest.mark.parametrize("configure_via", ["constructor", "add_metrics"])
def test_pipeline_configs_are_local(pair, configure_via):
    """Configuring another pipeline must not change an existing default one."""
    default = MetricPipeline(profile="pds", break_on_error=True)
    config = {"discrimination_score_l1": {"exclude_target_gene": False}}
    if configure_via == "constructor":
        configured = MetricPipeline(
            profile="pds", metric_configs=config, break_on_error=True
        )
    else:
        configured = MetricPipeline(profile=None, break_on_error=True)
        configured.add_metrics(["discrimination_score_l1"], configs=config)

    configured.compute_anndata_metrics(pair)
    default.compute_anndata_metrics(pair)

    assert configured.get_results()["discrimination_score_l1"].to_list() == [0.5, 1.0]
    assert default.get_results()["discrimination_score_l1"].to_list() == [1.0, 1.0]


def test_evaluator_does_not_retain_previous_call_config(pair, tmp_path):
    evaluator = MetricsEvaluator(
        adata_real=pair.real,
        adata_pred=pair.pred,
        pert_col=pair.pert_col,
        control_pert=pair.control_pert,
        allow_discrete=True,
        skip_de=True,
        outdir=str(tmp_path),
    )
    skip = [
        m for m in metrics_registry.list_metrics() if m != "discrimination_score_l1"
    ]
    kwargs: dict[str, Any] = {
        "profile": "anndata",
        "skip_metrics": skip,
        "write_csv": False,
        "break_on_error": True,
    }
    before, _ = evaluator.compute(**kwargs)
    configured, _ = evaluator.compute(
        metric_configs={"discrimination_score_l1": {"exclude_target_gene": False}},
        **kwargs,
    )
    after, _ = evaluator.compute(**kwargs)

    assert before["discrimination_score_l1"].to_list() == [1.0, 1.0]
    assert configured["discrimination_score_l1"].to_list() == [0.5, 1.0]
    assert after.equals(before)
