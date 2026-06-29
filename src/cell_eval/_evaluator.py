import logging
import multiprocessing as mp
import os
import warnings
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from pdex import pdex

from cell_eval.utils import guess_is_lognorm

from ._pipeline import MetricPipeline
from ._types import PerturbationAnndataPair, initialize_de_comparison
from .utils import _cast_float16_to_float32

logger = logging.getLogger(__name__)


def _available_cpus() -> int:
    """Return CPUs the current process is allowed to use.

    Uses ``os.sched_getaffinity`` on Linux so SLURM/cgroup/taskset limits are
    respected; falls back to ``mp.cpu_count`` on macOS/Windows where that API
    is unavailable (those platforms typically run locally without cgroup caps).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


class MetricsEvaluator:
    """
    Evaluates benchmarking metrics of a predicted and real anndata object.

    Arguments
    =========

    adata_pred: ad.AnnData | str
        Predicted anndata object or path to anndata object.
    adata_real: ad.AnnData | str
        Real anndata object or path to anndata object.
    de_pred: pl.DataFrame | str | None = None
        Predicted differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    de_real: pl.DataFrame | str | None = None
        Real differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    control_pert: str = "non-targeting"
        Control perturbation name.
    pert_col: str = "target"
        Perturbation column name.
    num_threads: int = -1
        Number of threads for parallel differential expression.
    outdir: str = "./cell-eval-outdir"
        Output directory.
    allow_discrete: bool = False
        Allow discrete data.
    prefix: str | None = None
        Prefix for output files.
    pdex_kwargs: dict[str, Any] | None = None
        Keyword arguments for parallel_differential_expression.
        These will overwrite arguments passed to MetricsEvaluator.__init__ if they conflict.
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: pl.DataFrame | str | None = None,
        de_real: pl.DataFrame | str | None = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        num_threads: int = -1,
        outdir: str = "./cell-eval-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        pdex_kwargs: dict[str, Any] | None = None,
        skip_de: bool = False,
    ):
        # Enable a global string cache for categorical columns
        pl.enable_string_cache()

        if num_threads == -1:
            num_threads = _available_cpus()

        if os.path.exists(outdir):
            logger.warning(
                f"Output directory {outdir} already exists, potential overwrite occurring"
            )
        os.makedirs(outdir, exist_ok=True)

        # Stored so the data ceiling (compute_ceiling) can reuse the exact same
        # DE / pdex configuration as the main evaluation for comparability.
        self._num_threads = num_threads
        self._allow_discrete = allow_discrete
        self._skip_de = skip_de
        self._pdex_kwargs = pdex_kwargs or {}

        self.anndata_pair = _build_anndata_pair(
            real=adata_real,
            pred=adata_pred,
            control_pert=control_pert,
            pert_col=pert_col,
            allow_discrete=allow_discrete,
        )

        if skip_de:
            self.de_comparison = None
        else:
            self.de_comparison = _build_de_comparison(
                anndata_pair=self.anndata_pair,
                de_pred=de_pred,
                de_real=de_real,
                num_threads=num_threads,
                allow_discrete=allow_discrete,
                outdir=outdir,
                prefix=prefix,
                pdex_kwargs=self._pdex_kwargs,
            )

        self.outdir = outdir
        self.prefix = prefix

    def compute(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            pipeline.skip_metrics(skip_metrics)
        pipeline.compute_de_metrics(self.de_comparison)
        pipeline.compute_anndata_metrics(self.anndata_pair)
        results = pipeline.get_results()
        agg_results = pipeline.get_agg_results()

        if write_csv:
            self._write_results(results, agg_results, basename)

        return results, agg_results

    def compute_ceiling(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata", "pds"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "ceiling_results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
        seed: int = 0,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Estimate a data ceiling: the maximum achievable score per metric.

        Uses the real data only. For each perturbation (and the control) the
        cells are bootstrapped to twice their count and split into two equal
        halves; one half is treated as "real" and the other as "prediction".
        Running the normal metric pipeline on that self-split yields, per metric,
        an upper bound on how well any model could score given the noise inherent
        in the real data.

        Outputs mirror :meth:`compute` (``ceiling_results.csv`` /
        ``agg_ceiling_results.csv``). The bootstrap DE is computed in-memory and
        not written to disk. The same ``pdex_kwargs`` and ``allow_discrete`` used
        for the main evaluation are reused so the ceiling is directly comparable.
        """
        logger.info(f"Computing data ceiling (seed={seed})")
        half_real, half_pred = self._bootstrap_halves(seed)

        ceiling_pair = PerturbationAnndataPair(
            real=half_real,
            pred=half_pred,
            control_pert=self.anndata_pair.control_pert,
            pert_col=self.anndata_pair.pert_col,
            embed_key=self.anndata_pair.embed_key,
        )

        if self._skip_de:
            ceiling_de = None
        else:
            ceiling_de = _build_de_comparison(
                anndata_pair=ceiling_pair,
                num_threads=self._num_threads,
                allow_discrete=self._allow_discrete,
                outdir=None,  # keep the bootstrap DE in-memory; never persisted
                prefix=None,
                pdex_kwargs=dict(self._pdex_kwargs),
            )

        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            pipeline.skip_metrics(skip_metrics)
        pipeline.compute_de_metrics(ceiling_de)
        pipeline.compute_anndata_metrics(ceiling_pair)
        results = pipeline.get_results()
        agg_results = pipeline.get_agg_results()

        if write_csv:
            self._write_results(results, agg_results, basename)

        return results, agg_results

    def _bootstrap_halves(self, seed: int) -> tuple[ad.AnnData, ad.AnnData]:
        """Build two same-size bootstrap halves of the real data.

        Resampling is stratified per perturbation (including the control): each
        group's ``n`` cells are drawn ``2n`` times with replacement and split
        into two halves of ``n`` cells. This guarantees both halves carry every
        perturbation plus the control with the same per-perturbation membership
        as the real data, so the resulting ``PerturbationAnndataPair`` validates
        and the bootstrap DE keeps the same statistical power.
        """
        real = self.anndata_pair.real
        pert_col = self.anndata_pair.pert_col

        rng = np.random.default_rng(seed)

        # Group row positions per perturbation in a single pass (`observed=True`
        # keeps only perturbations actually present). `.indices` yields positional
        # indices, so they can index the AnnData directly.
        half_real_idx: list[np.ndarray] = []
        half_pred_idx: list[np.ndarray] = []
        for _pert, idx in real.obs.groupby(pert_col, observed=True).indices.items():
            draws = rng.choice(idx, size=2 * idx.size, replace=True)
            half_real_idx.append(draws[: idx.size])
            half_pred_idx.append(draws[idx.size :])

        # Sampling with replacement duplicates obs names; anndata warns about the
        # non-unique index on slice, so silence that one known-benign warning and
        # make the names unique immediately afterwards.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Observation names are not unique"
            )
            half_real = real[np.concatenate(half_real_idx)].copy()
            half_pred = real[np.concatenate(half_pred_idx)].copy()
        half_real.obs_names_make_unique()
        half_pred.obs_names_make_unique()

        return half_real, half_pred

    def _write_results(
        self,
        results: pl.DataFrame,
        agg_results: pl.DataFrame,
        basename: str,
    ) -> None:
        # some prefixes/basenames (e.g. HepG2/C3A) may have slashes in them
        prefix = self.prefix.replace("/", "-") if self.prefix is not None else None
        basename = basename.replace("/", "-")

        outpath = os.path.join(
            self.outdir,
            f"{prefix}_{basename}" if prefix else basename,
        )
        agg_outpath = os.path.join(
            self.outdir,
            f"{prefix}_agg_{basename}" if prefix else f"agg_{basename}",
        )

        logger.info(f"Writing perturbation level metrics to {outpath}")
        results.write_csv(outpath)

        logger.info(f"Writing aggregate metrics to {agg_outpath}")
        agg_results.write_csv(agg_outpath)


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
):
    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    # Cast float16 to float32 since NUMBA (used by pdex) does not support float16
    _cast_float16_to_float32(real, which="real")
    _cast_float16_to_float32(pred, which="pred")

    # Validate that the input is normalized and log-transformed
    _convert_to_normlog(real, which="real", allow_discrete=allow_discrete)
    _convert_to_normlog(pred, which="pred", allow_discrete=allow_discrete)

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _convert_to_normlog(
    adata: ad.AnnData,
    which: str | None = None,
    allow_discrete: bool = False,
):
    """Performs a norm-log conversion if the input is integer data (inplace).

    Will skip if the input is not integer data.
    """
    if guess_is_lognorm(adata=adata, validate=not allow_discrete):
        logger.info(
            "Input is found to be log-normalized already - skipping transformation."
        )
        return  # Input is already log-normalized

    # User specified that they want to allow discrete data
    if allow_discrete:
        if which:
            logger.info(
                f"Discovered integer data for {which}. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        else:
            logger.info(
                "Discovered integer data. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        return  # proceed without conversion

    # Convert the data to norm-log
    if which:
        logger.info(f"Discovered integer data for {which}. Converting to norm-log.")
    sc.pp.normalize_total(adata=adata, inplace=True)  # normalize to median
    sc.pp.log1p(adata)  # log-transform (log1p)


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    num_threads: int = 1,
    allow_discrete: bool = False,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] | None = None,
):
    return initialize_de_comparison(
        real=_load_or_build_de(
            mode="real",
            de_path=de_real,
            anndata_pair=anndata_pair,
            num_threads=num_threads,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        ),
        pred=_load_or_build_de(
            mode="pred",
            de_path=de_pred,
            anndata_pair=anndata_pair,
            num_threads=num_threads,
            allow_discrete=allow_discrete,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        ),
    )


def _build_pdex_kwargs(
    reference: str,
    groupby: str,
    threads: int,
    allow_discrete: bool,
    pdex_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdex_kwargs = pdex_kwargs or {}
    if "reference" not in pdex_kwargs:
        pdex_kwargs["reference"] = reference
    if "groupby" not in pdex_kwargs:
        pdex_kwargs["groupby"] = groupby
    if "threads" not in pdex_kwargs:
        pdex_kwargs["threads"] = threads
    if "is_log1p" not in pdex_kwargs:
        if allow_discrete:
            pdex_kwargs["is_log1p"] = False
        else:
            pdex_kwargs["is_log1p"] = True
    # Keep cell-eval's default DE behavior unchanged from pdex<0.2.5: pin epsilon=0
    # (pdex>=0.2.5 defaults it to 1e-9) and leave the pooled-CPM floor filter OFF.
    # Both are opt-in — enable the filter via --cpm-filter / pdex_kwargs["cpm_filter"].
    if "epsilon" not in pdex_kwargs:
        pdex_kwargs["epsilon"] = 0.0
    return pdex_kwargs


def _load_or_build_de(
    mode: Literal["pred", "real"],
    de_path: pl.DataFrame | str | None = None,
    anndata_pair: PerturbationAnndataPair | None = None,
    num_threads: int = 1,
    outdir: str | None = None,
    prefix: str | None = None,
    allow_discrete: bool = False,
    pdex_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    if de_path is None:
        if anndata_pair is None:
            raise ValueError("anndata_pair must be provided if de_path is not provided")
        logger.info(f"Computing DE for {mode} data")
        pdex_kwargs = _build_pdex_kwargs(
            reference=anndata_pair.control_pert,
            groupby=anndata_pair.pert_col,
            threads=num_threads,
            allow_discrete=allow_discrete,
            pdex_kwargs=pdex_kwargs or {},
        )
        logger.info(f"Using the following pdex kwargs: {pdex_kwargs}")
        frame = pdex(
            adata=anndata_pair.real if mode == "real" else anndata_pair.pred,
            mode="ref",
            **pdex_kwargs,
        )
        if outdir is not None:
            if prefix is not None:
                prefix = prefix.replace(
                    "/", "-"
                )  # some prefixes (e.g. HepG2/C3A) may have slashes in them
            pathname = f"{mode}_de.csv" if not prefix else f"{prefix}_{mode}_de.csv"
            logger.info(f"Writing {mode} DE results to: {pathname}")
            frame.write_csv(os.path.join(outdir, pathname))

        return frame  # type: ignore
    elif isinstance(de_path, str):
        logger.info(f"Reading {mode} DE results from {de_path}")
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.read_csv(
            de_path,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )
    elif isinstance(de_path, pl.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return de_path
    elif isinstance(de_path, pd.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.from_pandas(de_path)
    else:
        raise TypeError(f"Unexpected type for de_path: {type(de_path)}")
