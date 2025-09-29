import logging
import re
from typing import Any, NamedTuple

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def guess_is_lognorm(
    adata: ad.AnnData,
    n_cells: int | float = 5e2,
    epsilon: float = 1e-2,
) -> bool:
    """Guess if the input is integer counts or log-normalized.

    This is an _educated guess_ based on whether the fractional component of cell sums.
    This _will not be able_ to distinguish between normalized input and log-normalized input.

    Returns:
        bool: True if the input is lognorm, False otherwise
    """
    # Determine the number of cells to use for the guess
    n_cells = int(min(adata.shape[0], n_cells))

    # Pick a random subset of cells
    cell_mask = np.random.choice(adata.shape[0], n_cells, replace=False)

    # Sum the counts for each cell
    cell_sums = adata.X[cell_mask].sum(axis=1)  # type: ignore (can be float but super unlikely)

    # Check if any cell sum's fractional part is greater than epsilon
    return bool(np.any(np.abs((cell_sums - cell_sums.round())) > epsilon))


class SplitSpec(NamedTuple):
    """Specification describing a paired real/pred AnnData subset."""

    label: str
    celltype: Any
    batch: Any | None
    real: ad.AnnData
    pred: ad.AnnData


def _format_prefix_value(value: Any) -> str:
    if isinstance(value, str):
        candidate = value
    elif pd.isna(value):
        candidate = "nan"
    else:
        candidate = str(value)
    candidate = candidate.strip()
    if not candidate:
        candidate = "empty"
    return re.sub(r"[\s/\\]+", "_", candidate)


def _subset_on_obs_value(
    adata: ad.AnnData,
    column: str,
    value: Any,
) -> ad.AnnData:
    series = adata.obs[column]
    if pd.isna(value):
        mask = series.isna()
    else:
        mask = series == value
    return adata[mask]


def split_anndata_on_celltype(
    adata: ad.AnnData,
    celltype_col: str,
) -> dict[Any, ad.AnnData]:
    """Split anndata on a celltype column, preserving raw labels."""
    if celltype_col not in adata.obs.columns:
        raise ValueError(
            f"Celltype column {celltype_col} not found in adata.obs: {adata.obs.columns}"
        )

    uniques = adata.obs[celltype_col].unique()
    return {ct: _subset_on_obs_value(adata, celltype_col, ct) for ct in uniques}


def split_anndata_on_column(
    adata: ad.AnnData,
    column: str,
) -> dict[Any, ad.AnnData]:
    """Split AnnData based on an arbitrary column."""
    if column not in adata.obs.columns:
        raise ValueError(
            f"Column {column} not found in adata.obs: {adata.obs.columns}"
        )
    uniques = adata.obs[column].unique()
    return {value: _subset_on_obs_value(adata, column, value) for value in uniques}


def build_celltype_split_specs(
    real: ad.AnnData,
    pred: ad.AnnData,
    celltype_col: str,
    batch_key: str | None = None,
) -> list[SplitSpec]:
    """Construct paired real/pred splits for downstream evaluation.

    Args:
        real: Ground-truth AnnData.
        pred: Predicted AnnData.
        celltype_col: Column used to split both AnnData objects.
        batch_key: Optional column used to further split predicted AnnData
            (and real AnnData when matching observations are present).

    Returns:
        list[SplitSpec]: List of paired subsets along with labels and metadata.
    """

    real_splits = split_anndata_on_celltype(real, celltype_col)
    pred_splits = split_anndata_on_celltype(pred, celltype_col)

    missing_in_real = [ct for ct in pred_splits.keys() if ct not in real_splits]
    missing_in_pred = [ct for ct in real_splits.keys() if ct not in pred_splits]
    if missing_in_real or missing_in_pred:
        msgs = []
        if missing_in_real:
            msgs.append(
                f"Missing cell types in real data: {sorted(map(str, missing_in_real))}"
            )
        if missing_in_pred:
            msgs.append(
                f"Missing cell types in predicted data: {sorted(map(str, missing_in_pred))}"
            )
        raise ValueError("; ".join(msgs))

    specs: list[SplitSpec] = []
    for ct_label, real_subset in real_splits.items():
        pred_subset = pred_splits[ct_label]
        celltype_label = str(ct_label)

        if batch_key is None:
            specs.append(
                SplitSpec(
                    label=celltype_label,
                    celltype=ct_label,
                    batch=None,
                    real=real_subset,
                    pred=pred_subset,
                )
            )
            continue

        if batch_key not in pred_subset.obs.columns:
            raise ValueError(
                f"Batch key '{batch_key}' missing from predicted AnnData for celltype '{celltype_label}'"
            )

        real_has_batch = batch_key in real_subset.obs.columns
        real_batch_splits = (
            split_anndata_on_column(real_subset, batch_key) if real_has_batch else None
        )

        for batch_value in pred_subset.obs[batch_key].unique():
            batch_pred_subset = _subset_on_obs_value(pred_subset, batch_key, batch_value)
            if batch_pred_subset.n_obs == 0:
                continue

            if real_batch_splits is not None:
                real_batch_subset = real_batch_splits.get(batch_value)
                if real_batch_subset is None or real_batch_subset.n_obs == 0:
                    logger.warning(
                        "Batch '%s' not found in real data for celltype '%s'; using full celltype subset",
                        batch_value,
                        celltype_label,
                    )
                    real_batch_subset = real_subset
            else:
                real_batch_subset = real_subset

            specs.append(
                SplitSpec(
                    label=f"{celltype_label}_{_format_prefix_value(batch_value)}",
                    celltype=ct_label,
                    batch=batch_value,
                    real=real_batch_subset,
                    pred=batch_pred_subset,
                )
            )

    return specs
