import logging

import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

logger = logging.getLogger(__name__)


def guess_is_lognorm(
    adata: ad.AnnData,
    epsilon: float = 1e-3,
    max_threshold: float = 11.0,
    validate: bool = False,
) -> bool:
    """Guess if the input is integer counts or log-normalized.

    This is an _educated guess_ based on whether there is a fractional component of values.
    When validate=True, checks that data with decimal values is in expected log1p range.

    Args:
        adata: AnnData object to check
        epsilon: Threshold for detecting fractional values (default 1e-3)
        max_threshold: Maximum valid value for log1p normalized data (default 11.0)
        validate: If True, validate that decimal data is in valid log1p range (default False)

    Returns:
        bool: True if the input is lognorm, False if integer counts

    Raises:
        ValueError: If validate=True and data has decimal values but falls outside
            valid log1p range (min < 0 or max >= max_threshold), indicating mixed or invalid scales
    """
    if adata.X is None:
        raise ValueError("adata.X is None")

    # Check for fractional values
    if isinstance(adata.X, csr_matrix) or isinstance(adata.X, csc_matrix):
        frac, _ = np.modf(adata.X.data)
    else:
        frac, _ = np.modf(adata.X)  # type: ignore

    has_decimals = bool(np.any(frac > epsilon))

    if not has_decimals:
        # All integer values - assume raw counts
        if validate:
            logger.info("Data appears to be integer counts (no decimal values detected)")
        return False

    # Data has decimals - perform validation if requested
    if validate:
        # Validate it's in valid log1p range
        if isinstance(adata.X, csr_matrix) or isinstance(adata.X, csc_matrix):
            max_val = adata.X.max()
            min_val = adata.X.min()
        else:
            max_val = float(np.max(adata.X))
            min_val = float(np.min(adata.X))

        # Validate range
        if min_val < 0:
            raise ValueError(
                f"Invalid scale: min value {min_val:.2f} is negative. "
                f"Log1p normalized data must have all values >= 0."
            )

        if max_val >= max_threshold:
            raise ValueError(
                f"Invalid scale: max value {max_val:.2f} exceeds log1p threshold of {max_threshold}. "
                f"Expected log1p normalized values in range [0, {max_threshold}), but found values suggesting "
                f"raw counts or incorrect normalization. Values above {max_threshold} indicate mixed scales "
                f"(some cells with raw counts, some with log1p values)."
            )

        # Valid log1p data
        logger.info(
            f"Data appears to be log1p normalized (decimals detected, range [{min_val:.2f}, {max_val:.2f}])"
        )

    return True


def split_anndata_on_celltype(
    adata: ad.AnnData,
    celltype_col: str,
) -> dict[str, ad.AnnData]:
    """Split anndata on celltype column.

    Args:
        adata: AnnData object
        celltype_col: Column name in adata.obs that contains the celltype labels

    Returns:
        dict[str, AnnData]: Dictionary of AnnData objects, keyed by celltype
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(
            f"Celltype column {celltype_col} not found in adata.obs: {adata.obs.columns}"
        )

    return {
        ct: adata[adata.obs[celltype_col] == ct]
        for ct in adata.obs[celltype_col].unique()
    }
