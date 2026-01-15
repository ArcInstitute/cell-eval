import anndata as ad
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def guess_is_lognorm(
    adata: ad.AnnData,
    epsilon: float = 1e-3,
) -> bool:
    """Guess if the input is integer counts or log-normalized.

    This is an _educated guess_ based on whether the fractional component of cell sums.
    This _will not be able_ to distinguish between normalized input and log-normalized input.

    Returns:
        bool: True if the input is lognorm, False otherwise
    """
    if isinstance(adata.X, csr_matrix) or isinstance(adata.X, csc_matrix):
        frac, _ = np.modf(adata.X.data)
    elif adata.X is None:
        raise ValueError("adata.X is None")
    else:
        frac, _ = np.modf(adata.X)  # type: ignore

    return bool(np.any(frac > epsilon))


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
