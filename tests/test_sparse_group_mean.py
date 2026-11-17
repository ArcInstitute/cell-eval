from typing import cast

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from cell_eval._types import PerturbationAnndataPair
from cell_eval.metrics._anndata import ClusteringAgreement


def _random_adata(n_cells: int, dtype, sparse_type=None) -> ad.AnnData:
    rng = np.random.default_rng(0)
    matrix = rng.integers(0, 20, size=(n_cells, 13)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        matrix = matrix * rng.random(matrix.shape).astype(dtype)
    matrix[rng.random(matrix.shape) < 0.7] = 0
    labels = rng.choice(["control", "a", "z", "α"], size=n_cells)
    obs = pd.DataFrame(
        {"perturbation": pd.Categorical(labels)},
        index=np.arange(n_cells).astype(str),
    )
    return ad.AnnData(
        X=sparse_type(matrix) if sparse_type else matrix,
        obs=obs,
    )


# 5000 cells spans more than one 4096-row block
@pytest.mark.parametrize("n_cells", [19, 5000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("sparse_type", [csr_matrix, csc_matrix])
def test_sparse_bulk_matches_dense(n_cells, dtype, sparse_type):
    dense = _random_adata(n_cells, dtype)
    sparse = _random_adata(n_cells, dtype, sparse_type)
    keys, values = PerturbationAnndataPair._bulk_anndata(sparse, "perturbation")
    exp_keys, exp_values = PerturbationAnndataPair._bulk_anndata(dense, "perturbation")
    np.testing.assert_array_equal(keys, exp_keys)
    assert values.dtype == exp_values.dtype
    rtol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(values, exp_values, rtol=rtol)


@pytest.mark.parametrize("sparse_type", [csr_matrix, csc_matrix])
def test_sparse_centroids_match_dense(sparse_type):
    dense = _random_adata(5000, np.float32)
    sparse = _random_adata(5000, np.float32, sparse_type)
    expected = ClusteringAgreement._centroid_ann(dense, "perturbation", "control")
    actual = ClusteringAgreement._centroid_ann(sparse, "perturbation", "control")
    np.testing.assert_array_equal(
        actual.obs["perturbation"], expected.obs["perturbation"]
    )
    np.testing.assert_allclose(
        cast(np.ndarray, actual.X), cast(np.ndarray, expected.X), rtol=1e-12
    )
