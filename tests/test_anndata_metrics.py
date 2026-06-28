import anndata as ad
import numpy as np
import sklearn.metrics as skm
from scipy import sparse
from scipy.stats import pearsonr

from cell_eval._types import PerturbationAnndataPair
from cell_eval.data import CONTROL_VAR, PERT_COL, build_random_anndata
from cell_eval.metrics._anndata import discrimination_score, pearson_delta


def _metric_pair() -> PerturbationAnndataPair:
    real = build_random_anndata(
        n_cells=500,
        n_genes=12,
        n_perts=5,
        n_celltypes=1,
        random_state=17,
    )
    pert_names = [f"pert_{idx}" for idx in range(5)]
    real.var_names = pert_names + [f"gene_{idx}" for idx in range(7)]
    labels = np.resize(np.array([CONTROL_VAR, *pert_names]), real.n_obs)
    real.obs[PERT_COL] = labels

    pred = real.copy()
    rng = np.random.default_rng(23)
    pred_x = np.asarray(pred.X)
    pred.X = np.clip(pred_x + rng.normal(0, 0.01, size=pred_x.shape), 0, None)

    return PerturbationAnndataPair(
        real=real,
        pred=pred,
        control_pert=CONTROL_VAR,
        pert_col=PERT_COL,
    )


def _reference_pearson_delta(data: PerturbationAnndataPair) -> dict[str, float]:
    res = {}
    for bulk_array in data.iter_bulk_arrays():
        x = bulk_array.perturbation_effect(which="pred", abs=False)
        y = bulk_array.perturbation_effect(which="real", abs=False)
        res[bulk_array.key] = float(pearsonr(x, y).correlation)
    return res


def _reference_discrimination_score(
    data: PerturbationAnndataPair,
    metric: str,
    exclude_target_gene: bool = True,
) -> dict[str, float]:
    real_effects = np.vstack(
        [
            d.perturbation_effect(which="real", abs=False)
            for d in data.iter_bulk_arrays()
        ]
    )
    pred_effects = np.vstack(
        [
            d.perturbation_effect(which="pred", abs=False)
            for d in data.iter_bulk_arrays()
        ]
    )

    norm_ranks = {}
    for p_idx, pert in enumerate(data.perts):
        if exclude_target_gene:
            include_mask = np.flatnonzero(data.genes != pert)
        else:
            include_mask = np.ones(real_effects.shape[1], dtype=bool)
        distances = skm.pairwise_distances(
            real_effects[:, include_mask],
            pred_effects[p_idx, include_mask].reshape(1, -1),
            metric=metric,
        ).flatten()
        sorted_indices = np.argsort(distances)
        pert_index = np.flatnonzero(data.perts == pert)[0]
        rank = np.flatnonzero(sorted_indices == pert_index)[0]
        norm_ranks[str(pert)] = 1 - rank / data.perts.size
    return norm_ranks


def test_bulk_anndata_matches_manual_means_for_dense_and_sparse() -> None:
    matrix = np.array(
        [
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
            [4.0, 6.0, 8.0],
            [5.0, 7.0, 9.0],
        ]
    )
    labels = np.array(["pert_b", "pert_a", "pert_b", CONTROL_VAR, "pert_a"])
    expected_keys = np.array([CONTROL_VAR, "pert_a", "pert_b"])
    expected_values = np.array(
        [
            [4.0, 6.0, 8.0],
            [3.5, 5.5, 7.5],
            [2.0, 4.0, 6.0],
        ]
    )

    dense = ad.AnnData(X=matrix)
    dense.obs[PERT_COL] = labels
    dense_keys, dense_values = PerturbationAnndataPair._bulk_anndata(dense, PERT_COL)

    sparse_adata = ad.AnnData(X=sparse.csr_matrix(matrix))
    sparse_adata.obs[PERT_COL] = labels
    sparse_keys, sparse_values = PerturbationAnndataPair._bulk_anndata(
        sparse_adata, PERT_COL
    )

    np.testing.assert_array_equal(dense_keys, expected_keys)
    np.testing.assert_allclose(dense_values, expected_values)
    np.testing.assert_array_equal(sparse_keys, expected_keys)
    np.testing.assert_allclose(sparse_values, expected_values)


def test_pearson_delta_matches_reference() -> None:
    data = _metric_pair()

    expected = _reference_pearson_delta(data)
    actual = pearson_delta(data)

    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(
        list(actual.values()),
        list(expected.values()),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_discrimination_score_matches_reference() -> None:
    data = _metric_pair()

    for metric in ["l1", "l2", "cosine"]:
        expected = _reference_discrimination_score(data, metric=metric)
        actual = discrimination_score(data, metric=metric)

        assert actual.keys() == expected.keys()
        np.testing.assert_allclose(
            list(actual.values()),
            list(expected.values()),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
