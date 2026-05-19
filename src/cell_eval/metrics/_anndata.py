"""Array metrics module."""

from logging import getLogger
from typing import Any, Callable, Literal, Sequence, cast

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.metrics as skm
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from .._types import PerturbationAnndataPair

logger = getLogger(__name__)

_L1_METRICS = {"l1", "manhattan", "cityblock"}
_L2_METRICS = {"l2", "euclidean"}


def pearson_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute Pearson correlation between mean differences from control."""
    real_effects, pred_effects = _bulk_effect_matrices(data, embed_key=embed_key)
    correlations = _rowwise_pearson(pred_effects, real_effects)
    return {
        str(pert): float(correlation)
        for pert, correlation in zip(data.perts, correlations)
    }


def mse(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=False, embed_key=embed_key
    )


def mae(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation from control."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=False, embed_key=embed_key
    )


def mse_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean squared error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_squared_error, use_delta=True, embed_key=embed_key
    )


def mae_delta(
    data: PerturbationAnndataPair, embed_key: str | None = None
) -> dict[str, float]:
    """Compute mean absolute error of each perturbation-control delta."""
    return _generic_evaluation(
        data, skm.mean_absolute_error, use_delta=True, embed_key=embed_key
    )


def edistance(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
    metric: str = "euclidean",
    **kwargs,
) -> float:
    """Compute Euclidean distance of each perturbation-control delta."""

    def _edistance(
        x: np.ndarray,
        y: np.ndarray,
        metric: str = "euclidean",
        precomp_sigma_y: float | None = None,
        **kwargs,
    ) -> float:
        sigma_x = skm.pairwise_distances(x, metric=metric, **kwargs).mean()
        sigma_y = (
            precomp_sigma_y
            if precomp_sigma_y is not None
            else skm.pairwise_distances(y, metric=metric, **kwargs).mean()
        )
        delta = skm.pairwise_distances(x, y, metric=metric, **kwargs).mean()
        return 2 * delta - sigma_x - sigma_y

    d_real = np.zeros(data.perts.size)
    d_pred = np.zeros(data.perts.size)

    # Precompute sigma for control data (reused by all perturbations)
    logger.info("Precomputing sigma for control data (real)")
    precomp_sigma_real = skm.pairwise_distances(
        data.ctrl_matrix(which="real", embed_key=embed_key), metric=metric, **kwargs
    ).mean()

    logger.info("Precomputing sigma for control data (pred)")
    precomp_sigma_pred = skm.pairwise_distances(
        data.ctrl_matrix(which="pred", embed_key=embed_key), metric=metric, **kwargs
    ).mean()

    for idx, delta in enumerate(data.iter_cell_arrays(embed_key=embed_key)):
        d_real[idx] = _edistance(
            delta.pert_real,
            delta.ctrl_real,
            precomp_sigma_y=precomp_sigma_real,
            metric=metric,
            **kwargs,
        )
        d_pred[idx] = _edistance(
            delta.pert_pred,
            delta.ctrl_pred,
            precomp_sigma_y=precomp_sigma_pred,
            metric=metric,
            **kwargs,
        )

    return pearsonr(d_real, d_pred).correlation


def discrimination_score(
    data: PerturbationAnndataPair,
    metric: str = "l1",
    embed_key: str | None = None,
    exclude_target_gene: bool = True,
) -> dict[str, float]:
    """Base implementation for discrimination score computation.

    Best score is 1.0 - worst score is 0.0.

    Args:
        data: PerturbationAnndataPair containing real and predicted data
        embed_key: Key for embedding data in obsm, None for expression data
        metric: Metric for distance calculation (e.g., "l1", "l2", see `scipy.metrics.pairwise.distance_metrics`)
        exclude_target_gene: Whether to exclude target gene from calculation

    Returns:
        Dictionary mapping perturbation names to normalized ranks
    """
    if metric == "l1" or metric == "manhattan" or metric == "cityblock":
        # Ignore the embedding key for L1
        embed_key = None

    real_effects, pred_effects = _bulk_effect_matrices(data, embed_key=embed_key)
    excluded_indices = _excluded_gene_indices(
        data,
        embed_key=embed_key,
        exclude_target_gene=exclude_target_gene,
    )
    distances = _pairwise_distances_with_exclusions(
        pred_effects=pred_effects,
        real_effects=real_effects,
        metric=metric,
        excluded_indices=excluded_indices,
    )
    order = np.argsort(distances, axis=1)
    ranks = np.argmax(order == np.arange(data.perts.size)[:, None], axis=1)

    return {
        str(pert): 1 - float(rank) / data.perts.size
        for pert, rank in zip(data.perts, ranks)
    }


def _bulk_effect_matrices(
    data: PerturbationAnndataPair,
    embed_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return real/pred perturbation-control effects in data.perts order."""
    data._initialize_bulk_arrays(embed_key)
    cache_key = embed_key or "_default"
    assert data.bulk_real is not None
    assert data.bulk_pred is not None
    keys, real_bulk = data.bulk_real[cache_key]
    _, pred_bulk = data.bulk_pred[cache_key]
    positions = {str(key): idx for idx, key in enumerate(keys)}
    pert_positions = np.array([positions[str(pert)] for pert in data.perts])
    ctrl_position = positions[str(data.control_pert)]
    real_effects = real_bulk[pert_positions] - real_bulk[ctrl_position]
    pred_effects = pred_bulk[pert_positions] - pred_bulk[ctrl_position]
    return np.asarray(real_effects), np.asarray(pred_effects)


def _rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered, axis=1)
        * np.sum(y_centered * y_centered, axis=1)
    )
    correlations = np.full(x.shape[0], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=correlations, where=denominator > 0)
    return correlations


def _excluded_gene_indices(
    data: PerturbationAnndataPair,
    embed_key: str | None,
    exclude_target_gene: bool,
) -> list[np.ndarray]:
    if embed_key or not exclude_target_gene:
        return [np.array([], dtype=np.int64) for _ in data.perts]
    return [np.flatnonzero(data.genes == pert) for pert in data.perts]


def _pairwise_distances_with_exclusions(
    pred_effects: np.ndarray,
    real_effects: np.ndarray,
    metric: str,
    excluded_indices: list[np.ndarray],
) -> np.ndarray:
    pred_effects = np.asarray(pred_effects, dtype=np.float64)
    real_effects = np.asarray(real_effects, dtype=np.float64)
    has_exclusions = any(indices.size > 0 for indices in excluded_indices)

    if metric in _L1_METRICS:
        distances = skm.pairwise_distances(
            pred_effects, real_effects, metric="manhattan"
        )
        for idx, excluded in enumerate(excluded_indices):
            if excluded.size:
                distances[idx] -= np.abs(
                    real_effects[:, excluded] - pred_effects[idx, excluded]
                ).sum(axis=1)
        np.maximum(distances, 0, out=distances)
        return distances

    if metric in _L2_METRICS:
        pred_sq = np.sum(pred_effects * pred_effects, axis=1)
        real_sq = np.sum(real_effects * real_effects, axis=1)
        distances_sq = (
            pred_sq[:, None] + real_sq[None, :] - 2 * (pred_effects @ real_effects.T)
        )
        for idx, excluded in enumerate(excluded_indices):
            if excluded.size:
                excluded_delta = real_effects[:, excluded] - pred_effects[idx, excluded]
                distances_sq[idx] -= np.sum(excluded_delta * excluded_delta, axis=1)
        np.maximum(distances_sq, 0, out=distances_sq)
        return np.sqrt(distances_sq)

    if metric == "cosine":
        return _cosine_distances_with_exclusions(
            pred_effects=pred_effects,
            real_effects=real_effects,
            excluded_indices=excluded_indices,
        )

    if not has_exclusions:
        return skm.pairwise_distances(pred_effects, real_effects, metric=metric)

    distances = np.empty((pred_effects.shape[0], real_effects.shape[0]))
    for idx, excluded in enumerate(excluded_indices):
        include_mask = np.ones(real_effects.shape[1], dtype=bool)
        include_mask[excluded] = False
        distances[idx] = skm.pairwise_distances(
            pred_effects[idx, include_mask].reshape(1, -1),
            real_effects[:, include_mask],
            metric=metric,
        ).ravel()
    return distances


def _cosine_distances_with_exclusions(
    pred_effects: np.ndarray,
    real_effects: np.ndarray,
    excluded_indices: list[np.ndarray],
) -> np.ndarray:
    dot = pred_effects @ real_effects.T
    pred_sq = np.sum(pred_effects * pred_effects, axis=1)
    real_sq = np.sum(real_effects * real_effects, axis=1)
    distances = np.empty_like(dot)

    for idx, excluded in enumerate(excluded_indices):
        row_dot = dot[idx].copy()
        row_pred_sq = pred_sq[idx]
        row_real_sq = real_sq.copy()
        if excluded.size:
            row_dot -= (real_effects[:, excluded] * pred_effects[idx, excluded]).sum(
                axis=1
            )
            row_pred_sq -= float(np.sum(pred_effects[idx, excluded] ** 2))
            row_real_sq -= np.sum(real_effects[:, excluded] ** 2, axis=1)
        denominator = np.sqrt(max(row_pred_sq, 0.0)) * np.sqrt(
            np.maximum(row_real_sq, 0.0)
        )
        similarity = np.zeros_like(row_dot)
        np.divide(row_dot, denominator, out=similarity, where=denominator > 0)
        distances[idx] = 1 - similarity

    np.clip(distances, 0, 2, out=distances)
    return distances


def _generic_evaluation(
    data: PerturbationAnndataPair,
    func: Callable[[np.ndarray, np.ndarray], float],
    use_delta: bool = False,
    embed_key: str | None = None,
) -> dict[str, float]:
    """Generic evaluation function for anndata pair."""
    res = {}
    for bulk_array in data.iter_bulk_arrays(embed_key=embed_key):
        if use_delta:
            x = bulk_array.perturbation_effect(which="pred", abs=False)
            y = bulk_array.perturbation_effect(which="real", abs=False)
        else:
            x = bulk_array.pert_pred
            y = bulk_array.pert_real

        result = func(x, y)
        if isinstance(result, tuple):
            result = result[0]

        res[bulk_array.key] = float(result)

    return res


# TODO: clean up this implementation
class ClusteringAgreement:
    """Compute clustering agreement between real and predicted perturbation centroids."""

    def __init__(
        self,
        embed_key: str | None = None,
        real_resolution: float = 1.0,
        pred_resolutions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
        metric: Literal["ami", "nmi", "ari"] = "ami",
        n_neighbors: int = 15,
    ) -> None:
        self.embed_key = embed_key
        self.real_resolution = real_resolution
        self.pred_resolutions = pred_resolutions
        self.metric = metric
        self.n_neighbors = n_neighbors

    @staticmethod
    def _score(
        labels_real: Sequence[int],
        labels_pred: Sequence[int],
        metric: Literal["ami", "nmi", "ari"],
    ) -> float:
        if metric == "ami":
            return adjusted_mutual_info_score(labels_real, labels_pred)
        if metric == "nmi":
            return normalized_mutual_info_score(labels_real, labels_pred)
        if metric == "ari":
            return (adjusted_rand_score(labels_real, labels_pred) + 1) / 2
        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _cluster_leiden(
        adata: ad.AnnData,
        resolution: float,
        key_added: str,
        n_neighbors: int = 15,
    ) -> None:
        if key_added in adata.obs:
            return
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(
                adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), use_rep="X"
            )
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
        )

    @staticmethod
    def _centroid_ann(
        adata: ad.AnnData,
        category_key: str,
        control_pert: str,
        embed_key: str | None = None,
    ) -> ad.AnnData:
        # Isolate the features
        # embed_key may be None; narrow it before .get (whose key must be str) and
        # keep feats untyped since it may be a dense array or a sparse matrix.
        feats: Any = (
            adata.X if embed_key is None else adata.obsm.get(embed_key, adata.X)
        )

        # Convert to float if not already
        if feats.dtype != np.dtype("float64"):
            feats = feats.astype(np.float64)

        # Densify if required
        if issparse(feats):
            feats = feats.toarray()

        cats = cast(pd.Series, adata.obs[category_key]).values
        uniq, inv = np.unique(cats, return_inverse=True)
        centroids = np.zeros((uniq.size, feats.shape[1]), dtype=feats.dtype)

        for i, cat in enumerate(uniq):
            mask = cats == cat
            if np.any(mask):
                centroids[i] = feats[mask].mean(axis=0)

        adc = ad.AnnData(X=centroids)
        adc.obs[category_key] = uniq
        return adc[adc.obs[category_key] != control_pert]

    def __call__(self, data: PerturbationAnndataPair) -> float:
        cats_sorted = sorted([c for c in data.perts if c != data.control_pert])

        # 2. build centroids
        ad_real_cent = self._centroid_ann(
            adata=data.real,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )
        ad_pred_cent = self._centroid_ann(
            adata=data.pred,
            category_key=data.pert_col,
            control_pert=data.control_pert,
            embed_key=self.embed_key,
        )

        # 3. cluster real once
        real_key = "real_clusters"
        self._cluster_leiden(
            ad_real_cent, self.real_resolution, real_key, self.n_neighbors
        )
        ad_real_cent.obs = (
            cast(pd.DataFrame, ad_real_cent.obs)
            .set_index(data.pert_col)
            .loc[cats_sorted]
        )
        real_labels = pd.Categorical(ad_real_cent.obs[real_key])

        # 4. sweep predicted resolutions
        best_score = 0.0
        ad_pred_cent.obs = (
            cast(pd.DataFrame, ad_pred_cent.obs)
            .set_index(data.pert_col)
            .loc[cats_sorted]
        )
        for r in self.pred_resolutions:
            pred_key = f"pred_clusters_{r}"
            self._cluster_leiden(ad_pred_cent, r, pred_key, self.n_neighbors)
            pred_labels = pd.Categorical(ad_pred_cent.obs[pred_key])
            score = self._score(real_labels, pred_labels, self.metric)  # type: ignore
            best_score = max(best_score, score)

        return float(best_score)
