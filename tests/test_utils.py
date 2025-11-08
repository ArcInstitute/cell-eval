import numpy as np
import pytest

from cell_eval.data import build_random_anndata
from cell_eval.utils import guess_is_lognorm


def test_is_lognorm_true():
    data = build_random_anndata(normlog=True)
    assert guess_is_lognorm(data)


def test_is_lognorm_false():
    data = build_random_anndata(normlog=False)
    assert not guess_is_lognorm(data)


def test_guess_is_lognorm_valid_lognorm():
    """Test that valid log1p normalized data returns True."""
    data = build_random_anndata(normlog=True, random_state=42)
    # Should return True without raising exception
    assert (
        guess_is_lognorm(
            data,
        )
        is True
    )


def test_guess_is_lognorm_valid_lognorm_sparse():
    """Test that valid log1p normalized sparse data returns True."""
    data = build_random_anndata(normlog=True, as_sparse=True, random_state=42)
    # Should return True without raising exception
    assert (
        guess_is_lognorm(
            data,
        )
        is True
    )


def test_guess_is_lognorm_integer_data():
    """Test that integer data (raw counts) returns False."""
    data = build_random_anndata(normlog=False, random_state=42)
    # Should return False - integer data indicates raw counts
    assert (
        guess_is_lognorm(
            data,
        )
        is False
    )


def test_guess_is_lognorm_edge_case_near_threshold():
    """Test that values near but below threshold return True."""
    data = build_random_anndata(normlog=True, random_state=42)
    # Modify data to have values near threshold (10.9)
    data.X = np.random.uniform(
        0,
        10.9,
        size=data.X.shape,  # type: ignore
    )
    # Should return True without raising exception
    assert (
        guess_is_lognorm(
            data,
        )
        is True
    )


def test_guess_is_lognorm_exceeds_threshold():
    """Test that data with max value > 11.0 raises ValueError when ."""
    data = build_random_anndata(normlog=True, random_state=42)
    # Modify data to exceed threshold (mix of valid and invalid)
    data.X = np.random.uniform(
        0,
        11.1,
        size=data.X.shape,  # type: ignore
    )

    with pytest.raises(ValueError, match="Invalid scale.*exceeds log1p threshold"):
        guess_is_lognorm(
            data,
        )


def test_guess_is_lognorm_negative_values():
    """Test that data with negative values raises ValueError when ."""
    data = build_random_anndata(normlog=True, random_state=42)
    # Modify data to include negative values
    data.X = np.random.uniform(
        -1,
        9,
        size=data.X.shape,  # type: ignore
    )

    with pytest.raises(ValueError, match="Invalid scale.*is negative"):
        guess_is_lognorm(
            data,
        )


def test_guess_is_lognorm_mixed_scales():
    """Test mixed scenario: some cells with raw counts, some with log1p."""
    data = build_random_anndata(normlog=True, random_state=42)
    n_cells = data.X.shape[0]  # type: ignore
    half = n_cells // 2
    data.X[:half] = np.random.uniform(0, 9, size=(half, data.X.shape[1]))  # type: ignore
    data.X[half:] = np.random.uniform(100, 5000, size=(n_cells - half, data.X.shape[1]))  # type: ignore

    with pytest.raises(ValueError, match="Invalid scale.*exceeds log1p threshold"):
        guess_is_lognorm(
            data,
        )
