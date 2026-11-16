import polars as pl
import pytest

from cell_eval import score_agg_metrics


@pytest.mark.parametrize(
    ("metric", "user", "base"),
    [
        ("mae", 1.0, 2.0),
        ("discrimination_score_l1", 0.75, 0.5),
        ("overlap_at_N", 0.75, 0.5),
    ],
)
def test_score_single_normalization_group(metric, user, base):
    """Profiles and skip_metrics can leave only one normalization group."""
    user_agg = pl.DataFrame({metric: [user, user]}).describe()
    base_agg = pl.DataFrame({metric: [base, base]}).describe()

    result = score_agg_metrics(user_agg, base_agg)

    assert result["metric"].to_list() == [metric, "avg_score"]
    assert result["from_baseline"].to_list() == pytest.approx([0.5, 0.5])
