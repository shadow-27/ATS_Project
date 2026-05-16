"""
Unit tests for estimate_spread().

Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.estimate_spread import estimate_spread


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_bounce_series(
    n_seconds: int = 300,
    mid_price: float = 1.0,
    half_spread: float = 0.01,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Synthetic trade series with pure bid-ask bounce (no drift).
    Alternates between bid and ask, giving known true spread = 2 * half_spread.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_seconds, freq="s", tz="UTC")
    signs = rng.choice([-1, 1], size=n_seconds)
    prices = mid_price + signs * half_spread
    volumes = rng.uniform(10, 100, size=n_seconds)
    return (
        pd.Series(prices, index=idx, name="price"),
        pd.Series(volumes, index=idx, name="amount"),
    )


def _make_trend_series(
    n_seconds: int = 300,
    start_price: float = 1.0,
    drift_per_second: float = 0.001,
) -> tuple[pd.Series, pd.Series]:
    """Strongly trending series — Roll covariance will be positive, clamp to 0."""
    idx = pd.date_range("2026-01-01", periods=n_seconds, freq="s", tz="UTC")
    prices = start_price + drift_per_second * np.arange(n_seconds)
    volumes = np.ones(n_seconds) * 50.0
    return (
        pd.Series(prices, index=idx, name="price"),
        pd.Series(volumes, index=idx, name="amount"),
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestReturnShape:
    def test_returns_series(self):
        price, vol = _make_bounce_series()
        out = estimate_spread(price, vol, window=30)
        assert isinstance(out, pd.Series)

    def test_index_is_datetime(self):
        price, vol = _make_bounce_series()
        out = estimate_spread(price, vol, window=30)
        assert isinstance(out.index, pd.DatetimeIndex)

    def test_frequency_is_1hz(self):
        price, vol = _make_bounce_series(n_seconds=120)
        out = estimate_spread(price, vol, window=30)
        assert out.index.freq is not None or (out.index.diff()[1:].median().total_seconds() == 1.0)

    def test_output_name(self):
        price, vol = _make_bounce_series()
        out = estimate_spread(price, vol)
        assert out.name == "estimated_spread"


class TestValues:
    def test_bounce_series_positive_spread(self):
        """Pure bid-ask bounce must yield a positive estimated spread."""
        price, vol = _make_bounce_series(n_seconds=300, half_spread=0.01)
        out = estimate_spread(price, vol, window=60)
        valid = out[out > 0]
        assert len(valid) > 0, "expected positive spread values for bounce series"

    def test_bounce_series_approximate_magnitude(self):
        """
        For a perfect alternating +/- h bounce, Roll's formula gives 2h.
        Relative spread ≈ 2h / mid.  We allow ±50% tolerance given the
        rolling window warm-up.
        """
        mid = 1.0
        h = 0.01
        true_rel = 2 * h / mid  # = 0.02
        price, vol = _make_bounce_series(n_seconds=600, mid_price=mid, half_spread=h)
        out = estimate_spread(price, vol, window=60)
        median_est = out[out > 0].median()
        assert abs(median_est - true_rel) / true_rel < 0.50, (
            f"Estimated relative spread {median_est:.4f} too far from true {true_rel:.4f}"
        )

    def test_trending_series_clamps_to_zero(self):
        """Strongly trending prices → positive covariance → spread should be 0."""
        price, vol = _make_trend_series(n_seconds=300, drift_per_second=0.01)
        out = estimate_spread(price, vol, window=60)
        assert (out >= 0.0).all(), "spread must be non-negative"
        assert out.median() == 0.0, "pure trend should produce zero spread"

    def test_no_negative_values(self):
        price, vol = _make_bounce_series(n_seconds=200, seed=99)
        out = estimate_spread(price, vol, window=30)
        assert (out >= 0.0).all()

    def test_no_nan_in_output(self):
        price, vol = _make_bounce_series(n_seconds=200)
        out = estimate_spread(price, vol, window=30)
        assert not out.isna().any(), "output must not contain NaN"


class TestEdgeCases:
    def test_sparse_ticks_no_crash(self):
        """One trade every 10 seconds — many empty buckets."""
        idx = pd.date_range("2026-01-01", periods=30, freq="10s", tz="UTC")
        price = pd.Series(np.random.uniform(0.99, 1.01, 30), index=idx)
        vol = pd.Series(np.ones(30) * 100.0, index=idx)
        out = estimate_spread(price, vol, window=30)
        assert isinstance(out, pd.Series)
        assert (out >= 0.0).all()

    def test_nanosecond_int_index(self):
        """trade_price with int64 nanosecond index (as stored in parquet)."""
        idx_dt = pd.date_range("2026-01-01", periods=120, freq="s", tz="UTC")
        idx_ns = idx_dt.view("int64")
        price = pd.Series(
            np.where(np.arange(120) % 2 == 0, 1.01, 0.99), index=idx_ns
        )
        vol = pd.Series(np.ones(120) * 50.0, index=idx_ns)
        out = estimate_spread(price, vol, window=30)
        assert isinstance(out.index, pd.DatetimeIndex)

    def test_constant_price_no_crash(self):
        """All prices identical — covariance = 0 → spread = 0."""
        idx = pd.date_range("2026-01-01", periods=120, freq="s", tz="UTC")
        price = pd.Series(np.ones(120), index=idx)
        vol = pd.Series(np.ones(120) * 50.0, index=idx)
        out = estimate_spread(price, vol, window=30)
        assert (out == 0.0).all()

    def test_different_window_sizes(self):
        """Smoke test: various window sizes should not raise."""
        price, vol = _make_bounce_series(n_seconds=400)
        for w in [30, 60, 120, 300]:
            out = estimate_spread(price, vol, window=w)
            assert isinstance(out, pd.Series)


class TestWindowTuning:
    def test_larger_window_smoother_output(self):
        """
        After warm-up, a wider window should be smoother than a narrow one.
        Compare only the settled tail (last 60% of the series) to exclude
        the leading zeros while both windows are filling up.
        """
        price, vol = _make_bounce_series(n_seconds=600, seed=7)
        out_30  = estimate_spread(price, vol, window=30)
        out_120 = estimate_spread(price, vol, window=120)
        cutoff = int(len(out_30) * 0.40)
        assert out_120.iloc[cutoff:].std() <= out_30.iloc[cutoff:].std()
