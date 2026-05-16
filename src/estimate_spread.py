"""
Roll's model bid-ask spread estimator — ATS Project 1 core deliverable.
Owner: S

Public API
----------
estimate_spread(trade_price, trade_volume, window, min_periods) -> pd.Series
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import build_trade_buckets


def estimate_spread(
    trade_price: pd.Series,
    trade_volume: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Estimate a 1-Hz time series of relative bid-ask spread using Roll's (1984) model.

    Uses only unsigned trade information (price + volume). The aggressor side
    (``side`` column in the raw data) must NOT be passed in — Roll's model
    recovers the spread purely from the serial covariance of price changes.

    Parameters
    ----------
    trade_price  : time-indexed pd.Series of trade prices.
                   Index must be a DatetimeIndex (UTC) or convertible from
                   int64 nanoseconds via pd.to_datetime(..., unit="ns").
    trade_volume : time-indexed pd.Series of trade volumes, aligned with
                   trade_price.
    window       : rolling window size in seconds (default 60).
                   Tune with cross-validation; see notebooks/02_validation.ipynb.
    min_periods  : minimum observations required in a window to produce a
                   non-NaN estimate (default: window // 2).

    Returns
    -------
    pd.Series
        1-Hz DatetimeIndex (UTC), values = estimated relative spread (dimensionless).
        Seconds with no trades are forward-filled from the last active second.
        Seconds where Roll's covariance is positive (trending market) return 0.0.
        The series is named "estimated_spread".

    Notes
    -----
    Roll's model:
        Ŝ = 2 * sqrt(max(0, -Cov(Δp_t, Δp_{t-1})))
    Relative spread:
        Ŝ_rel = Ŝ / mid_price
    where mid_price is the VWAP of trades within each 1-second bucket.
    """
    if min_periods is None:
        min_periods = max(window // 2, 2)

    trade_price = _ensure_datetime_index(trade_price)
    trade_volume = _ensure_datetime_index(trade_volume)

    trades = pd.DataFrame({"price": trade_price, "amount": trade_volume})
    buckets = build_trade_buckets(trades)

    price_1hz = buckets["last_price"]
    mid_1hz = buckets["vwap"]

    delta_p = price_1hz.diff()

    roll = delta_p.rolling(window=window, min_periods=min_periods)
    cov = roll.cov(delta_p.shift(1))

    spread_abs = 2.0 * np.sqrt(np.maximum(0.0, -cov))
    spread_rel = spread_abs / mid_1hz.replace(0, np.nan)

    spread_rel = spread_rel.fillna(0.0)
    spread_rel.name = "estimated_spread"
    return spread_rel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.to_datetime(s.index, unit="ns", utc=True)
    elif s.index.tz is None:
        s = s.copy()
        s.index = s.index.tz_localize("UTC")
    return s
