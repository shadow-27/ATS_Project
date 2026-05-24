This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: **/*.py
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
__init__.py
estimate_spread.py
preprocessing.py
validation.py
```

# Files

## File: __init__.py
```python
from .estimate_spread import estimate_spread

__all__ = ["estimate_spread"]
```

## File: estimate_spread.py
```python
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
```

## File: preprocessing.py
```python
"""
Data loading and 1-second bucketing for ATS Project 1 (Roll's model).
Owner: Sh

Public API
----------
load_trades(ticker, date, data_dir)     -> pd.DataFrame
build_trade_buckets(trades)             -> pd.DataFrame  (1-Hz: last_price, vwap, volume)
load_all_trades(ticker, data_dir)       -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def load_trades(
    ticker: str,
    date: str,
    data_dir: Path | str = _DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """
    Load one day of trades for *ticker* from a parquet file.

    Parameters
    ----------
    ticker : e.g. "WIFUSDT" or "ZAMAUSDT"
    date   : "2026-04-12" (used to build the filename)
    data_dir : path to the folder containing parquet files

    Returns
    -------
    DataFrame with DatetimeIndex (UTC), columns: price, amount
    Sorted ascending, duplicate timestamps removed (keep last).
    The ``side`` column is intentionally dropped — Roll's model must not use it.
    """
    path = Path(data_dir) / f"{ticker}_trades_{date}.parquet"
    df = pd.read_parquet(path)

    df.index = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
    df.index.name = "time"
    df = df[["price", "amount"]].sort_index()

    df = df[~df.index.duplicated(keep="last")]
    return df


def build_trade_buckets(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trade-level data to 1-Hz OHLCV buckets.

    Parameters
    ----------
    trades : DataFrame with DatetimeIndex, columns price + amount

    Returns
    -------
    DataFrame at 1-second frequency with columns:
        last_price  — last traded price in the second (used for Roll's Δp)
        vwap        — volume-weighted average price (used as mid-price proxy)
        volume      — total trade volume in the second
    Empty seconds are forward-filled for last_price/vwap; volume = 0.
    """
    resampled = trades.resample("1s").agg(
        last_price=("price", "last"),
        volume=("amount", "sum"),
        _pv=("price", lambda x: (x * trades.loc[x.index, "amount"]).sum()),
    )

    resampled["vwap"] = np.where(
        resampled["volume"] > 0,
        resampled["_pv"] / resampled["volume"],
        np.nan,
    )
    resampled = resampled.drop(columns=["_pv"])

    resampled["last_price"] = resampled["last_price"].ffill()
    resampled["vwap"] = resampled["vwap"].ffill()
    resampled["volume"] = resampled["volume"].fillna(0.0)

    return resampled


def load_all_trades(
    ticker: str,
    data_dir: Path | str = _DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """
    Load and concatenate all available trading days for *ticker*.
    Handles timestamp overlaps at day boundaries by deduplicating.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(f"{ticker}_trades_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No trade files found for {ticker} in {data_dir}")

    dates = [f.stem.split("_trades_")[1] for f in files]
    frames = [load_trades(ticker, date, data_dir) for date in dates]
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined
```

## File: validation.py
```python
"""
Orderbook-based spread calculation and evaluation metrics.
Owner: Sh

Public API
----------
compute_ob_spread(ticker, date, data_dir)   -> pd.Series  (1-Hz relative spread)
evaluate(estimated, actual)                 -> dict
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def compute_ob_spread(
    ticker: str,
    date: str,
    data_dir: Path | str = _DEFAULT_DATA_DIR,
) -> pd.Series:
    """
    Compute the 1-Hz relative bid-ask spread from orderbook snapshots.

    Uses best bid (bid0) and best ask (ask0).
    Relative spread = (ask0 - bid0) / mid, where mid = (ask0 + bid0) / 2.

    Each 1-second bucket takes the snapshot whose timestamp is closest to
    the second boundary (i.e. resample with nearest-neighbour).

    Returns
    -------
    pd.Series with DatetimeIndex (UTC, 1-second frequency), name="ob_spread"
    """
    path = Path(data_dir) / f"{ticker}_orderbook_{date}.parquet"
    ob = pd.read_parquet(path, columns=["timestamp", "bid0", "ask0"])

    ob.index = pd.to_datetime(ob["timestamp"], unit="ns", utc=True)
    ob.index.name = "time"
    ob = ob[["bid0", "ask0"]].sort_index()

    mid = (ob["ask0"] + ob["bid0"]) / 2
    rel_spread = (ob["ask0"] - ob["bid0"]) / mid

    # resample: take the value of the snapshot closest to each 1-second mark
    spread_1hz = rel_spread.resample("1s").last()
    spread_1hz.name = "ob_spread"
    return spread_1hz


def evaluate(
    estimated: pd.Series,
    actual: pd.Series,
) -> dict:
    """
    Compute out-of-sample metrics comparing estimated vs actual spread.

    Both series must have a DatetimeIndex. They are inner-joined on their
    common timestamps before computing metrics.

    Returns
    -------
    dict with keys: mae, rmse, correlation, n_obs, frac_valid
        frac_valid: fraction of seconds where estimated is non-NaN
    """
    df = pd.concat({"est": estimated, "act": actual}, axis=1).dropna()
    n = len(df)
    if n == 0:
        return dict(mae=np.nan, rmse=np.nan, correlation=np.nan, n_obs=0, frac_valid=0.0)

    mae = (df["est"] - df["act"]).abs().mean()
    rmse = np.sqrt(((df["est"] - df["act"]) ** 2).mean())
    corr = df["est"].corr(df["act"])

    total = max(len(estimated), len(actual))
    frac_valid = n / total if total > 0 else 0.0

    return dict(mae=mae, rmse=rmse, correlation=corr, n_obs=n, frac_valid=frac_valid)


def print_evaluation_table(results: dict[str, dict]) -> None:
    """
    Pretty-print a table of evaluation results.

    Parameters
    ----------
    results : {label: evaluate(...) output, ...}
              e.g. {"WIFUSDT Apr-12": {...}, "ZAMAUSDT Apr-14": {...}}
    """
    header = f"{'Label':<25} {'MAE':>10} {'RMSE':>10} {'Corr':>8} {'N obs':>8}"
    print(header)
    print("-" * len(header))
    for label, m in results.items():
        print(
            f"{label:<25} {m['mae']:>10.5f} {m['rmse']:>10.5f} "
            f"{m['correlation']:>8.4f} {m['n_obs']:>8d}"
        )
```
