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
