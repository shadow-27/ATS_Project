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
