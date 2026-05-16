# ATS Project 1 — Roll's Model Spread Estimation

## Overview

Estimate a 1-Hz time series of **relative bid-ask spread** from trades-only data, then validate against the true 1-second spread from order-book quotes.

**Course:** Algorithmic Trading Systems (ATS)  
**Team:** S, Sh, Gr

---

## Method

Price changes exhibit negative serial autocovariance due to bid-ask bounce (Roll, 1984):

$$\hat{S} = 2\sqrt{\max(0,\; -\text{Cov}(\Delta p_t,\, \Delta p_{t-1}))}$$

Converted to relative terms using the VWAP mid-price proxy from trades:

$$\hat{s}_t^{\text{rel}} = \frac{\hat{S}}{m_t}$$

When the estimated covariance is non-negative (trending market), the spread is clamped to 0.

---

## Repository Structure

```
ATS_Project/
├── data/                        # raw parquet files (not committed to git)
│   ├── WIFUSDT_trades_*.parquet
│   ├── WIFUSDT_orderbook_*.parquet
│   ├── ZAMAUSDT_trades_*.parquet
│   └── ZAMAUSDT_orderbook_*.parquet
├── src/
│   ├── __init__.py
│   ├── estimate_spread.py       # core deliverable — estimate_spread() (S)
│   ├── preprocessing.py         # data loading + 1-sec bucketing (Sh)
│   └── validation.py            # orderbook spread + metrics (Sh)
├── notebooks/
│   ├── 01_eda.ipynb             # exploratory data analysis (Gr)
│   ├── 02_validation.ipynb      # validation plots + metrics table (S)
│   └── 03_extensions.ipynb      # EWM Roll, volume-weighted Roll (Gr)
├── tests/
│   └── test_estimate_spread.py  # unit tests — 14 tests (S)
└── requirements.txt
```

---

## Data

Two crypto tickers, 3 days each (Apr 12–14 2026):

| Ticker | Total trades | Notes |
|--------|-------------|-------|
| WIFUSDT | ~55,607 | Lower frequency, good for development |
| ZAMAUSDT | ~641,585 | Higher frequency; Apr 14 has a ~70% price move (stress test) |

**Trades schema:** `price`, `amount`, `timestamp` (int64 nanoseconds). The `side` column exists in the raw files but is intentionally unused — Roll's model requires only unsigned trade information.  
**Orderbook schema:** `bid0`…`bid9`, `ask0`…`ask9` + volumes, `timestamp` (nanoseconds). 10-level L2 snapshots.

**Train/test split:** Apr 12–13 for development and window tuning, Apr 14 as final out-of-sample validation.

---

## Setup

### 1. Create the virtual environment

The project uses Python 3.13. Create the venv at a short path to avoid Windows 260-character path limits:

```
py -3.13 -m venv C:\venvs\ats
```

### 2. Install dependencies

```
C:\venvs\ats\Scripts\pip install -r requirements.txt
```

### 3. Select the interpreter in VS Code

Command palette → `Python: Select Interpreter` → enter `C:\venvs\ats\Scripts\python.exe`

---

## Usage

### Core function

```python
from src.estimate_spread import estimate_spread

# trade_price and trade_volume are time-indexed pd.Series
spread_1hz = estimate_spread(trade_price, trade_volume, window=60)
```

**Input:** time-indexed `pd.Series` of trade prices and volumes (DatetimeIndex or int64 nanosecond index).  
**Output:** 1-Hz `pd.Series` of estimated relative spread, indexed by second.  
**Window:** rolling window in seconds (default 60); tune via cross-validation — see `notebooks/02_validation.ipynb`.

### Loading data

```python
from src.preprocessing import load_trades, load_all_trades
from src.validation import compute_ob_spread, evaluate

trades = load_trades('WIFUSDT', '2026-04-12')        # single day
trades = load_all_trades('WIFUSDT')                  # all 3 days

ob_spread = compute_ob_spread('WIFUSDT', '2026-04-12')
metrics = evaluate(estimated_spread, ob_spread)
# -> {'mae': ..., 'rmse': ..., 'correlation': ..., 'n_obs': ..., 'frac_valid': ...}
```

---

## Running Tests

```
C:\venvs\ats\Scripts\python -m pytest tests/ -v
```

14 tests covering return shape, value correctness, edge cases (sparse ticks, constant prices, nanosecond index), and window behaviour.

---

## Notebooks

| Notebook | Owner | Purpose |
|----------|-------|---------|
| `01_eda.ipynb` | Gr | Price paths, trade frequency heatmaps, intraday spread patterns, ZAMAUSDT Apr-14 anomaly |
| `02_validation.ipynb` | S | Window CV, full metrics table, time-series and scatter plots |
| `03_extensions.ipynb` | Gr | EWM Roll and volume-weighted Roll vs basic Roll comparison |

---

## Validation

Validation target: `(ask0 - bid0) / mid` resampled to 1-Hz from orderbook snapshots.

**Metrics:** MAE, RMSE, Pearson correlation (reported per ticker per day in `02_validation.ipynb`).

---

## Extensions

Beyond basic Roll's model (`03_extensions.ipynb`):

- **EWM Roll** — exponentially weighted covariance; faster reaction, less lag
- **Volume-weighted Roll** — weight Δp by √(volume / mean_volume); up-weights high-turnover seconds

---

## References

- Roll, R. (1984). *A simple implicit measure of the effective bid-ask spread in an efficient market.* Journal of Finance, 39(4), 1127–1139.
