# Data Prep — Handoff Notes

## What was done

Loaded, validated, and prepared raw trade and orderbook data for both symbols across all 3 days.
The `results/` folder now contains everything you need to start implementing the Roll Model.

---

## Output files

### `results/{SYMBOL}_trades_clean.parquet`
Clean trade data combined across all dates, ready to use.

| Column | Type | Description |
|--------|------|-------------|
| `price` | float64 | Trade price |
| `volume` | float64 | Trade volume |
| `is_sell_aggressor` | bool | True = sell aggressor, False = buy aggressor |

- Index: datetime (UTC, tz-naive)
- `timestamp` column removed (redundant with index)

---

### `results/{SYMBOL}_ground_truth_spread_1s.parquet`
The actual spread computed from the orderbook, resampled to 1-second intervals.
**This is what the Roll Model estimate will be validated against.**

| Column | Type | Description |
|--------|------|-------------|
| `mid` | float64 | Mid price: (ask0 + bid0) / 2 |
| `abs_spread` | float64 | Absolute spread: ask0 − bid0 |
| `relative_spread` | float64 | Relative spread: abs_spread / mid |

- Index: datetime (UTC, tz-naive), 1-second frequency
- Only seconds with at least one orderbook update are included (gaps are dropped)

---

## Key observations

- **No missing values** in either trades or orderbook data — no extra cleaning needed.
- **WIFUSDT** has a very stable spread (~0.52% on average).
- **ZAMAUSDT** has a much tighter spread on average (~0.044%) but shows a large spike on 2026-04-14 (max ~1.1%). Worth investigating during the analysis phase.

---

## Notebooks (in `notebooks/`)

| File | What it does |
|------|-------------|
| `01_data_inspection.ipynb` | Explores raw data structure, dtypes, null counts |
| `02_compute_ground_truth_spread.ipynb` | Computes and saves ground truth spread from orderbook |
| `03_prepare_trades.ipynb` | Cleans and saves combined trades data |