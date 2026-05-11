# ATS_Project

## Project 1 Draft (ATS Class): Roll-Model Spread Estimation

### Objective
Estimate a 1-Hz (1-second) time series of **relative spread** from trades-only data, then validate it against the true 1-second spread from order-book quotes.

### Data and Constraints
- **Allowed trade inputs only (unsigned):** trade price, trade volume, trade timestamp.
- **No signed trade direction** (no buyer/seller initiator labels).
- Validation target uses order-book data:  
  $$
  \text{Relative Spread}_t = \frac{\text{Ask}_t - \text{Bid}_t}{(\text{Ask}_t + \text{Bid}_t)/2}
  $$

### Method (Roll Model Core)
At the trade level, compute price changes:
$$
\Delta p_i = p_i - p_{i-1}
$$
Under Roll-style microstructure assumptions, adjacent trade-price changes have negative autocovariance:
$$
\text{Cov}(\Delta p_i,\Delta p_{i-1}) < 0
$$
The implied effective spread estimate is:
$$
\hat{S}_i = 2\sqrt{-\text{Cov}(\Delta p_i,\Delta p_{i-1})}
$$
Here, $\hat{S}_i$ denotes the trade-level effective spread estimate; the final 1-Hz output is the aggregated relative series $\hat{s}_t^{\text{roll}}$. If the estimated covariance is non-negative in a window, set the estimator to 0 (or mark as missing) for that window and flag it for diagnostics, since the Roll assumption is not supported there.

Convert this to a **1-second series** by estimating covariance in rolling windows and mapping spread to relative terms using a local mid-price proxy from trades.

### Suggested 1-Hz Pipeline
1. Clean trade prints (de-duplicate obvious repeats, sort by time, remove bad ticks).
2. Aggregate trades into 1-second buckets.
3. In each rolling window (e.g., 30-300 seconds), estimate lag-1 covariance of $\Delta p$; choose smaller windows for faster regime tracking (higher noise) and larger windows for smoother estimates (more lag), then select via out-of-sample validation error.
4. Convert $\hat{S}$ to relative spread using a local price level.
5. Output $\hat{s}_t^{\text{roll}}$: estimated 1-second relative spread series.
6. Compare to observed order-book relative spread $s_t^{\text{book}}$.

### Validation Plan
- **Primary metrics:** correlation, MAE, RMSE between $\hat{s}_t^{\text{roll}}$ and $s_t^{\text{book}}$.
- **Diagnostics:** intraday stability plots, error by liquidity regime (high versus low trade activity), robustness to window length.
- **Baseline checks:** persistence baseline and simple moving-average benchmark.

### Deliverables
- A reproducible notebook/script implementing the estimator.
- Plots of estimated vs observed 1-second relative spread.
- Short discussion of assumptions, limitations, and where Roll-model estimates break down (e.g., fast regime changes, sparse trading, nonstationarity).
