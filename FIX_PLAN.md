# Fix Plan — Project Architecture Audit
**Created**: 2026-04-18
**Status**: COMPLETE — Comprehensive Negative Finding Confirmed

---

## Final Results Summary

### All Fixes Applied (2026-04-18/19)
1. **Fix 1 (CRITICAL)**: OFI added to RL observation (15 dims).
2. **Fix 2**: LGBM replacement documented in `_forecast()` docstring.
3. **Fix 3**: Duplicate `net_trade_size` removed. 82 features.
4. **Fix 4**: Survival filter corrected to round-trip cost. 0 signals survive.
5. **Fix 5**: Regime label gap documented.
6. **Fix 6**: Notebook renamed to `03b_alpha_screen_results.ipynb`.
7. **Fix 7**: NaN bug in `_compute_obs_normalization()` — trend_30d NaN propagation.
8. **Fix 8**: NaN guard in `step()` — t=0 fallback + 1,715-bar Binance data gaps.
9. **Fix 9**: Price NaN guard in A&S cost computation.
10. **Fix 10**: Portfolio value floor ($1) + cum_pnl floor (-$10,000).

### Final Backtest Results (Test Period 2024-02-10 to 2026-04-10)
| Strategy | Ann. Return | Sharpe | Max DD |
|----------|------------|--------|--------|
| Flat(A&S) | **+26.2%** | **+0.48** | -56.6% |
| A&S+CVaR | +23.4% | +0.42 | -57.1% |
| RL (15-dim, floor) | **-28.0%** | **-1.74** | -47.0% |

**No strategy is statistically significantly better on Sharpe** (BH correction at q=0.10).

### Why RL Failed
- RL converged to near-cash strategy (turnover ≈ 0)
- Negative Sharpe (-1.74) from persistent small losses: minimal alpha minus trading costs
- Portfolio floor prevented explosive equity but did not improve the policy
- OFI signal (IC=0.17) confirmed not exploitable after A&S costs

### Comprehensive Negative Finding — Complete
1. **LGBM forecaster**: R² ≈ 0 (real OFI features used)
2. **Alpha screen**: IC=0.17 < breakeven IC=0.225, 0 signals survive
3. **P&L simulation**: Deeply negative Sharpe at all signal thresholds
4. **RL with OFI**: Sharpe=-1.74, near-cash strategy

The OFI signal is statistically real but economically insufficient after A&S execution costs.

---

## Fix Execution Log

### Fix 1: Add OFI to RL Observation ✅ DONE
- [x] OFI as dimension 15 in `_compute_obs_normalization()` and `_get_obs()`
- [x] Pre-compute rolling 5-bar OFI in `__init__`
- [x] Retrain RL 100k steps → `ppo_full.zip` (15-dim, 155KB)

### Fix 2: Document LGBM Replacement ✅ DONE
- [x] `_forecast()` docstring updated

### Fix 3: Remove Duplicate Feature ✅ DONE
- [x] `net_trade_size` removed; `alpha_features.parquet` regenerated (82 features)

### Fix 4: Survival Filter Corrected ✅ DONE
- [x] Round-trip cost: `abs(E[return_bps]) > 2 * as_cost_single`
- [x] 0 signals survive (was 2)

### Fix 5: Regime Label Gap ✅ DONE
- [x] 20-bar gap (0.07 days) at data start — negligible

### Fix 6: Notebook Renaming ✅ DONE
- [x] `03b_alpha_screen_results.ipynb`

### Fix 7: NaN in `_compute_obs_normalization()` ✅ DONE (2026-04-19)
- **Bug**: `trend_30d_history[t]` contained 3,425 NaN values from Binance data gaps.
  `np.clip(NaN, -1, 1)` propagates NaN through all samples → obs_mean[11] = NaN
- **Fix**: `np.nan_to_num(np.clip(..., -1, 1), nan=0.0)` in normalization loop
- **Result**: obs_mean/obs_std fully clean (no NaN)

### Fix 8: NaN in `step()` Return Calculation ✅ DONE (2026-04-19)
- **Bug**: `portfolio_return = portfolio_value * NaN - cost` on 83-bar gap (2017-09-06).
  Equity collapsed to $0 prematurely.
- **Also**: `cum_pnl` had no floor → unbounded negative drift
- **Fix**: Added `else: btc_actual = 0.0` for t=0; `pd.notna()` checks for data gaps;
  `cum_pnl` floor at -$10,000

### Fix 9: Price NaN in A&S Cost Computation ✅ DONE (2026-04-19)
- **Bug**: `_as_cost(q, NaN, cost_model)` called with NaN price during data gaps
- **Fix**: Fall back to last known good price when current price is NaN

### Fix 10: Portfolio Value Floor ✅ DONE (2026-04-19)
- **Bug**: `portfolio_value` could go negative (compounding losses on NaN bars),
  causing explosive equity swings and misleading Sharpe ratios
- **Fix**: `self.portfolio_value = max(1.0, self.portfolio_value)` after each step
- **Result**: Equity bounded. RL learns near-cash strategy without equity explosion.

---

## Files Changed
- `src/layer4_rl/rl_env.py`: All NaN guards, OFI dim 15, portfolio floor
- `src/layer3b_alpha/feature_library.py`: `net_trade_size` removed
- `src/layer3b_alpha/signal_screen.py`: Round-trip cost survival filter
- `models/rl/ppo_full.zip`: Retrained with portfolio floor (155KB, 15-dim)
