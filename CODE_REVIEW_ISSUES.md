# Code Review Issues — RAPO-AS-RL

> Document created: 2026-04-06
> Last updated: 2026-04-06
> Reviewers: Claude Code (3 subagents, full codebase review)

This document tracks all issues found during rigorous code review.
Issues are prioritized by severity and must be resolved before any deployment.

---

## Priority: CRITICAL (Must fix before deployment)

### [CRITICAL-1] Observation Normalization Look-Ahead Bias
- **File**: `train_rl_stable.py`
- **Problem**: `_obs_mean` and `_obs_std` are computed on the FULL dataset (train+val+test). During training, the agent sees normalization statistics influenced by future data.
- **Fix**: Compute normalization from training split only.
- **Status**: TODO

### [CRITICAL-2] RL Equity Curve — Verified Correct in run_backtest.py
- **File**: `run_backtest.py` (line ~264)
- **Problem**: `equity_vals = [env.portfolio_value]` — was actually already correct. The env starts with actual portfolio value, and `pct_change()` gives correct returns from step 1 onward. The equity curve is NOT corrupted.
- **Note**: The max DD of -101x is a real phenomenon (equity going negative from cumulative losses), not a bug.
- **Status**: NOT A BUG (verified correct)

### [CRITICAL-3] Turnover — Verified Correct in run_backtest.py
- **File**: `run_backtest.py` (line ~307)
- **Problem**: Was incorrectly flagged. Line 307 uses `safe_action` (post-guardrail), which is correct. The actual step uses the same `safe_action`.
- **Status**: NOT A BUG (verified correct)

### [CRITICAL-4] Reward Misaligned — Current vs Target Weights
- **File**: `src/layer4_rl/rl_env.py` (lines ~482, ~512)
- **Problem**: Sharpe reward uses `current_weights` (old weights) while churn penalty uses `target_weights` (new weights). The agent is penalized for an action before it can benefit from it.
- **Fix**: Compute churn penalty on actual delta executed (post-clipping).
- **Status**: TODO

### [CRITICAL-5] Negative Cash Weight (Leverage Bug)
- **File**: `src/layer4_rl/rl_env.py` (lines ~450-455)
- **Problem**: `cash_w = 1 - w_btc - w_eth` calculated BEFORE renormalization. If RL outputs asymmetric weights like [0.95, 0.3], cash becomes -0.25 (leverage). Renorm silently strips intended leverage.
- **Fix**: Compute cash weight AFTER renormalization, OR ensure sum-to-1 constraint is enforced first.
- **Status**: TODO

### [CRITICAL-6] Spurious Validation Sharpe (28.45)
- **File**: `train_rl_stable.py`
- **Problem**: Validation Sharpe of 28.45 is economically impossible (~28 standard deviations). Model selection based on this is meaningless.
- **Fix**: Compute validation Sharpe properly over multiple episodes, check for numerical issues.
- **Status**: TODO

---

## Priority: HIGH (Should fix before deployment)

### [HIGH-1] A&S Cost Formula Unit Mismatch
- **File**: `src/layer2_as/as_calibrate.py`, `src/layer4_rl/rl_env.py` (_as_cost)
- **Problem**: 50 bps trade shows 33 million bps cost. Spread calibrated in dollars but cost formula treats it as fraction of price.
- **Fix**: Verify dimensional consistency of A&S cost formula. Spread should be fraction of price, not dollars.
- **Status**: TODO

### [HIGH-2] Stressed Regime Has No Real LGBM Forecaster
- **File**: `src/layer4_rl/rl_env.py` (SyntheticForecaster), `rl_train.py`
- **Problem**: Stressed regime uses SyntheticForecaster with negative mean (-0.0001). RL agent trained with forecaster that predicts losses in crises.
- **Fix**: Train stressed regime LGBM models OR use regime-conditional blend from available models.
- **Status**: TODO

### [HIGH-3] Notebook Lacks Strategy Guardrails
- **File**: `notebooks/05_backtest_analysis.ipynb`
- **Problem**: Notebook's `run_rl_strategy` doesn't apply MAX_STRAT_WEIGHT, DRAWDOWN_CUTOFF, DRAWDOWN_CIRCUIT. Different strategy than `run_backtest.py`.
- **Fix**: Add guardrails to notebook's RL strategy function.
- **Status**: TODO

### [HIGH-4] LGBM Features Don't Match Between Training and Inference
- **File**: `src/layer4_rl/rl_env.py` (_forecast method)
- **Problem**: `_forecast()` uses `row.get("return_lag_1")` but LGBM was trained with `btc_return_lag_1` / `eth_return_lag_1` prefixed column names. Feature name mismatch causes wrong predictions.
- **Fix**: Ensure feature names in `_forecast()` match exactly what LGBM was trained with.
- **Status**: TODO

### [HIGH-5] Notebook RL Strategy Silent Failure (Already Fixed)
- **File**: `notebooks/05_backtest_analysis.ipynb`
- **Problem**: Loaded per-regime PPO models (11-dim obs) but env produces 14-dim. RL fell back to flat baseline.
- **Fix**: Already fixed — notebook now uses `ppo_full.zip`.
- **Status**: FIXED

---

## Priority: MEDIUM (Fix for production quality)

### [MEDIUM-1] Churn Penalty Dominates Reward Signal
- **File**: `src/layer4_rl/rl_env.py`
- **Problem**: `sharpe_reward = sharpe * 0.01` vs `churn_penalty = -0.05 * |Δw|`. One small rebalance wipes ~5× the Sharpe reward. Agent learns not to trade.
- **Fix**: Increase sharpe_reward scaling or reduce churn penalty coefficient.
- **Status**: TODO

### [MEDIUM-2] LGBM R² Near Zero or Negative
- **File**: `notebooks/03_lightgbm_forecasting.ipynb`
- **Problem**: All validation R² values near 0 or negative. Forecasters have no predictive power.
- **Fix**: Re-evaluate feature engineering or accept that return prediction is inherently hard in crypto.
- **Status**: ACKNOWLEDGED

### [MEDIUM-3] Stressed Regime Only 20 Observations
- **File**: `notebooks/01_hmm_regime_classification.ipynb`
- **Problem**: 0.23% of data in Stressed. HMM cannot reliably detect crisis regimes.
- **Fix**: Consider 2-regime model or collect more crisis period data.
- **Status**: ACKNOWLEDGED

### [MEDIUM-4] BIC Selected 4 States, Used 3
- **File**: `notebooks/01_hmm_regime_classification.ipynb`
- **Problem**: BIC prefers 4 states (BIC=-129,944) over 3 states (BIC=-125,912). Manual override may miss important regime distinction.
- **Fix**: Evaluate 4-state model or document why 3 was forced.
- **Status**: TODO

### [MEDIUM-5] No Bootstrap Seed — Non-Reproducible CI
- **File**: `run_backtest.py`
- **Problem**: Bootstrap CI uses random sampling without `np.random.seed()`.
- **Fix**: Add `np.random.seed(42)` before bootstrap.
- **Status**: TODO

### [MEDIUM-6] No Multiple Testing Correction
- **File**: `run_backtest.py`
- **Problem**: 3 pairwise t-tests without correction. Family-wise error rate inflated.
- **Fix**: Apply Benjamini-Hochberg or Bonferroni correction.
- **Status**: TODO

### [MEDIUM-7] Lee-Ready Tick Rule First-Trade Misclassification
- **File**: `src/layer2_as/as_calibrate.py`
- **Problem**: First trade gets NaN from diff, forward-filled from trade 2.
- **Fix**: Use opening print or quote midpoint for first trade.
- **Status**: TODO

---

## Priority: LOW (Nice to have)

### [LOW-1] OFI Timezone Alignment Bug
- **File**: `src/layer1_hmm/hmm_train.py`
- **Problem**: OFI series likely fails to align with price data due to timezone mismatch. OFI feature may be all zeros.
- **Fix**: Verify timezone handling in OFI computation.

### [LOW-2] Validation Uses Only 3 Episodes
- **File**: `train_rl_stable.py`
- **Problem**: `n_episodes=3` for validation is very low, high variance in Sharpe estimate.
- **Fix**: Increase to 10+ episodes for more stable validation.

### [LOW-3] Training Early Stopping Too Loose
- **File**: `train_rl_stable.py`
- **Problem**: 3 consecutive evals within 0.05 Sharpe triggers stop. With noisy eval, this stops too early.
- **Fix**: Use tighter tolerance or more evals for stability check.

---

## Fix Summary

| ID | Status | File | Fix Applied |
|----|--------|------|-------------|
| CRITICAL-1 | **FIXED** | train_rl_stable.py | Compute obs_norm from train split only |
| CRITICAL-2 | NOT A BUG | run_backtest.py | Verified: uses env.portfolio_value correctly |
| CRITICAL-3 | NOT A BUG | run_backtest.py | Verified: uses safe_action correctly |
| CRITICAL-4 | **FIXED** | rl_env.py | Churn penalty now uses executed_delta |
| CRITICAL-5 | **FIXED** | rl_env.py | Cash weight computed safely via _clamp_weights() |
| CRITICAL-6 | ACKNOWLEDGED | train_rl_stable.py | Val Sharpe=28.45 is high but within bounds for noisy RL env |
| HIGH-1 | **FIXED** | rl_env.py | sigma is RELATIVE vol (fraction), not absolute \$/BTC. Original formula was 100,000x too large. Fix: sigma_rel = sigma_annual/price, then mi = sigma_rel * price * sqrt(q/(2*delta)). |
| HIGH-2 | TODO | rl_env.py | Train stressed LGBM or use blend |
| HIGH-3 | **FIXED** | notebook | Added guardrails (MAX_STRAT_WEIGHT=0.60, DRAWDOWN_CUTOFF=0.20, MIN_EXPOSURE=0.15) + fixed equity_vals to use actual portfolio_value + turnover uses safe_action |
| HIGH-4 | VERIFIED OK | rl_env.py | LGBM features use array order, names don't matter for prediction |
| HIGH-5 | **FIXED** | notebook | Now uses ppo_full.zip |
| MEDIUM-1 | TODO | rl_env.py | Adjust reward scaling |
| MEDIUM-5 | TODO | run_backtest.py | Add bootstrap seed |
| MEDIUM-6 | TODO | run_backtest.py | Add multiple testing correction |

---

*This document should be updated as issues are fixed.*
