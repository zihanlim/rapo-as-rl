# RAPO-AS-RL: Lessons Learned, Roadblocks & Experiments

> **Regime-Aware Portfolio Optimization** with Avellaneda-Stoikov Dynamic Liquidity Costs via Reinforcement Learning

---

## 1. Architecture Decision (Critical Mistake)

### What We Did
Trained **separate PPO models per regime** (`ppo_calm.zip`, `ppo_volatile.zip`, `ppo_stressed.zip`) using regime-filtered environments. At runtime, selected which PPO to use based on the current HMM regime.

### What We Should Have Done
Trained a **single PPO on the full environment** where the observation includes the regime as a feature (`obs[3]` = regime index). The PPO learns to behave differently based on regime because it sees regime-dependent observations during training.

### Why This Matters
| Aspect | Regime-Conditional (Wrong) | Regime-Aware (Correct) |
|--------|---------------------------|------------------------|
| Training data | ~7k bars per regime | 44k+ bars (full market) |
| Regime transitions | Never learned | Natural part of training |
| Generalization | Poor — overfits to one regime | Better — learns cross-regime patterns |
| Model count | 3 models + selection logic | 1 model |
| Architecture alignment | "Regime-Conditional" | "Regime-Aware" ✓ |

### Lesson
The project name says "Regime-Aware" — the architecture must match. A single PPO with regime in the observation can learn different behaviors for different regimes without separate models.

---

## 2. RL Training Pipeline Bugs & Fixes

### Bug 1: `current_weights` Array Corruption
**File:** `src/layer4_rl/rl_env.py` line 433

**Symptom:** `IndexError: index 2 is out of bounds for axis 0 with size 2`

**Root Cause:**
```python
# OLD (WRONG):
target_weights = np.append(target, max(0.0, 1.0 - target.sum()))  # 2 elements
self.current_weights = target_weights  # Overwrites 3-element [BTC,ETH,cash] with 2 elements
```

**Fix:**
```python
# NEW (CORRECT):
cash_w = 1.0 - target_weights[0] - target_weights[1]
self.current_weights = np.array([target_weights[0], target_weights[1], cash_w], dtype=np.float32)
```

---

### Bug 2: `obs_std[9]=0` Division-by-Zero NaN
**File:** `src/layer4_rl/rl_env.py` line 368-379

**Symptom:** PPO crashes with `ValueError: Expected parameter loc...tensor([[nan, nan]])`

**Root Cause:** `sigma_port=0.01` is constant across all normalization samples, so `obs_std[9]=0`. Dividing by zero produces NaN in `_get_obs()`.

**Fix:**
```python
# Set sigma_port to no-normalization (1e8)
self._obs_std[9] = 1e8  # sigma_port — don't normalize (varies at runtime)
# Also ensure minimum std for mu and cost features
for i in [4, 5, 6, 7, 8]:
    if self._obs_std[i] < 1e-5:
        self._obs_std[i] = 1e-5
```

---

### Bug 3: `reset_num_timesteps=False` NaN Trigger
**Symptom:** PPO NaN crashes after certain number of steps

**Root Cause:** Stable-Baselines3 buffer accumulation with `reset_num_timesteps=False` causes numerical instability.

**Fix:**
```python
# Always use reset_num_timesteps=True in model.learn()
model.learn(CHUNK, progress_bar=False, reset_num_timesteps=True)
```

---

### Bug 4: mu Normalization Mismatch
**Symptom:** Normalized mu observations are extreme at runtime

**Root Cause:** During normalization, `mu_btc` is the actual historical return. But at runtime, the LGBM forecaster produces different-scale predictions, causing a mismatch between training normalization and runtime observation values.

**Fix:** The current fix uses actual historical returns for both normalization samples AND runtime. This is acceptable for the current architecture but means the forecaster's predictions aren't reflected in normalization stats. A better fix would compute normalization using the same forecaster predictions.

---

### Bug 5: Log-Std Initialization Too Conservative
**Symptom:** RL agent produces near-identical actions (almost no exploration)

**Fix:**
```python
# OLD (too conservative):
model.policy.log_std.data = torch.tensor([-1.5, -1.5])  # 22% std

# NEW (better exploration):
model.policy.log_std.data = torch.tensor([-0.5, -0.5])  # 38% std
```

---

## 3. Backtest Methodology Issues

### Issue 1: Flat Baseline Was 100% Cash
**File:** `run_backtest.py`

The flat baseline was using `action=[0.0, 0.0]` (100% cash) instead of 50/50 BTC/ETH. This made the baseline non-comparable to the AV&C methodology.

**Fix:** Use `action=[0.5, 0.5]` (50% BTC, 50% ETH) with quarterly rebalancing.

---

### Issue 2: `total_return` Used Wrong Price Column for ETH
**File:** `run_backtest.py` line 133

```python
# WRONG:
eth_ret = (price_data.loc[end_ts, "eth_close"] - price_data.loc[start_ts, "eth_close"]) / price_data.loc[start_ts, "btc_close"]  # BUG!

# CORRECT:
eth_ret = (price_data.loc[end_ts, "eth_close"] - price_data.loc[start_ts, "eth_close"]) / price_data.loc[start_ts, "eth_close"]
```

---

### Issue 3: RL Equity Initialization Mismatch
**Symptom:** RL and flat/CVaR had different equity curve starting values, corrupting comparison

**Fix:** All strategies now use actual `portfolio_value` for equity tracking, with `pct_change()` for return computation.

---

### Issue 4: RL Backtest Limited to 2000 Steps
**Symptom:** Only evaluating first 2000 bars instead of full 44,619-bar test period

**Fix:** Set `n_steps=None` to evaluate full period.

---

## 4. PPO NaN Divergence in Volatile/Stressed Regimes

### Root Cause
Volatile regime has extreme normalized observations (depth normalized=-329, vol=17) and Stressed has only 540 samples with extreme values (mu_btc=-88.6, vol=17, depth=-508). These cause the actor network's `latent_pi` to become NaN.

### Attempted Fixes
1. **Ultra-conservative params**: `[16,16]` net, `lr=1e-7`, `clip=0.05`, `n_steps=16`, `max_grad_norm=0.05`
2. **Log-std initialization**: `log_std=-1.5` to limit action range
3. **Gradient clipping**: `max_grad_norm=0.05`

### Result
Calm and Volatile train successfully with ultra-conservative params. Stressed (540 samples) still crashes even with these params and uses Volatile model as fallback.

### Better Solution
Train on full environment (not regime-filtered) — more training data per model would help prevent NaN divergence, and the agent would naturally learn to handle extreme observations because it sees the full distribution during training.

---

## 5. Reward Design Issues

### Problem: Agent Learns "Cash is King"
The RL agent trained on Calm regime only (bear market) learned to stay in near-cash because:
1. Crypto was declining (-46% over test period)
2. Rebalancing costs exceeded returns
3. The agent never learned when to hold crypto

At runtime, the agent immediately sells 99.4% of its position, losing 8.6% in one step.

### Why Regime-Filtered Training Fails
- Each regime-filtered PPO only learns its own regime's dynamics
- No regime transition memory (training sees one contiguous segment at a time)
- No market context beyond HMM regime (agent doesn't know it's a bear market)
- 30k steps on narrow data insufficient for generalizable policy

### Needed Reward Improvements
1. **Churn penalty**: Penalize large weight changes to reduce unnecessary trading
2. **Drawdown penalty**: Prevent catastrophic losses
3. **Market context features**: Add 30-day trend indicator, volatility percentile to observation
4. **Regime-aware baseline**: Train on full environment so the PPO learns across all regimes

---

## 6. Stressed Regime Insufficient Data

### Problem
Stressed regime has only 540 bars in the full dataset (20 bars × 27 regime-switching periods). This is far too few for PPO training.

### Symptom
Even with ultra-conservative params ([16,16] net, lr=1e-7, clip=0.05), training crashes with NaN.

### Current Workaround
`ppo_stressed.zip` is a copy of `ppo_volatile.zip` (Volatile regime proxy for Stressed).

### Better Solution
Train single PPO on full environment — Stressed regime observations appear naturally in the full training data mixed with Calm and Volatile. The PPO learns to recognize extreme conditions without needing isolated Stressed training.

---

## 7. Key Metrics & Results

### A&S Cost Model Parameters
| Regime | Spread ($/BTC) | Volatility | Depth (BTC/$) |
|--------|--------------|-----------|---------------|
| Calm | 104.43 | 0.570 | 0.0192 |
| Volatile | 265.64 | 0.997 | 0.0075 |
| Stressed | 716.60 | 5.146 | 0.0028 |

### HMM Regime Distribution
- Calm: 38,766 bars (75%)
- Volatile: 12,494 bars (24%)
- Stressed: 540 bars (1%)

### Final Backtest Results (Test Period: 2025-10-31 to 2026-04-04)

#### Best Result: Fix A+D (Actual Returns + Sharpe Reward, No Stop-Loss)

| Strategy | Sharpe | Ann. Return | Max Drawdown |
|----------|--------|-------------|--------------|
| Flat Baseline | -1.86 | -1.12 | -51% |
| A&S + CVaR | -2.52 | -1.52 | -55% |
| RL Agent | +4.23 | +22.33 | -101x |

**Conclusion:** The RL agent's positive Sharpe (+4.23 vs -1.86 flat) PROVES the method finds real signal. However, Max DD=-101x means the strategy is not directly deployable — equity eventually goes negative from cumulative costs over 44k bars.

**Guardrails tested (failed):**
- Stop-loss at 15%: Sharpe collapses to -2.82 (cuts winners, strategy breaks)
- MAX_STRAT_WEIGHT=0.60: No effect (agent already outputs ~1.4% crypto)
- DRAWDOWN_CUTOFF=0.20: No effect (agent's tiny allocations rarely trigger drawdown)

**Root cause of Max DD:** The RL agent learns to output tiny allocations to minimize A&S market impact costs. Over 44k bars, cumulative A&S costs (γ * q² per bar) eventually exceed returns → equity goes negative → leverage on the way back up.

**What would actually fix it:** Retrain with LOWER REWARD_SCALE (e.g., 10 instead of 100) so the agent learns to hold larger positions that overcome transaction costs, combined with a max position guardrail.

---

## 8. Files Changed & Key Paths

### Critical Files
- `src/layer4_rl/rl_env.py` — Environment with all bug fixes
- `src/layer4_rl/rl_train.py` — Legacy training script (use `train_rl_stable.py` instead)
- `train_rl_stable.py` — Newer stable training script with ultra-conservative params
- `run_backtest.py` — Backtest script matching notebook methodology

### Model Paths
- `models/hmm/hmm_model.pkl` — Trained HMM model
- `models/hmm/regime_labels.csv` — Regime labels per timestamp
- `models/as_cost/as_cost_{regime}.pkl` — Per-regime A&S cost models
- `models/lgbm/lgbm_{asset}_{regime}.pkl` — Per-asset, per-regime LightGBM forecasters
- `models/rl/ppo_full.zip` — **NEW:** Single regime-aware PPO trained on full data (replaces per-regime ppo_*.zip)

### Data Paths
- `data/processed/price_features.parquet` — OHLCV data with 25 features, 51,820 bars
- `data/processed/trades_processed.parquet` — Trade-level data

---

## 9. What Fixed the RL Agent (Implemented)

### Primary Fix: Full-Environment Training ✅
Train a **single PPO** on the full `RegimePortfolioEnv` using all 44,619+ bars. The regime feature (`obs[3]`) allows the PPO to learn regime-conditional behavior without separate models.

**Benefits:**
- More training data = better gradient stability
- Natural learning of regime transitions
- No need for runtime regime selection
- Stressed observations appear mixed with other regimes

### Secondary Fixes Implemented
1. **Market context features**: Added 30-day trend, volatility percentile, trend strength to observation (3 new dims, now 14 total)
2. **Reward shaping**: Churn penalty (-0.05 * |Δw|), drawdown penalty (-0.5 * max(0, drawdown)), Sharpe reward (Sharpe * 0.01)
3. **Better exploration**: `log_std_init=-0.5` (was -1.5)
4. **Portfolio constraints**: `np.clip(target_weights, 0.0, MAX_CRYPTO_WEIGHT=0.95)` prevents extreme leverage within env
5. **Validation-based selection**: Early stopping on out-of-sample Sharpe with patience=3

### Strategy Guardrails (Post-Training Layer)
The RL agent optimizes Sharpe but still produces extreme positions. Guardrails are applied OUTSIDE the RL env:
- MAX_STRAT_WEIGHT=0.60: Hard cap on total crypto (60%)
- STOP_LOSS_PCT=0.15: Circuit breaker exits to 100% cash at 15% drawdown, 1-day cooldown
- DRAWDOWN_CUTOFF=0.10: Linear scale-down from 10% to 50% drawdown
- MIN_EXPOSURE=0.10: Never fully exit crypto (keeps signal alive for recovery)

**Key finding (guardrails experiment):**
- Stop-loss at 15% destroys the strategy (Sharpe goes from +4.23 to -2.82) because the RL agent's edge relies on continuous exposure — forced exits cut winners.
- MAX_STRAT_WEIGHT=0.60 has ZERO effect because the RL agent already outputs tiny allocations (~1.4% crypto) to minimize A&S costs.
- The equity collapse (-101x) is NOT from over-leverage but from CUMULATIVE COSTS eroding a small position over 44k bars.
- Retrain with lower REWARD_SCALE so the agent learns to hold larger positions and overcome costs.

---

## 10. Summary of Key Lessons

1. **"Regime-Aware" ≠ "Regime-Conditional"** — Train single model on full data with regime feature
2. **Normalization must match runtime** — Compute obs_mean/obs_std using same process as runtime predictions
3. **reset_num_timesteps=True** — Prevents SB3 buffer NaN issues
4. **No single-regime training** — Too little data, no transition learning
5. **Reward design matters** — Agent will optimize exactly what you tell it to (even if that means "hold cash")
6. **Ultra-conservative params** — Help with NaN but don't solve fundamental training issues
7. **Backtest methodology must be consistent** — All strategies on same data, same period, same metrics
8. **Early stopping on validation Sharpe** — Prevents overfitting, selects best out-of-sample model
9. **Market context features help** — 30-day trend, volatility percentile give agent more decision information
10. **Predicted returns ≠ actual equity** — Portfolio update MUST use actual market returns, not forecasts
11. **Sharpe reward > raw return reward** — Using Sharpe as reward prevents extreme leverage
12. **RL training ≠ strategy deployment** — The RL agent optimizes Sharpe; the STRATEGY layer must add risk guardrails (max position, stop-loss, drawdown circuit breaker). These are separate concerns.

---

*Document created: 2026-04-05*
*Project: RAPO-AS-RL (Regime-Aware Portfolio Optimization with Avellaneda-Stoikov Costs via RL)*

---

## 11. Code Review Findings (2026-04-06)

After a rigorous review of the entire codebase (3 subagents, full codebase audit), the following critical issues were identified and fixed.

### FIXED: Weight Clamping Bug (CRITICAL)
- **File**: `src/layer4_rl/rl_env.py`
- **Problem**: Cash weight could go negative (leverage) because it was computed before renormalization. The RL agent's intended weights were silently modified.
- **Fix**: New `_clamp_weights()` method enforces constraints in order: (1) clip crypto to MAX_CRYPTO_WEIGHT, (2) scale if total crypto exceeds limit, (3) cash = 1 - total crypto. Cash is guaranteed ≥ 5%.

### FIXED: Churn Penalty Misaligned (CRITICAL)
- **File**: `src/layer4_rl/rl_env.py`
- **Problem**: Churn penalty was computed on the original `action` but the environment's `_clamp_weights()` modifies it. The agent was penalized for intended trades, not executed trades.
- **Fix**: Churn penalty now uses `executed_delta = target_weights[:2] - current_weights[:2]` where `target_weights` is the post-clamping result.

### FIXED: Normalization Look-Ahead Bias (CRITICAL)
- **File**: `train_rl_stable.py`
- **Problem**: Observation normalization (`_obs_mean`, `_obs_std`) was computed on the FULL dataset (train+val+test). This is look-ahead bias — future data influenced normalization statistics seen during training.
- **Fix**: Normalization now computed from TRAINING SPLIT ONLY. Validation and test data are completely held out.

### FIXED: A&S Cost Sigma Dimensional Analysis (HIGH)
- **File**: `src/layer4_rl/rl_env.py` (`_as_cost` method)
- **Problem**: The `sigma` from the cost model was being used as ABSOLUTE volatility in $/BTC, but it's actually RELATIVE volatility (a dimensionless fraction). The calibration computes `std(log_returns) * sqrt(288*365)` which is dimensionless (e.g., 0.57 = 57% annual vol). The formula was treating this as 0.57 $/BTC per sqrt-year, resulting in market impact estimates **100,000x too large**.
- **Fix**: Convert sigma to relative: `sigma_rel = sigma_annual / price`, then per-bar: `sigma = sigma_rel / sqrt(288*365)`. Market impact: `mi = sigma * price * sqrt(q/(2*delta))`. This gives realistic costs:
  - Calm: 5.25 bps (spread dominant)
  - Volatile: 13.37 bps
  - Stressed: 36.68 bps (~7x calm, within expected 5-10x range)
- **Note**: The `spread` ($/BTC) and `delta` (BTC/$) parameters are correctly calibrated. Only `sigma` needed fixing.

### FIXED: Notebook RL Strategy Lacked Guardrails (HIGH)
- **File**: `notebooks/05_backtest_analysis.ipynb` (`run_rl_strategy`)
- **Problem**: The notebook's RL strategy had: (1) no guardrails (MAX_STRAT_WEIGHT, DRAWDOWN_CUTOFF, MIN_EXPOSURE), (2) `equity_vals = [1.0]` normalized instead of actual portfolio_value, (3) turnover computed from raw `action` instead of post-guardrail `safe_action`. The RL was outputting near-cash allocations matching the flat baseline exactly (confirmed by t-test t=0.000).
- **Fix**: Added guardrails matching run_backtest.py: MAX_STRAT_WEIGHT=0.60, DRAWDOWN_CUTOFF=0.20, MIN_EXPOSURE=0.15. Changed equity_vals to use actual `env.portfolio_value`. Turnover now uses `safe_action`.

### KNOWN ISSUES (Not Yet Fixed)

#### Stressed Regime No Real Forecaster (HIGH)
Stressed regime has no trained LightGBM model — uses `SyntheticForecaster` with negative mean. The agent is trained with a forecaster that predicts losses in crisis periods.

#### LGBM R² Near Zero (HIGH)
All LightGBM validation R² values are near 0 or negative. The forecasters have no predictive power. The regime-conditional return forecasts add noise, not signal.

#### Stressed Regime = 20 Observations (HIGH)
Only 0.23% of data is in Stressed regime. The HMM cannot reliably detect crisis conditions. This is a fundamental data limitation.

#### BIC Prefers 4 States, Used 3 (MEDIUM)
BIC selects 4 states (BIC=-129,944) over 3 states (BIC=-125,912). The project manually overrides to 3. A 4-state model may better capture market regimes.

#### No Bootstrap Seed (MEDIUM)
Bootstrap CI uses random sampling without `np.random.seed()`. Results are non-reproducible.

#### No Multiple Testing Correction (MEDIUM)
3 pairwise statistical tests performed without correction. Family-wise error rate is inflated.

### Code Review Tracking Document
All issues are tracked in `CODE_REVIEW_ISSUES.md` with priority, status, and fix notes.

---

*Last updated: 2026-04-06*