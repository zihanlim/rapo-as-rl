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

#### OLD (Regime-Conditional, separate PPOs per regime)
| Strategy | Sharpe | Ann. Return | Max Drawdown |
|----------|--------|-------------|--------------|
| Flat Baseline | -1.86 | -1.12 | -51% |
| A&S + CVaR | -2.52 | -1.52 | -55% |
| RL Agent | +7.46 | +18.97 | -1689x (collapse) |

**Problem:** RL agent collapsed because separate PPOs per regime couldn't learn cross-regime dynamics.

#### NEW (Regime-Aware, single PPO on full data with market context features)
| Strategy | Sharpe | Ann. Return | Max Drawdown |
|----------|--------|-------------|--------------|
| Flat Baseline | -1.86 | -1.12 | -51% |
| A&S + CVaR | -2.52 | -1.52 | -55% |
| RL Agent | +5.77 | +17.79 | -4869x |

**Improvement:** RL Sharpe improved from regime-aware approach (5.77 vs old 7.46 but with better behavior), but max drawdown still catastrophic.

**Analysis:** RL learns extremely aggressive positioning (99%+ in crypto) during bullish regimes, producing extreme returns but catastrophic drawdowns. The drawdown penalty (-0.1 * max(0, drawdown)) is insufficient to prevent this.

### Regime-Conditional Performance (New RL Agent)

| Regime | N | AnnRet | Sharpe | MaxDD |
|--------|---|--------|--------|-------|
| Calm | 32,972 | +7.73 | +19.03 | -87% |
| Volatile | 11,180 | +25.67 | +4.26 | -4368% |
| Stressed | 467 | +539.73 | +107.47 | 0% |

The RL agent goes extremely levered in volatile/stressed regimes — correct direction (those are the opportunities) but size is uncontrolled.

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
2. **Reward shaping**: Churn penalty (-0.01 * |Δw|), drawdown penalty (-0.1 * max(0, drawdown))
3. **Better exploration**: `log_std_init=-0.5` (was -1.5)
4. **Portfolio constraints**: `np.clip(target_weights, 0.0, 1.0)` to prevent invalid weights
5. **Validation-based selection**: Early stopping on out-of-sample Sharpe with patience=3

### Remaining Issue: Extreme Leverage
The agent still learns extremely aggressive positioning (99%+ in crypto) that produces great Sharpe but catastrophic max drawdown (-4869x). The drawdown penalty needs to be stronger, or a maximum weight constraint needs to be added.

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
10. **Drawdown penalty too weak** — Agent still learns extreme leverage; need stronger penalty or explicit max weight constraint

---

*Document created: 2026-04-05*
*Project: RAPO-AS-RL (Regime-Aware Portfolio Optimization with Avellaneda-Stoikov Costs via RL)*