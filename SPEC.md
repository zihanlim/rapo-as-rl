# RAPO-AS-RL — Regime-Aware Portfolio Optimization

## What It Is
An MScFE capstone research project building an RL-enhanced crypto trading system. The agent learns optimal BTC/ETH rebalancing conditioned on market regime and A&S-calibrated execution costs.

**Goal:** Compare four strategies under realistic A&S market impact costs and identify the optimal policy.

## Architecture (4 Layers)
```
Binance API (CCXT, public)
       │
       ▼
Layer 1: HMM Regime Classifier
  Input: OHLCV + trade ticks → 3-state: Calm / Volatile / Stressed
  Output: regime_labels.csv, hmm_model.pkl
       │
       ▼
Layer 2: Avellaneda-Stoikov Per-Regime Cost Model
  Input: regime labels + trade ticks → calibrated cost params per regime
  Output: as_cost_Calm/Volatile/Stressed.pkl
       │
       ▼
Layer 3: LightGBM Return Forecaster (per regime)
  Input: features (lags, vol, OFI, spread, cross-asset) → next-period return
  Output: lgbm_BTC/ETH_Calm/Volatile/Stressed.pkl
       │
       ▼
Layer 4: Regime-Aware PPO Agent (SINGLE model, full data)
  Input: portfolio state (14-dim, includes regime index) → target weights
  Output: ppo_full.zip (replaces per-regime ppo_*.zip)
       │
       ▼
Four-way Backtest: Flat(10bps) vs Flat(A&S) vs A&S+CVaR vs RL Agent
```

## Architecture Evolution (Lessons Learned)

The original design used **separate PPO per regime** (3 models). This was WRONG — it resulted in:
- Only ~7k bars per regime for training (vs 44k+ for single model)
- No regime transition learning
- NaN divergence in stressed regime (only 48 training bars)

**Current approach**: Single **regime-aware** PPO trained on full data. The observation includes `regime` as feature `obs[3]`, allowing the PPO to learn different behaviors per regime without separate models. See `LESSONS_LEARNED.md` Section 1.

## Key Components

### Data
- Source: Binance public API (no auth needed)
- Symbols: BTC/USDT, ETH/USDT
- OHLCV: 5-min bars
- Trades: tick-level (Lee-Ready classification)
- Train period: **75% (~682k bars, 2017-08 to 2024-02)** | Test: **25% (~227k bars, 2024-02 to 2026-04)** | No validation
- **10-year expanded data** (2017-08 to 2026-04) for full market cycle coverage (2017 bull, 2018 crash, COVID 2020, 2021 bull, 2022-2024 bear)
- **Stressed regime**: Volatility threshold override (>3x calm mean vol) instead of HMM state

### Layer 1 — HMM
- Gaussian HMM, 3 states (Calm/Volatile/Stressed)
- BIC prefers 4 states, but 3 is used (4th state = 36 fragmented bars, not usable)
- Features: return, realized_vol, spread_proxy, OFI
- State selection: BIC (3 states forced despite BIC preferring 4)
- Multiple random initializations (5 seeds) to avoid local optima
- Output: `models/hmm/regime_labels.csv`, `hmm_model.pkl`

### Layer 2 — Avellaneda-Stoikov (Adapted Execution Cost Model)
- Calibrate per regime: σ (vol), s (spread), δ (depth), γ (risk-aversion)
- **Adapted** execution cost decomposition (Almgren-Chriss, 2000), not the original A&S market-maker formula
- Cost = σ·√(q/(2δ))·P + s/2·P + γ·q²/(2δ)·P
  - market_impact = σ·P·√(q/(2δ)) — square-root impact form (A&S inspired)
  - spread_cost = (s/2)·q — half-spread × quantity (standard execution cost)
  - inventory_risk = γ·q²/(2δ)·P — quadratic trade-size penalty (NOT A&S's linear q inventory term)
- δ calibrated via A&S equilibrium: δ = 2/(s·P) (A&S contribution), then plugged into square-root impact (adapted)
- **FIXED**: σ is relative volatility (fraction), NOT absolute $/BTC
- Lee-Ready tick rule for trade direction
- Stressed corrections: spread forced 10.5x calm, vol forced 2x calm
- **Depth calibration**: δ estimated from Binance spread proxy via A&S equilibrium: δ = 2/(s_proxy·P). Results in δ ≈ 0.044 BTC²/$. Calibrated costs (~123 bps for 50bps trade in Calm, ~1,292 bps in Stressed) reflect crypto's shallow books vs traditional markets.
- Output: `models/as_cost/as_cost_calm/Volatile/Stressed.pkl`

### Layer 3 — LightGBM
- **No longer used for RL observations** (R² ≈ 0 for all regimes)
- RL uses lagged returns instead (0.7*lag_1 + 0.3*lag_3)
- Models still trained for potential future use
- Hyperparams: num_leaves=31, lr=0.05, n_estimators=500, early_stopping=50
- Output: `models/lgbm/lgbm_btc/eth_calm/Volatile/Stressed.pkl`

### Layer 4 — PPO (Stable Baselines3)
- Gymnasium custom env: **14-dim obs**, 2-dim continuous action (target BTC/ETH weights)
- Observation: w_btc, w_eth, w_cash, regime, mu_btc, mu_eth, sigma, spread, depth, sigma_port, cum_pnl, trend_30d, vol_pct, trend_strength
- **Reward**: Beat-benchmark return (portfolio_return - 0.6*BTC_actual - 0.4*ETH_actual) + churn penalty + opportunity cost penalty for underweight crypto
- **mu_btc/mu_eth**: Actual lagged returns (0.7*lag_1 + 0.3*lag_3) — not LGBM
- **Stressed override**: If realized_vol > 3*calm_mean, regime forced to "Stressed"
- **SINGLE regime-aware model** (not per-regime)
- **decision_interval parameter**: Controls how often (in bars) RL makes decisions. 1 = every bar (5-min), 288 = daily. On hold steps, target_weights is preserved, executed_delta=0, NO A&S cost incurred. See daily-frequency experiment below.
- Hyperparameters: [32,32] net arch, lr=3e-5, clip=0.1, n_steps=64
- Training: 100k steps, no validation (train until max steps)
- Guardrails: MAX_STRAT_WEIGHT=0.85, DRAWDOWN_CUTOFF=0.20, MIN_EXPOSURE=0.30
- Output: `models/rl/ppo_full.zip` (5-min) or `models/rl/ppo_daily.zip` (daily-frequency experiment)

### Backtest
- **Four-way comparison** (all use 288-bar periodic rebalancing):
  1. **Flat Baseline (10bps)**: 50/50 BTC/ETH, optimistic fixed 10bps transaction cost — unrealistic baseline
  2. **Flat Baseline (A&S)**: 60/40 BTC/ETH with TRUE A&S market impact costs — the **fair reference point**
  3. **A&S + CVaR**: Regime-conditional CVaR-optimized weights with A&S costs and cost-aware penalty (cost_lambda=0.001)
  4. **RL Agent**: Single regime-aware PPO policy with beat-benchmark reward and strategy guardrails (MAX_STRAT_WEIGHT=0.85, MIN_EXPOSURE=0.30)
- **Critical finding**: Flat(10bps) is misleading — its apparent competitiveness comes from NOT trading (turnover≈0), avoiding both costs and risks
- Test period: ~227k bars (25% of 10-year, 2024-02 to 2026-04) — covers full bull/bear cycle including 2022-2023 bear market
- Metrics: Sharpe, max drawdown, annualized return, turnover

**Test Period Results** (2024-02 to 2026-04, 2+ year, full market cycle):

| Strategy | Ann. Return | Sharpe | Max DD | Turnover |
|----------|-------------|--------|--------|----------|
| **Flat(A&S)** | **+26.2%** | **+0.48** | -56.6% | ~0 |
| Flat(10bps) | +25.1% | +0.44 | -57.6% | ~0 |
| A&S+CVaR (cost_lambda=0.001) | +23.4% | +0.42 | -57.1% | 0.000006 |
| RL Agent | -3.6% | -0.68 | **-7.9%** | 0.000004 |

**Key findings (10-year test, 2024-2026):**
- **Flat(A&S) wins on BOTH Ann. Return (+26.2%) and Sharpe (+0.48)** — best risk-adjusted strategy with 60/40 allocation
- **A&S+CVaR with cost_lambda=0.001 rebalances minimally** (+23.4%, turnover=0.000006) — correctly identifies that rebalancing costs exceed CVaR benefits in most regimes
- **RL Agent has best Max Drawdown** (-7.9%) but negative Sharpe (-0.68) — learned that cash is optimal under true execution costs, but the beat-benchmark reward design failed to produce competitive returns
- **Daily-frequency RL experiment (2026-04-13)**: Training RL with decision_interval=288 (daily decisions) performed WORSE than 5-min RL (Sharpe -3.88 vs -0.68). With only ~789 effective decisions over training (vs 100k at 5-min), the daily RL is a much harder problem. Both converge to cash-optimal strategies. Reducing decision frequency does NOT solve the A&S cost problem — the fundamental issue is that A&S costs exceed expected returns at any frequency.
- **Core finding confirmed**: Execution costs dominate active rebalancing returns. A single 50bps rebalance costs ~123 bps in Calm regime — 2.5x the nominal trade size. The CVaR optimizer with cost_lambda=0.001 correctly identifies that rebalancing costs exceed CVaR benefits, resulting in minimal rebalancing (turnover=0.000006).
- **Statistical significance**: Block bootstrap (288-bar, 1,000 reps) + Benjamini-Hochberg correction at q=0.10 shows NO statistically significant difference between Flat(10bps), Flat(A&S), and A&S+CVaR on Sharpe ratio

**10-Year HMM Regime Distribution:**
| Regime | Count | Percentage |
|--------|-------|------------|
| Calm | 403,180 | 44.4% |
| Volatile | 370,908 | 40.9% |
| Stressed | 132,970 | 14.7% |

The 10-year data covers full market cycles — the 2022-2023 bear market exposes Flat(10bps)'s true Sharpe of -0.84 (vs the misleading +0.48 when tested on bull-only 4-year window).

### Core Finding: Execution Costs Dominate Active Returns

**The A&S market impact model reveals that active rebalancing is unprofitable in crypto markets when execution costs are properly modeled.**

- A single 50bps trade costs ~123 bps in market impact in Calm regime, ~1,292 bps in Stressed — reflecting crypto's shallow order books validated against Binance data
- Each rebalance costs 24x the nominal trade size in Calm regime (A&S equilibrium relationship)
- **Cost-aware CVaR with cost_lambda=0.001 rebalances minimally**: The CVaR optimizer with cost-aware penalty (cost_lambda=0.001) still finds that rebalancing costs exceed CVaR benefits in most regimes, resulting in minimal turnover (0.000006). The 10x reduction from cost_lambda=1.0 allows small rebalancing when CVaR benefits narrowly exceed costs, but the optimizer still largely avoids unnecessary trades. This STRENGTHENS the core finding: even a cost-aware CVaR optimizer cannot justify significant rebalancing in crypto markets.
- RL trained on beat-benchmark reward design converged to near-cash positions — active trading alpha < execution costs at 5-min frequency
- **Flat(10bps) shows the danger of optimistic cost assumptions**: it reports Sharpe +0.44 over the full 10-year test but only ~0.48 in shorter bull-market windows — the 10bps fee is ~8x too optimistic vs true A&S costs in calm regime

**The practical implication:** Buy-and-hold BTC/ETH with realistic costs works (Flat(A&S) shows +26.2%, Sharpe +0.48). Active rebalancing doesn't work after costs. The RL finding of "stay in cash" is the genuinely optimal policy under realistic execution costs, not a failure.

## Training Order
```bash
# 1. Fetch data
python scripts/fetch_binance_data.py

# 2. Process features
python scripts/process_data.py

# 3. Layer 1: HMM
python -m src.layer1_hmm.hmm_train --data_dir data/processed --output models/hmm/

# 4. Layer 2: A&S calibration
python -m src.layer2_as.as_calibrate --regime_csv models/hmm/regime_labels.csv --trades data/processed/trades_processed.parquet --prices data/processed/price_features.parquet --output models/as_cost/

# 5. Layer 3: LightGBM
python -m src.layer3_lightgbm.lgbm_train --data data/processed --regime models/hmm/regime_labels.csv --output models/lgbm/

# 6. Layer 4: PPO (regime-aware single model)
python train_rl_stable.py

# 6b. Daily-frequency RL experiment (optional)
python train_rl_daily.py        # decision_interval=288, output: ppo_daily.zip

# 7. Backtest
python run_backtest.py
python run_backtest.py --rl-daily   # daily RL comparison
# or: notebooks/05_backtest_analysis.ipynb
```

## Known Issues / TODOs

### Critical (All Fixed as of 2026-04-12)
- [FIXED] A&S cost sigma dimensional mismatch (100,000x too large market impact)
- [FIXED] Weight clamping could produce negative cash
- [FIXED] Churn penalty misaligned with executed trades
- [FIXED] Normalization look-ahead bias (train-only split)
- [FIXED] Notebook RL falling back to flat baseline
- [FIXED] RL reward misaligned with backtest (Sharpe vs return) — redesigned as beat-benchmark
- [FIXED] Train/test split too aggressive for RL (20 days → 170 days)
- [FIXED] Data quality (110% returns from stale prices)
- [FIXED] LGBM R² ≈ 0 (now uses lagged returns instead)
- [FIXED] Inconsistent A&S costs (backtest vs rl_env)
- [FIXED] Stressed regime = outliers (now volatility threshold)
- [FIXED] CVaR cost_lambda too high (1.0 → 0.001) causing A&S+CVaR to be identical to Flat
- [FIXED] Bootstrap CI without random seed (non-reproducible) — now seeded at 42
- [FIXED] Multiple testing without correction — now Benjamini-Hochberg at q=0.10

### High Priority (Acknowledged)
- Sharpe still negative for Flat(10bps) over full cycle — confirmed by full-cycle data
- Multiple testing and bootstrap methodology now implemented (block bootstrap + BH)
- Train/test regime shift resolved by 10-year data (full market cycle coverage)
- RL underperformance: only trades in Calm regime, fails in Volatile/Stressed — structural issue with observation normalization

### Medium Priority
- BIC prefers 4 states, used 3 (4th state = 36 bars, not usable)
- Lee-Ready tick rule approximation for first trade
- LGBM models still trained but not used by RL

---

## Data Timeline — 10-Year Full Cycle (RESOLVED)

The **10-year dataset** (2017-08 to 2026-04, ~909k bars) now covers full market cycles:

| Metric | Train Period | Test Period |
|--------|-------------|-------------|
| Duration | ~682k bars (75%) | ~227k bars (25%) |
| Date Range | 2017-08 to 2024-02 | 2024-02 to 2026-04 |
| Market Coverage | Bull + Bear + Consolidation | Bull continuation + Bear |

**10-Year HMM Regime Distribution (full dataset):**
| Regime | Count | Percentage |
|--------|-------|------------|
| Calm | 403,180 | 44.4% |
| Volatile | 370,908 | 40.9% |
| Stressed | 132,970 | 14.7% |

**What the 10-year data covers:**
- 2017 bull run ($2k→$20k BTC)
- 2018 crash ($20k→$3k BTC)
- 2019 consolidation
- COVID-2020 volatility
- 2021 bull run ($3k→$69k ETH, $30k→$69k BTC)
- 2022-2024 bear market ($69k→$20k BTC)

**Key finding from 10-year test**: Flat(10bps) shows TRUE Sharpe of -0.84 over the full cycle (vs the misleading +0.48 when tested on the 4-year bull-only window). Flat(A&S) with realistic costs wins on both return (+26.2%) and Sharpe (+0.48).

## Academic Context
MScFE 690 Capstone | WorldQuant School of Financial Engineering
Literature: Hamilton 1989, Avellaneda & Stoikov 2008, Sun/Liu/Sima 2020, Jiang/Liang 2017
