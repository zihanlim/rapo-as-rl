# RAPO-AS-RL: Regime-Aware Portfolio Optimization with Avellaneda-Stoikov-Calibrated Execution Liquidity Costs via Reinforcement Learning

**Student:** Zihan Lim | **MScFE 690 Capstone** | WorldQuant School of Financial Engineering

---

## Overview

This capstone builds and validates an RL-enhanced regime-conditional portfolio optimization system in which an RL agent learns the optimal rebalancing policy conditioned on the prevailing market regime and A&S-calibrated dynamic execution costs, operating on BTC/ETH cryptocurrency markets via the Binance public API.

> **Cost Model Note:** Layer 2 uses the **Almgren-Chriss execution cost decomposition** (A&C, 2000) with parameters calibrated via the **Avellaneda-Stoikov** approach (A&S, 2008). The participation-rate market impact formula `η·σ·P·√(q/ADV)` follows the standard market microstructure literature (Gatheral, 2010; Tóth et al., 2011), NOT the original A&S market-maker formula. The A&S contribution is the **per-regime parameter calibration method** (σ, s per regime) applied to a standard execution cost formula. The 2026-04-19 fix replaces A&S equilibrium depth inference (`δ = 2/(s·P)`, ~2,685 bps error) with empirical volume-based participation-rate (`ADV` from Binance trades, ~10 bps calm). See `SPEC.md` and `LESSONS_LEARNED.md` Section 20 for details.

## Architecture

```
Binance API (CCXT)
       │
       ▼
Layer 1: HMM Regime Classifier (3-state: Calm / Volatile / Stressed)
       │
       ▼
Layer 2: A&S-Calibrated Almgren-Chriss Cost Model (slippage + market impact)
       │
       ▼
Layer 3: LightGBM Return Forecaster (BTC & ETH, regime-conditional)
       │
       ▼
Layer 4: Regime-Aware PPO Agent (SINGLE regime-aware model, Stable Baselines3)
       │
       ▼
Four-Way Backtest: Flat(10bps) vs Flat(A&S) vs A&S+CVaR vs RL Agent
```

## Project Structure

```
rapo-as-rl/
├── data/
│   ├── raw/              # Binance OHLCV and trade tick data
│   └── processed/         # Feature matrices, regime labels
├── models/
│   ├── hmm/              # Trained HMM model
│   ├── as_cost/          # Per-regime A&S calibrations
│   ├── lgbm/             # Trained LightGBM forecasters
│   └── rl/               # Trained PPO policy (single regime-aware model)
├── src/
│   ├── layer1_hmm/       # HMM regime classifier
│   ├── layer2_as/        # A&S cost model calibration
│   ├── layer3_lightgbm/  # LightGBM return forecaster
│   ├── layer4_rl/        # Gym environment + PPO training
│   ├── data_pipeline/    # CCXT data ingestion + features
│   ├── notebooks/        # Jupyter analysis notebooks
│   └── utils/            # Metrics, visualization, config
├── Makefile              # `make data`, `make train`, `make backtest`
└── requirements.txt       # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch Binance data (requires Binance free API — no authentication)
make data

# Train Layer 1: HMM Regime Classifier
make hmm

# Train Layer 2: Per-Regime A&S Cost Calibration
make as_calibrate

# Train Layer 3: LightGBM Return Forecaster
make lgbm

# Train Layer 4: RL Policy (PPO, single regime-aware model)
make rl_train

# Run four-way backtest
make backtest
```

## Key Results

**Test Period (2024-02 to 2026-04, ~227k bars):**

| Strategy | Ann. Return | Sharpe | Max DD | Turnover |
|----------|-------------|--------|--------|----------|
| **Flat(A&S)** | **+26.2%** | **+0.48** | -56.6% | ~0 |
| Flat(10bps) | +25.1% | +0.44 | -57.6% | ~0 |
| A&S+CVaR | +23.4% | +0.42 | -57.1% | 0.000006 |
| RL Agent | -3.6% | -0.68 | **-7.9%** | 0.000004 |

**Key Findings:**
- **Flat(A&S) wins** on both Ann. Return (+26.2%) and Sharpe (+0.47) vs Flat(10bps) — participation-rate costs show 60/40 BTC/ETH buy-and-hold is optimal at 5-min frequency
- **No strategy is statistically significantly better** on Sharpe ratio (block bootstrap 95% CI + Benjamini-Hochberg correction at q=0.10)
- **RL Agent confirmed genuine cash-convergence** (2026-04-19 retraining): Even after retraining on corrected ~10–52 bps participation-rate costs, the RL agent converges to near-cash (Sharpe -0.68). This proves the cash-convergence is a genuine microstructure finding — not a training artifact of the buggy cost model. The RL cannot reliably beat the 60/40 BTC/ETH benchmark with weak momentum and realistic participation-rate execution costs.
- **Participation-rate cost model corrected** (2026-04-19): A depth-based calibration bug (δ from A&S equilibrium `δ = 2/(s·P)`, ~2,685 bps) was replaced with the standard participation-rate form (Gatheral, 2010; Tóth et al., 2011) with η ∈ {0.20, 0.20, 0.55}, yielding ~10–52 bps validated against Makarov and Schoar (2020)
- **Core finding**: Per-regime participation-rate execution costs (~10 bps Calm to ~52 bps Stressed) plus the benchmark-relative reward make active rebalancing unviable for weak-momentum strategies in crypto markets at 5-min frequency.

**Statistical Testing (Bootstrap 95% CI for Sharpe, block size=288 bars):**
- Flat(10bps): [−0.84, +1.65]
- Flat(A&S): [−0.79, +1.68]
- A&S+CVaR: [−0.84, +1.61]
- RL: [−0.005, 0.000] (near-cash: Sharpe undefined, ~0.000)

## Literature

This project is documented in the accompanying MScFE 690 capstone thesis. The approach integrates four bodies of literature:

1. **Markov-switching models** for unsupervised regime detection (Hamilton 1989; Malekinezjad & Rafati 2026)
2. **Avellaneda-Stoikov** microstructure cost modeling (Avellaneda & Stoikov 2008)
3. **LightGBM** for regime-conditional return forecasting (Sun, Liu & Sima 2020)
4. **Deep reinforcement learning** for portfolio management (Jiang, Xu & Liang 2017; Jiang & Liang 2017)

## License

MIT License
