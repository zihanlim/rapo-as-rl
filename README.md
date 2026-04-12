# RAPO-AS-RL: Regime-Aware Portfolio Optimization with Avellaneda-Stoikov-Calibrated Dynamic Liquidity Costs via Reinforcement Learning

**Student:** Zihan Lim | **MScFE 690 Capstone** | WorldQuant School of Financial Engineering

---

## Overview

This capstone builds and validates an RL-enhanced regime-conditional portfolio optimization system in which an RL agent learns the optimal rebalancing policy conditioned on the prevailing market regime and A&S-calibrated dynamic execution costs, operating on BTC/ETH cryptocurrency markets via the Binance public API.

## Architecture

```
Binance API (CCXT)
       │
       ▼
Layer 1: HMM Regime Classifier (3-state: Calm / Volatile / Stressed)
       │
       ▼
Layer 2: Avellaneda-Stoikov Per-Regime Cost Model (slippage + market impact)
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
- **Flat(A&S) wins** on both Ann. Return (+26.2%) and Sharpe (+0.48) vs Flat(10bps) — realistic costs show 60/40 BTC/ETH buy-and-hold is optimal
- **No strategy is statistically significantly better** on Sharpe ratio (block bootstrap 95% CI + Benjamini-Hochberg correction at q=0.10)
- **RL Agent has best Max Drawdown** (-7.9%) but negative Sharpe (-0.68) — learned cash is optimal under true execution costs at 5-min frequency
- **Daily-frequency RL experiment**: Training with decision_interval=288 (daily decisions) performed WORSE (Sharpe -3.88, MaxDD -39.1%) than 5-min RL (Sharpe -0.68). Only ~789 effective decisions during training vs 100k at 5-min. Both converge to cash-optimal. Reducing decision frequency does NOT solve the A&S cost problem.
- **Core finding**: Execution costs dominate active rebalancing returns. A single 50bps trade costs ~123 bps in Calm regime — 2.5x nominal. CVaR optimizer with realistic costs correctly stays near passive holding.

**Statistical Testing (Bootstrap 95% CI for Sharpe, block size=288 bars):**
- Flat(10bps): [−0.84, +1.65]
- Flat(A&S): [−0.79, +1.68]
- A&S+CVaR: [−0.84, +1.61]
- RL: [0.000, 0.000] (insufficient rebalancing for bootstrap)

## Literature

This project is documented in the accompanying MScFE 690 capstone thesis. The approach integrates four bodies of literature:

1. **Markov-switching models** for unsupervised regime detection (Hamilton 1989; Malekinezjad & Rafati 2026)
2. **Avellaneda-Stoikov** microstructure cost modeling (Avellaneda & Stoikov 2008)
3. **LightGBM** for regime-conditional return forecasting (Sun, Liu & Sima 2020)
4. **Deep reinforcement learning** for portfolio management (Jiang, Xu & Liang 2017; Jiang & Liang 2017)

## License

MIT License
