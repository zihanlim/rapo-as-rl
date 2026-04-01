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
Layer 4: RL Agent per Regime (PPO, Stable Baselines3)
       │
       ▼
Three-Way Backtest: Flat Baseline vs A&S+CVaR vs Full RL Agent
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
│   └── rl/               # Trained PPO policies (per regime)
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

# Train Layer 4: RL Policies (PPO, per regime)
make rl_train

# Run three-way backtest
make backtest
```

## Key Results

*(Results will be populated as training and backtesting complete)*

## Literature

This project is documented in the accompanying MScFE 690 capstone thesis. The approach integrates four bodies of literature:

1. **Markov-switching models** for unsupervised regime detection (Hamilton 1989; Malekinezjad & Rafati 2026)
2. **Avellaneda-Stoikov** microstructure cost modeling (Avellaneda & Stoikov 2008)
3. **LightGBM** for regime-conditional return forecasting (Sun, Liu & Sima 2020)
4. **Deep reinforcement learning** for portfolio management (Jiang, Xu & Liang 2017; Jiang & Liang 2017)

## License

MIT License
