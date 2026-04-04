# RAPO-AS-RL — Regime-Aware Portfolio Optimization

## What It Is
An MScFE capstone research project building an RL-enhanced crypto trading system. The agent learns optimal BTC/ETH rebalancing conditioned on market regime and A&S-calibrated execution costs.

**Goal:** Validate that regime-conditional RL outperforms flat portfolio + Avellaneda-Stoikov baseline.

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
Layer 4: PPO Agent per Regime
  Input: portfolio state → target weights
  Output: ppo_Calm/Volatile/Stressed.zip
       │
       ▼
Three-way Backtest: Flat vs A&S+CVaR vs RL Agent
```

## Key Components

### Data
- Source: Binance public API (no auth needed)
- Symbols: BTC/USDT, ETH/USDT
- OHLCV: 5-min bars
- Trades: tick-level (Lee-Ready classification)
- Fallback: synthetic data if rate limited
- Train period: ~2021–2024 | Test period: 2024–2025

### Layer 1 — HMM
- Gaussian HMM, 3 states (Calm/Volatile/Stressed)
- Features: return, realized_vol, spread_proxy, OFI
- State selection: BIC (vs AIC)
- Multiple random initializations (5 seeds) to avoid local optima
- Output: `models/hmm/regime_labels.csv`, `hmm_model.pkl`

### Layer 2 — Avellaneda-Stoikov
- Calibrate per regime: σ (vol), s (spread), δ (depth), γ (risk-aversion)
- A&S cost = σ·√(q/(2δ))·P + s/2·P + γ·q²/(2δ)·P
- Lee-Ready tick rule for trade direction
- Post-hoc corrections for Stressed (sparse data)
- Output: `models/as_cost/as_cost_calm/Volatile/Stressed.pkl`

### Layer 3 — LightGBM
- Regime-conditional: separate model per asset per regime
- Features: return_lag_1/3/6, realized_vol, spread_proxy, OFI, cross-asset return, regime indicators
- Hyperparams: num_leaves=31, lr=0.05, n_estimators=500, early_stopping=50
- Output: `models/lgbm/lgbm_btc/eth_calm/Volatile/Stressed.pkl`

### Layer 4 — PPO (Stable Baselines3)
- Gymnasium custom env: 11-dim obs, 2-dim continuous action (target BTC/ETH weights)
- Reward: (portfolio_return - A&S_cost) / σ_portfolio
- 3 separate policies (one per regime)
- Hyperparameter grid: lr∈{3e-4,1e-4} × γ∈{0.99,0.95} × clip∈{0.1,0.2}
- Early stopping: rolling Sharpe stabilization ±0.05 over 50k steps
- Architecture: MLP [64,64] ReLU
- Output: `models/rl/ppo_calm/Volatile/Stressed.zip`

### Backtest
- Three-way comparison: Flat vs A&S+CVaR vs RL Agent
- Metrics: Sharpe, max drawdown, PnL, turnover
- Notebook: `05_backtest_analysis.ipynb`

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

# 6. Layer 4: PPO
python -m src.layer4_rl.rl_train --data data/processed --regime models/hmm/regime_labels.csv --as_cost models/as_cost --lgbm models/lgbm --output models/rl/

# 7. Backtest
# Open notebooks/05_backtest_analysis.ipynb
```

## File Structure
```
rapo-as-rl/
├── data/raw/                 # Raw Binance OHLCV + trades
├── data/processed/            # Feature matrices
├── models/hmm/               # HMM model + regime labels
├── models/as_cost/            # A&S calibration per regime
├── models/lgbm/              # LightGBM forecasters per regime
├── models/rl/                # PPO policies per regime
├── src/
│   ├── layer1_hmm/           # HMM train + evaluate
│   ├── layer2_as/             # A&S calibration
│   ├── layer3_lightgbm/      # LightGBM training
│   └── layer4_rl/             # Gym env + PPO training
├── scripts/                   # Data fetch, process, notebook generation
├── notebooks/                 # Analysis notebooks (00–05)
└── requirements.txt
```

## Dependencies
- torch, stable-baselines3, gymnasium
- lightgbm, hmmlearn, scikit-learn
- pandas, numpy, scipy
- ccxt (Binance data)
- jupyter, matplotlib, seaborn

## Known Issues / TODOs
- Synthetic data fallbacks used when Binance rate-limited
- Stressed regime has sparse data → post-hoc corrections applied
- Lee-Ready tick rule approximation for trade direction
- PPO eval uses same env as training (no separate validation env)

## Academic Context
MScFE 690 Capstone | WorldQuant School of Financial Engineering
Literature: Hamilton 1989, Avellaneda & Stoikov 2008, Sun/Liu/Sima 2020, Jiang/Liang 2017
