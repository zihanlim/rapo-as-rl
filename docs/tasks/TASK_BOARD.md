# rapo-as-rl Task Board

## Status Legend
- `[ ]` = not started
- `[~]` = in progress
- `[✅]` = done
- `[❌]` = blocked/error

## Phase 0: Data Infrastructure
- [✅] Fetch Binance OHLCV + trade tick data
- [✅] Process features (OHLCV bars, Lee-Ready classification, OFI, spread_proxy)
- [✅] Synthetic data fallback when rate-limited

## Phase 1: Layer 1 — HMM Regime Classifier
- [✅] Feature engineering (return, realized_vol, spread_proxy, OFI)
- [✅] BIC/AIC state selection
- [✅] Multi-seed training (5 seeds)
- [✅] State labeling (Calm/Volatile/Stressed by vol rank)

## Phase 2: Layer 2 — A&S Cost Calibration
- [✅] Lee-Ready tick rule for trade direction
- [✅] Spread / vol / depth estimation per regime
- [✅] Post-hoc Stressed regime corrections
- [✅] Per-regime cost model persistence (.pkl)

## Phase 3: Layer 3 — LightGBM Forecaster
- [✅] Feature matrix (lags, vol, OFI, cross-asset, regime indicators)
- [✅] Regime-conditional training (BTC + ETH × 3 regimes)
- [✅] Chronological validation split
- [✅] Early stopping (50 rounds)
- [✅] Per-regime model persistence

## Phase 4: Layer 4 — PPO Agent
- [✅] Gymnasium environment (RegimePortfolioEnv)
- [✅] Single regime-aware PPO policy (replaced per-regime approach due to insufficient training data per regime)
- [✅] Ultra-conservative hyperparameters ([32,32] net, lr=3e-5, clip=0.1)
- [✅] Policy persistence (ppo_full.zip)

## Phase 5: Backtest & Analysis
- [✅] Complete `05_backtest_analysis.ipynb` — four-way comparison
- [✅] Metrics: Sharpe, max drawdown, PnL, turnover, regime exposure
- [✅] Statistical significance testing (block bootstrap + Benjamini-Hochberg)
- [✅] Regime-conditional performance breakdown
- [~] Transaction cost sensitivity analysis (partially covered)
- [✅] Backtest results: Flat(A&S) wins on Sharpe (+0.48) and Return (+26.2%)

## Phase 6: Thesis / Writeup
- [ ] Literature review integration
- [ ] Results tables and figures
- [ ] Citation formatting
- [ ] Abstract + introduction

*Last updated: 2026-04-12*
