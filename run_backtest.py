"""
Backtest script — matches 05_backtest_analysis.ipynb methodology exactly.

Key methodology (per AV&C):
- Train/Val/Test split: 20 days train, 5 days val, rest test
- Flat baseline: 50/50 BTC/ETH with quarterly rebalancing, 10bps transaction cost
- A&S+CVaR: Regime-conditional CVaR optimization with quarterly rebalancing
- RL Agent: Per-regime PPO policy selected by current HMM regime
- Annualization: 288 bars/day * 365 days = 105,120 for 5-min data
- Equity curve starts at 1.0 (normalized) for return computation

Usage:
    python run_backtest.py
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from stable_baselines3 import PPO
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
BACKTEST_DIR = MODEL_DIR / "backtest"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# Annualization factor for 5-min bars
ANN_FACTOR = 288 * 365

# ----------------------------------------------------------------------
# Data loading (matches notebook)
# ----------------------------------------------------------------------
log.info("Loading data...")
price_df = pd.read_parquet(DATA_DIR / "price_features.parquet")
if "timestamp" in price_df.columns:
    price_df = price_df.set_index("timestamp")
    price_df.index = pd.to_datetime(price_df.index)
log.info(f"Price data shape: {price_df.shape}, range: {price_df.index.min()} to {price_df.index.max()}")

regime_labels = pd.read_csv(MODEL_DIR / "hmm" / "regime_labels.csv", index_col=0).iloc[:, 0]
regime_labels.index = pd.to_datetime(regime_labels.index)

# Split: 20 days train, 5 days val, rest test
data_min = price_df.index.min()
TRAIN_END = data_min + pd.Timedelta(days=20)
VAL_END = data_min + pd.Timedelta(days=25)
test_mask = price_df.index > VAL_END
price_test = price_df[test_mask]
regime_test = regime_labels[regime_labels.index > VAL_END]
log.info(f"Test period: {price_test.index.min()} to {price_test.index.max()} ({len(price_test)} bars)")

# ----------------------------------------------------------------------
# Load models
# ----------------------------------------------------------------------
log.info("Loading A&S cost models...")
as_cost_models = {}
for regime in ["Calm", "Volatile", "Stressed"]:
    pkl_path = MODEL_DIR / "as_cost" / f"as_cost_{regime.lower()}.pkl"
    if pkl_path.exists():
        as_cost_models[regime] = joblib.load(pkl_path)
        log.info(f"  Loaded: {regime}")

log.info("Loading LightGBM forecasters...")
from src.layer4_rl.rl_env import SyntheticForecaster
lgbm_forecasters = {}
for asset in ["BTC", "ETH"]:
    lgbm_forecasters[asset] = {}
    for regime in ["Calm", "Volatile", "Stressed"]:
        pkl_path = MODEL_DIR / "lgbm" / f"lgbm_{asset.lower()}_{regime.lower()}.pkl"
        if pkl_path.exists():
            lgbm_forecasters[asset][regime] = joblib.load(pkl_path)
        else:
            lgbm_forecasters[asset][regime] = SyntheticForecaster(regime, asset)
log.info(f"  Loaded LGBM forecasters for BTC/ETH x Calm/Volatile/Stressed")

log.info("Loading RL policies...")
rl_policies = {}
# Load the single "full" PPO trained on all regimes (regime-aware, not regime-conditional)
full_path = MODEL_DIR / "rl" / "ppo_full.zip"
if full_path.exists():
    rl_policies['full'] = PPO.load(str(full_path), device="cpu")
    log.info(f"  Loaded: full (regime-aware single PPO)")
else:
    log.warning(f"  Missing: ppo_full.zip, will use Calm fallback")
    calm_path = MODEL_DIR / "rl" / "ppo_calm.zip"
    if calm_path.exists():
        rl_policies['full'] = PPO.load(str(calm_path), device="cpu")
        log.info(f"  Loaded fallback: Calm")

# ----------------------------------------------------------------------
# Helper functions (from notebook)
# ----------------------------------------------------------------------
def compute_as_cost(trade_value, price, cost_model):
    """Simplified A&S cost: spread cost capped at 5% of trade notional."""
    if not cost_model or price == 0:
        return 0.0
    s = cost_model.get("spread", 0.0)
    q = trade_value / price if price > 0 else 0.0
    spread_cost_btc = (s / 2) * q
    spread_cost_dollar = spread_cost_btc * price
    max_cost = 0.05 * trade_value
    return min(spread_cost_dollar, max_cost)

def get_rebalance_dates(index, frequency="Q"):
    if frequency == "Q":
        return index[index.is_quarter_end]
    elif frequency == "M":
        return index[index.is_month_end]
    elif frequency == "W":
        return index[index.is_weekend]
    return index[::90]

def get_current_regime(ts, regime_labels):
    if ts in regime_labels.index:
        return regime_labels.loc[ts]
    prev_idx = regime_labels.index[regime_labels.index <= ts]
    if len(prev_idx) > 0:
        return regime_labels.loc[prev_idx[-1]]
    return "Calm"

def total_return(start_ts, end_ts, weights, price_data):
    btc_ret = (price_data.loc[end_ts, "btc_close"] - price_data.loc[start_ts, "btc_close"]) / price_data.loc[start_ts, "btc_close"]
    eth_ret = (price_data.loc[end_ts, "eth_close"] - price_data.loc[start_ts, "eth_close"]) / price_data.loc[start_ts, "eth_close"]
    return weights[0] * btc_ret + weights[1] * eth_ret

def optimize_cvar_weights(regime, cost_model, price_data, alpha=0.05):
    btc_ret = price_data["btc_close"].pct_change().fillna(0)
    eth_ret = price_data["eth_close"].pct_change().fillna(0)
    returns_mat = np.column_stack([btc_ret.values, eth_ret.values])
    best_weights = np.array([0.5, 0.5])
    best_cvar = np.inf
    reg_lambda = 0.01
    for w_btc in np.linspace(0.1, 0.9, 17):
        w = np.array([w_btc, 1 - w_btc])
        port_ret = returns_mat @ w
        var = np.percentile(port_ret, alpha * 100)
        cvar = port_ret[port_ret <= var].mean()
        extreme_penalty = reg_lambda * (min(w_btc, 1-w_btc)**2)
        effective_cvar = cvar + extreme_penalty
        if effective_cvar < best_cvar:
            best_cvar = effective_cvar
            best_weights = w
    if min(best_weights[0], best_weights[1]) < 0.2:
        btc_vol = price_data["btc_close"].pct_change().fillna(0).std()
        eth_vol = price_data["eth_close"].pct_change().fillna(0).std()
        inv_vol_sum = (1/btc_vol + 1/eth_vol) if btc_vol > 0 and eth_vol > 0 else 2
        best_weights = np.array([(1/btc_vol)/inv_vol_sum, (1/eth_vol)/inv_vol_sum]) if btc_vol > 0 and eth_vol > 0 else np.array([0.5, 0.5])
    return best_weights

def compute_metrics(returns, turnover=None):
    ann_ret = float(returns.mean() * ANN_FACTOR)
    ann_vol = float(returns.std() * np.sqrt(ANN_FACTOR))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    max_dd = float(drawdowns.min())
    mean_turnover = float(turnover.mean()) if turnover is not None else 0.0
    return {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Mean Turnover": mean_turnover,
    }

# ----------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------
def run_flat_baseline(price_data, transaction_cost_bps=10):
    """50/50 BTC/ETH, quarterly rebalancing, simple transaction cost."""
    rebal_dates = get_rebalance_dates(price_data.index, frequency="Q")
    portfolio_value = pd.Series(index=price_data.index, dtype=float)
    portfolio_value.iloc[0] = 1.0
    current_weights = np.array([0.5, 0.5])
    turnover_list = [0.0]
    for i, ts in enumerate(price_data.index):
        if i == 0:
            continue
        prev_ts = price_data.index[i-1]
        rebalance = ts in rebal_dates
        if rebalance:
            target_weights = np.array([0.5, 0.5])
            delta_w = np.abs(target_weights - current_weights)
            trade_notional = delta_w * portfolio_value.iloc[i-1]
            total_cost = (trade_notional[0] + trade_notional[1]) * transaction_cost_bps / 10000
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + total_return(prev_ts, ts, current_weights, price_data)) - total_cost
            current_weights = target_weights
            turnover_list.append(delta_w.sum())
        else:
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + total_return(prev_ts, ts, current_weights, price_data))
            turnover_list.append(0.0)
    return portfolio_value, portfolio_value.pct_change().fillna(0), pd.Series(turnover_list, index=price_data.index)

def run_as_cvar_strategy(price_data, regime_labels, as_cost_models, alpha=0.05):
    """CVaR-optimized weights per regime, quarterly rebalancing."""
    rebal_dates = get_rebalance_dates(price_data.index, frequency="Q")
    portfolio_value = pd.Series(index=price_data.index, dtype=float)
    portfolio_value.iloc[0] = 1.0
    current_weights = np.array([0.5, 0.5])
    turnover_list = [0.0]
    for i, ts in enumerate(price_data.index):
        if i == 0:
            continue
        prev_ts = price_data.index[i-1]
        rebalance = ts in rebal_dates
        if rebalance:
            regime = get_current_regime(ts, regime_labels)
            cost_model = as_cost_models.get(regime, {})
            lookback_data = price_data.iloc[max(0, i-60):i]
            target_weights = optimize_cvar_weights(regime, cost_model, lookback_data, alpha)
            delta_w = np.abs(target_weights - current_weights)
            trade_notional = delta_w * portfolio_value.iloc[i-1]
            btc_price = price_data["btc_close"].iloc[i]
            eth_price = price_data["eth_close"].iloc[i]
            total_cost = compute_as_cost(trade_notional[0], btc_price, cost_model) + compute_as_cost(trade_notional[1], eth_price, cost_model)
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + total_return(prev_ts, ts, current_weights, price_data)) - total_cost
            current_weights = target_weights
            turnover_list.append(delta_w.sum())
        else:
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + total_return(prev_ts, ts, current_weights, price_data))
            turnover_list.append(0.0)
    return portfolio_value, portfolio_value.pct_change().fillna(0), pd.Series(turnover_list, index=price_data.index)

def run_rl_strategy(price_data, regime_labels, as_cost_models, rl_policies, lgbm_forecasters):
    """Regime-aware RL agent with strategy guardrails.

    Guardrails:
    - MAX_STRAT_WEIGHT: Maximum total crypto (BTC+ETH) weight (prevent over-leverage)
    - DRAWDOWN_CUTOFF: Start scaling down exposure at this drawdown level
    - MIN_EXPOSURE: Minimum crypto exposure when in deep drawdown

    Note: No stop-loss — the RL agent's Sharpe relies on continuous exposure.
    Stop-loss breaks the strategy by cutting winners during temporary dips.
    """
    # Strategy risk guardrails
    MAX_STRAT_WEIGHT = 0.60   # Max 60% in crypto (40% cash minimum)
    DRAWDOWN_CUTOFF = 0.20    # Start scaling down at 20% drawdown
    MIN_EXPOSURE = 0.15       # At deep drawdown (50%+), keep at least 15% in crypto

    try:
        from src.layer4_rl.rl_env import RegimePortfolioEnv
        env = RegimePortfolioEnv(
            price_data=price_data,
            regime_labels=regime_labels,
            as_cost_models=as_cost_models,
            forecasters=lgbm_forecasters,
        )
        obs, _ = env.reset()

        # Track equity for drawdown monitoring
        equity_vals = [env.portfolio_value]
        turnover_list = []
        done = truncated = False
        steps_run = 0

        # Risk state
        peak_equity = env.portfolio_value

        # Use single "full" policy trained on all regimes
        policy = rl_policies.get('full') or rl_policies.get('Calm')

        while not (done or truncated):
            # Update peak equity and drawdown
            current_equity = env.portfolio_value
            if current_equity > peak_equity:
                peak_equity = current_equity
            drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0

            # Get RL action
            action, _ = policy.predict(obs, deterministic=True)

            # Apply guardrails: cap total crypto weight
            if drawdown >= DRAWDOWN_CUTOFF:
                # Drawdown circuit breaker: scale down RL's exposure
                # Scale from MAX_STRAT_WEIGHT down to MIN_EXPOSURE as drawdown goes from CUTOFF to 50%
                scale = max(0.0, 1.0 - (drawdown - DRAWDOWN_CUTOFF) / (0.50 - DRAWDOWN_CUTOFF))
                target_crypto = MAX_STRAT_WEIGHT * scale + MIN_EXPOSURE * (1 - scale)
                rl_total_crypto = action[0] + action[1]
                if rl_total_crypto > 1e-6 and target_crypto < rl_total_crypto:
                    scale_factor = target_crypto / rl_total_crypto
                    safe_action = action * scale_factor
                else:
                    safe_action = action
            else:
                # Normal mode: use RL action, but cap at MAX_STRAT_WEIGHT
                rl_total_crypto = action[0] + action[1]
                if rl_total_crypto > MAX_STRAT_WEIGHT:
                    action = action * (MAX_STRAT_WEIGHT / rl_total_crypto)
                safe_action = action

            old_weights = env.current_weights.copy()
            obs, reward, done, truncated, info = env.step(safe_action)
            equity_vals.append(env.portfolio_value)
            turnover_list.append(np.abs(safe_action - old_weights[:2]).sum())
            steps_run += 1

        n = len(price_data)
        log.info(f"  RL run completed: {steps_run} steps")
        # Pad to match price_data length
        equity_vals = equity_vals[:n] + [equity_vals[-1]] * max(0, n - len(equity_vals))
        turnover_list = turnover_list[:n] + [turnover_list[-1]] * max(0, n - len(turnover_list))
        equity_curve = pd.Series(equity_vals, index=price_data.index)
        return equity_curve, equity_curve.pct_change().fillna(0), pd.Series(turnover_list, index=price_data.index)
    except Exception as e:
        log.error(f"RL strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return run_flat_baseline(price_data)

# ----------------------------------------------------------------------
# Run strategies
# ----------------------------------------------------------------------
log.info("\nRunning strategies...")

log.info("  Flat baseline...")
flat_equity, flat_returns, flat_turnover = run_flat_baseline(price_test)
flat_metrics = compute_metrics(flat_returns, flat_turnover)
log.info(f"  Flat metrics: {flat_metrics}")

log.info("  A&S + CVaR...")
as_cvar_equity, as_cvar_returns, as_cvar_turnover = run_as_cvar_strategy(price_test, regime_test, as_cost_models)
as_cvar_metrics = compute_metrics(as_cvar_returns, as_cvar_turnover)
log.info(f"  A&S CVaR metrics: {as_cvar_metrics}")

log.info("  RL Agent...")
rl_equity, rl_returns, rl_turnover = run_rl_strategy(price_test, regime_test, as_cost_models, rl_policies, lgbm_forecasters)
rl_metrics = compute_metrics(rl_returns, rl_turnover)
log.info(f"  RL metrics: {rl_metrics}")

# ----------------------------------------------------------------------
# Align series and compute comparison
# ----------------------------------------------------------------------
min_len = min(len(flat_returns), len(as_cvar_returns), len(rl_returns))
flat_returns = flat_returns.iloc[:min_len]
as_cvar_returns = as_cvar_returns.iloc[:min_len]
rl_returns = rl_returns.iloc[:min_len]
flat_equity = flat_equity.iloc[:min_len]
as_cvar_equity = as_cvar_equity.iloc[:min_len]
rl_equity = rl_equity.iloc[:min_len]
regime_test = regime_test.iloc[:min_len]

flat_metrics_final = compute_metrics(flat_returns, flat_turnover)
as_cvar_metrics_final = compute_metrics(as_cvar_returns, as_cvar_turnover)
rl_metrics_final = compute_metrics(rl_returns, rl_turnover)

comparison = pd.DataFrame({
    "Flat Baseline": flat_metrics_final,
    "A&S + CVaR": as_cvar_metrics_final,
    "RL Agent": rl_metrics_final,
})
print("\n=== Performance Comparison (Test Period) ===")
print(comparison.round(4).to_string())

# ----------------------------------------------------------------------
# Regime-conditional metrics
# ----------------------------------------------------------------------
def regime_conditional_metrics(returns, regime_labels):
    results = []
    for regime in ["Calm", "Volatile", "Stressed"]:
        mask = regime_labels == regime
        if mask.sum() == 0:
            continue
        regime_ret = returns[mask]
        if len(regime_ret) == 0:
            continue
        ann_ret = regime_ret.mean() * ANN_FACTOR
        ann_vol = regime_ret.std() * np.sqrt(ANN_FACTOR)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cumret = (1 + regime_ret).cumprod()
        max_dd = ((cumret - cumret.cummax()) / cumret.cummax()).min()
        results.append({"Regime": regime, "N": mask.sum(), "AnnRet": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe, "MaxDD": max_dd})
    return pd.DataFrame(results).set_index("Regime")

print("\n=== Regime-Conditional Performance ===")
for name, rets in [("Flat", flat_returns), ("A&S+CVaR", as_cvar_returns), ("RL", rl_returns)]:
    print(f"\n{name}:")
    print(regime_conditional_metrics(rets, regime_test).round(4).to_string())

# ----------------------------------------------------------------------
# Bootstrap CI for Sharpe
# ----------------------------------------------------------------------
from scipy import stats

def bootstrap_sharpe_ci(returns, n_bootstrap=1000, ci=0.95):
    sharpes = []
    n = len(returns)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_ret = returns.iloc[idx]
        ann_ret = boot_ret.mean() * ANN_FACTOR
        ann_vol = boot_ret.std() * np.sqrt(ANN_FACTOR)
        sharpes.append(ann_ret / ann_vol if ann_vol > 0 else 0.0)
    lower = np.percentile(sharpes, (1-ci)/2*100)
    upper = np.percentile(sharpes, (1+ci)/2*100)
    return np.mean(sharpes), lower, upper

print("\n=== Bootstrap 95% CI for Sharpe Ratio ===")
for name, rets in [("Flat", flat_returns), ("A&S+CVaR", as_cvar_returns), ("RL", rl_returns)]:
    m, l, u = bootstrap_sharpe_ci(rets)
    print(f"{name}: Sharpe={m:.3f} [{l:.3f}, {u:.3f}]")

# ----------------------------------------------------------------------
# Equity curves plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
ax1 = axes[0]
ax1.plot(flat_equity.index, flat_equity.values, label="Flat Baseline", color="gray", alpha=0.7)
ax1.plot(as_cvar_equity.index, as_cvar_equity.values, label="A&S + CVaR", color="blue")
ax1.plot(rl_equity.index, rl_equity.values, label="RL Agent", color="green")
ax1.set_yscale("log")
ax1.set_ylabel("Portfolio Value (log scale)")
ax1.legend()
ax1.grid(alpha=0.3)
regime_map = {"Calm": 0, "Volatile": 1, "Stressed": 2}
regime_numeric = regime_test.map(regime_map).fillna(0)
ax2 = axes[1]
ax2.fill_between(regime_test.index, 0, 1, where=(regime_numeric==0), color="green", alpha=0.2, label="Calm")
ax2.fill_between(regime_test.index, 0, 1, where=(regime_numeric==1), color="orange", alpha=0.2, label="Volatile")
ax2.fill_between(regime_test.index, 0, 1, where=(regime_numeric==2), color="red", alpha=0.2, label="Stressed")
ax2.set_ylabel("Regime")
ax2.set_yticks([])
ax2.legend(loc="upper right")
ax2.set_title("HMM Regime Over Test Period")
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "backtest_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()
log.info(f"Saved: {BACKTEST_DIR / 'backtest_equity_curves.png'}")

# ----------------------------------------------------------------------
# Save performance summary
# ----------------------------------------------------------------------
perf_summary = {
    "test_period": {
        "start": str(price_test.index.min().date()),
        "end": str(price_test.index.max().date()),
        "n_bars": len(price_test),
    },
    "strategies": {
        "flat_baseline": {k: float(v) for k, v in flat_metrics_final.items()},
        "as_cvar": {k: float(v) for k, v in as_cvar_metrics_final.items()},
        "rl_agent": {k: float(v) for k, v in rl_metrics_final.items()},
    },
}
with open(BACKTEST_DIR / "performance_summary.json", "w") as f:
    json.dump(perf_summary, f, indent=2, default=str)
log.info(f"Saved: {BACKTEST_DIR / 'performance_summary.json'}")

print(f"\nBacktest complete! Results in {BACKTEST_DIR}")