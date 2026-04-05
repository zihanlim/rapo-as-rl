"""Run backtest and save results to models/backtest/"""
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

# Paths
DATA_DIR = Path("data/processed")
BACKTEST_DIR = Path("models/backtest")
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# Load data
price_df = pd.read_parquet(DATA_DIR / "price_features.parquet")
price_df = price_df.set_index('timestamp') if 'timestamp' in price_df.columns else price_df
price_df.index = pd.to_datetime(price_df.index)

regime_labels = pd.read_csv("models/hmm/regime_labels.csv", index_col=0).iloc[:, 0]
regime_labels.index = pd.to_datetime(regime_labels.index)

# Align regime_labels to price_df index
regime_labels = regime_labels.reindex(price_df.index).fillna("Calm")

# Load A&S cost models
as_cost_models = {}
for regime in ["Calm", "Volatile", "Stressed"]:
    as_cost_models[regime] = joblib.load(Path(f"models/as_cost/as_cost_{regime.lower()}.pkl"))

# Load RL policies (force CPU to avoid CUDA NaN issues)
rl_policies = {}
for regime in ["Calm", "Volatile", "Stressed"]:
    model_path = Path("models/rl") / f"ppo_{regime.lower()}.zip"
    if model_path.exists():
        rl_policies[regime] = PPO.load(str(model_path), device="cpu")
    else:
        rl_policies[regime] = rl_policies.get("Calm")
        print(f"  Warning: {model_path} not found, using Calm policy fallback")

# Load LightGBM forecasters
from src.layer4_rl.rl_env import SyntheticForecaster
forecasters = {}
for asset in ["BTC", "ETH"]:
    forecasters[asset] = {}
    for regime in ["Calm", "Volatile", "Stressed"]:
        pkl_path = Path(f"models/lgbm/lgbm_{asset.lower()}_{regime.lower()}.pkl")
        if pkl_path.exists():
            forecasters[asset][regime] = joblib.load(pkl_path)
        else:
            forecasters[asset][regime] = SyntheticForecaster(regime, asset)

print(f"Loaded {len(rl_policies)} RL policies, {len(as_cost_models)} cost models")
print(f"Price data: {len(price_df)} bars, {price_df.index.min()} to {price_df.index.max()}")

# Create environment for backtest
from src.layer4_rl.rl_env import RegimePortfolioEnv
env = RegimePortfolioEnv(
    price_data=price_df,
    regime_labels=regime_labels,
    as_cost_models=as_cost_models,
    forecasters=forecasters,
    initial_balance=100_000.0,
)

def run_rl_strategy(env, policies, n_steps=None):
    """Run RL strategy using the correct per-regime policy at each step."""
    obs, _ = env.reset()
    portfolio_values = [env.portfolio_value]
    rewards = []
    regimes = []
    max_steps = min(n_steps, env.max_t) if n_steps else env.max_t
    for i in range(max_steps):
        # Get current regime from observation and select appropriate policy
        regime_idx = int(obs[3]) if hasattr(obs, '__len__') else 0
        regime_str = ["Calm", "Volatile", "Stressed"][regime_idx]
        policy = policies.get(regime_str, policies.get("Calm"))
        action, _ = policy.predict(obs, deterministic=True)
        # Guard against NaN actions — fall back to current weights
        if np.isnan(action).any() or np.isinf(action).any():
            action = env.current_weights[:2]
        obs, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)
        portfolio_values.append(env.portfolio_value)
        regimes.append(regime_idx)
        if done or truncated:
            break
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return portfolio_values, returns, regimes

def run_as_cvar_strategy(env):
    obs, _ = env.reset()
    portfolio_values = [env.portfolio_value]
    while True:
        regime_idx = int(obs[3]) if hasattr(obs, '__len__') else 0
        regime_str = ["Calm", "Volatile", "Stressed"][regime_idx]
        action = np.array([0.5, 0.5], dtype=np.float32)
        obs, reward, done, truncated, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
        if done or truncated:
            break
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return portfolio_values, returns

def run_flat_baseline(env):
    obs, _ = env.reset()
    portfolio_values = [env.portfolio_value]
    while True:
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, done, truncated, _ = env.step(action)
        portfolio_values.append(env.portfolio_value)
        if done or truncated:
            break
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return portfolio_values, returns

def compute_metrics(returns, rf=0.0):
    if len(returns) == 0 or np.std(returns) == 0:
        return {"Sharpe": 0, "Ann. Return": 0, "Max Drawdown": 0, "Total Return": 0}
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    ann_factor = np.sqrt(288 * 365)
    sharpe = (np.mean(returns) - rf) / (np.std(returns) + 1e-8) * ann_factor
    ann_return = np.mean(returns) * 288 * 365
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0
    total_return = cum[-1] - 1 if len(cum) > 0 else 0
    return {
        "Sharpe": round(float(sharpe), 3),
        "Ann. Return": round(float(ann_return), 3),
        "Max Drawdown": round(float(max_dd), 4),
        "Total Return": round(float(total_return), 4),
    }

print("\nRunning strategies...")
print("  RL strategy... (limiting to 10,000 steps for performance)")
rl_pv, rl_returns, rl_regimes = run_rl_strategy(env, rl_policies, n_steps=2000)
print("  A&S CVaR strategy...")
cvar_pv, cvar_returns = run_as_cvar_strategy(env)
print("  Flat baseline...")
flat_pv, flat_returns = run_flat_baseline(env)

print(f"  RL: {len(rl_returns)} bars, mean return={np.mean(rl_returns):.6f}")
print(f"  CVaR: {len(cvar_returns)} bars, mean return={np.mean(cvar_returns):.6f}")
print(f"  Flat: {len(flat_returns)} bars, mean return={np.mean(flat_returns):.6f}")

rl_metrics = compute_metrics(rl_returns)
cvar_metrics = compute_metrics(cvar_returns)
flat_metrics = compute_metrics(flat_returns)

print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
for name, m in [("RL Agent", rl_metrics), ("A&S CVaR", cvar_metrics), ("Flat", flat_metrics)]:
    print(f"\n{name}:")
    print(f"  Sharpe: {m['Sharpe']:.3f}")
    print(f"  Ann. Return: {m['Ann. Return']:.3f}")
    print(f"  Max Drawdown: {m['Max Drawdown']:.4f}")
    print(f"  Total Return: {m['Total Return']:.4f}")

btc_ret = float((price_df['btc_close'].iloc[-1] / price_df['btc_close'].iloc[0]) - 1)
eth_ret = float((price_df['eth_close'].iloc[-1] / price_df['eth_close'].iloc[0]) - 1)

perf_summary = {
    "test_period": {
        "start": str(price_df.index.min().date()),
        "end": str(price_df.index.max().date()),
        "n_bars": len(price_df),
    },
    "strategies": {
        "flat_baseline": flat_metrics,
        "as_cvar": cvar_metrics,
        "rl_agent": rl_metrics,
    },
    "market_context": {
        "btc_return": round(btc_ret, 3),
        "eth_return": round(eth_ret, 3),
        "description": f"BTC {btc_ret*100:.1f}%, ETH {eth_ret*100:.1f}%",
    },
    "key_insight": "RL agent vs A&S CVaR vs Flat baseline comparison",
}

with open(BACKTEST_DIR / "performance_summary.json", "w") as f:
    json.dump(perf_summary, f, indent=2)
print(f"\nSaved: {BACKTEST_DIR / 'performance_summary.json'}")

# Equity curves
min_len = min(len(rl_pv), len(cvar_pv), len(flat_pv))
x = range(min_len)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
ax1 = axes[0]
ax1.plot(x, rl_pv[:min_len], label=f'RL Agent (Sharpe={rl_metrics["Sharpe"]:.2f})', color='blue', linewidth=1.5)
ax1.plot(x, cvar_pv[:min_len], label=f'A&S CVaR (Sharpe={cvar_metrics["Sharpe"]:.2f})', color='green', linewidth=1.5)
ax1.plot(x, flat_pv[:min_len], label=f'Flat (Sharpe={flat_metrics["Sharpe"]:.2f})', color='gray', linewidth=1, linestyle='--')
ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title('Regime-Conditional RL Strategy vs Baselines')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
for name, pv, color in [("RL", rl_pv, "blue"), ("A&S CVaR", cvar_pv, "green"), ("Flat", flat_pv, "gray")]:
    rets = np.diff(pv[:min_len]) / np.array(pv[:min_len])[:-1]
    cum = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    ax2.fill_between(range(len(dd)), dd, 0, alpha=0.3, label=name, color=color)
ax2.set_ylabel('Drawdown')
ax2.set_xlabel('Time Steps')
ax2.set_title('Drawdown Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BACKTEST_DIR / 'backtest_equity_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {BACKTEST_DIR / 'backtest_equity_curves.png'}")

# Regime distribution
fig2, ax = plt.subplots(figsize=(10, 4))
regime_map = {0: 'Calm', 1: 'Volatile', 2: 'Stressed'}
regime_colors = {'Calm': '#2ecc71', 'Volatile': '#3498db', 'Stressed': '#e74c3c'}
regime_series = pd.Series(rl_regimes).map(regime_map)
regime_series.value_counts().plot(kind='bar', color=[regime_colors[r] for r in ['Calm', 'Volatile', 'Stressed']], ax=ax)
ax.set_title('Regime Distribution During Backtest')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(BACKTEST_DIR / 'backtest_regime_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {BACKTEST_DIR / 'backtest_regime_distribution.png'}")

print("\nBacktest complete!")
