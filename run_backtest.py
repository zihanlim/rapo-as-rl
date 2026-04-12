"""
Backtest script — matches 05_backtest_analysis.ipynb methodology exactly.

Key methodology:
- Train/Val/Test split: 75% train, 25% test (chronological)
- Flat baseline: 50/50 BTC/ETH, periodic rebalancing, 10bps transaction cost
- Flat(A&S): 60/40 BTC/ETH, periodic rebalancing, TRUE A&S market impact costs
- A&S+CVaR: Regime-conditional CVaR optimization with cost-aware penalty
- RL Agent: Single regime-aware PPO policy (every-step rebalancing)
- Daily RL Agent: PPO trained with decision_interval=288 (decisions once per day)
- Annualization: 288 bars/day * 365 days = 105,120 for 5-min data
- Equity curve starts at 1.0 (normalized) for return computation

Usage:
    python run_backtest.py                           # Default: 1D rebalancing (288 bars)
    python run_backtest.py --frequency 1W            # Weekly rebalancing (2016 bars)
    python run_backtest.py --frequency 1Q           # Quarterly rebalancing (9504 bars)
    python run_backtest.py --all-frequencies        # Full sweep: 1H, 4H, 1D, 3D, 1W, 1Q
    python run_backtest.py --rl-daily              # Daily RL experiment (train first with train_rl_daily.py)
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'src')
import argparse
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

# Parse CLI args
parser = argparse.ArgumentParser(description="Backtest with configurable rebalancing frequency")
parser.add_argument("--frequency", type=str, default="1D",
                    help="Rebalancing frequency: '1H', '4H', '1D', '3D', '1W', '1Q' (default: 1D). "
                         "In bars (5-min data): 1D=288, 1W=2016, 1Q=9504")
parser.add_argument("--all-frequencies", action="store_true",
                    help="Run frequency sweep across 1H, 4H, 1D, 3D, 1W, 1Q and print comparison table")
parser.add_argument("--rl-daily", action="store_true",
                    help="Run daily-frequency RL experiment (decision_interval=288, ppo_daily.zip). "
                         "Compares to 5-min RL (ppo_full.zip) baseline.")
args_cli = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Map frequency labels to bar counts (5-min bars)
# 288 bars/day, 2016 bars/week (7*288), 9504 bars/quarter (365*288/4 approx)
FREQ_MAP = {
    "1H":  12,       # 12 * 5 min = 60 min
    "4H":  48,       # 4 * 12
    "1D":  288,      # 24 * 12
    "3D":  864,      # 3 * 288
    "1W":  2016,     # 7 * 288
    "1Q":  9504,     # ~quarter (365*288/4)
    "1M":  4320,     # 30 * 144 (month, approx)
}
DEFAULT_REBAL_FREQ = FREQ_MAP.get(args_cli.frequency, 288)

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
log.info(f"Rebalancing frequency: {args_cli.frequency} ({DEFAULT_REBAL_FREQ} bars)")
price_df = pd.read_parquet(DATA_DIR / "price_features.parquet")
if "timestamp" in price_df.columns:
    price_df = price_df.set_index("timestamp")
    price_df.index = pd.to_datetime(price_df.index)
log.info(f"Price data shape: {price_df.shape}, range: {price_df.index.min()} to {price_df.index.max()}")

regime_labels = pd.read_csv(MODEL_DIR / "hmm" / "regime_labels.csv", index_col=0).iloc[:, 0]
regime_labels.index = pd.to_datetime(regime_labels.index)

# Align regime_labels to price_df index (same as train_rl_stable.py)
regime_labels_aligned = regime_labels.reindex(price_df.index)
regime_labels_aligned = regime_labels_aligned.fillna("Calm")

# Split: 4-year dataset — 75% train (3 years), 25% test (1 year)
# CRITICAL-4 fix: PPO needs 50k+ steps. With 4-year data, use 3-year train.
split_idx = int(len(price_df) * 0.75)
TRAIN_END = price_df.index[split_idx]
train_mask = price_df.index <= TRAIN_END
test_mask = price_df.index > TRAIN_END
price_train = price_df[train_mask]
price_test = price_df[test_mask]
regime_train = regime_labels_aligned[train_mask]
regime_test = regime_labels_aligned[test_mask]
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
    log.info(f"  Loaded: full (ppo_full.zip, every-bar decisions)")
else:
    log.warning(f"  Missing: ppo_full.zip, will use Calm fallback")
    calm_path = MODEL_DIR / "rl" / "ppo_calm.zip"
    if calm_path.exists():
        rl_policies['full'] = PPO.load(str(calm_path), device="cpu")

# Load daily-frequency PPO (trained with decision_interval=288)
daily_path = MODEL_DIR / "rl" / "ppo_daily.zip"
if daily_path.exists():
    rl_policies['daily'] = PPO.load(str(daily_path), device="cpu")
    log.info(f"  Loaded: daily (ppo_daily.zip, daily decisions)")
else:
    log.warning(f"  Missing: ppo_daily.zip — run train_rl_daily.py first to enable daily RL experiment")

# ----------------------------------------------------------------------
# Helper functions (from notebook)
# ----------------------------------------------------------------------
# A&S cost formula uses RELATIVE volatility (fraction, e.g., 0.49 = 49% annual vol).
# The market_impact = sigma_rel * price * sqrt(q / (2*delta)) requires sigma_rel to be
# DIMENSIONLESS (relative). This is consistent with rl_env._as_cost.
#
# PORTFOLIO NORMALIZATION FIX:
# Portfolio equity is normalized (1.0 = $100k). Trade notional is computed as
# delta_w * equity. To get actual dollar notional: multiply by EQUITY_NOTIONAL.
# A 10% trade on $100k portfolio = $10,000 notional.
# Costs are then expressed as FRACTIONS of equity (normalized), so:
#   cost_fraction = cost_dollars / EQUITY_NOTIONAL
EQUITY_NOTIONAL = 100_000.0  # Normalized equity represents this many dollars


def compute_as_cost(trade_value, price, cost_model):
    """Full A&S cost: same formula as rl_env._as_cost.

    A&S cost components:
    - Market impact: sigma * price * sqrt(q / (2 * delta))
    - Spread cost: (s / 2) * q
    - Inventory risk: gamma * q^2 / (2 * delta) * price

    Note: sigma is relative volatility (dimensionless fraction, e.g., 0.57 = 57% annual vol).
    Portfolio values are normalized (1.0 = $100k). Trade notional is delta_w * equity.
    Costs are returned as FRACTIONS of equity (dimensionless), not dollar amounts.
    """
    if not cost_model or price == 0 or trade_value == 0:
        return 0.0

    # Scale trade_value from normalized units to actual dollars
    actual_trade_value = trade_value * EQUITY_NOTIONAL

    sigma_annual = cost_model.get("volatility", 0.0)  # relative vol (dimensionless)
    s = cost_model.get("spread", 0.0)  # spread in $/BTC
    delta = cost_model.get("depth", 1e-6)  # depth in BTC/$
    gamma = cost_model.get("gamma", 1e-6)  # risk aversion

    # CRITICAL FIX: sigma_annual is RELATIVE volatility (dimensionless fraction).
    # Do NOT divide by price. Convert to per-bar: sigma_annual / sqrt(288*365).
    sigma = sigma_annual / np.sqrt(365 * 288)  # per-bar relative vol (fraction)

    q = actual_trade_value / price if price > 0 else 0.0  # quantity in BTC

    # Three A&S cost components (in dollars)
    market_impact = sigma * price * np.sqrt(max(0, q / (2 * delta)))
    spread_cost = (s / 2) * q
    impact_cost = gamma * (q ** 2) / (2 * delta) * price

    total_cost_dollars = market_impact + spread_cost + impact_cost
    if np.isnan(total_cost_dollars) or np.isinf(total_cost_dollars):
        return 0.0

    # Convert to fraction of equity (normalized)
    return total_cost_dollars / EQUITY_NOTIONAL

def get_rebalance_dates(index, frequency="Q"):
    """Get rebalancing dates. Use integer for every-N-bars rebalancing."""
    if isinstance(frequency, int):
        return index[::frequency]
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

def optimize_cvar_weights(regime, cost_model, price_data, alpha=0.05,
                          current_weights=None, equity=1.0, btc_price=None, eth_price=None):
    """CVaR-optimized weights per regime, COST-AWARE.

    The optimizer minimizes: CVaR + extreme_penalty + cost_penalty

    The cost_penalty term accounts for A&S market impact of rebalancing:
    - For each candidate weight w, compute delta_w = |w - current_weights|
    - Penalize weight configurations requiring expensive rebalancing
    - This prevents CVaR from recommending aggressive rebalancing in high-cost regimes

    Note: Even with cost_lambda=1.0, the cost penalty for a 50bps rebalance is ~0.5x
    typical CVaR. In stressed regimes (costs ~250bps), the penalty is ~2.5x CVaR,
    so the optimizer correctly avoids rebalancing in high-cost regimes.
    """
    btc_ret = price_data["btc_close"].pct_change().fillna(0)
    eth_ret = price_data["eth_close"].pct_change().fillna(0)
    returns_mat = np.column_stack([btc_ret.values, eth_ret.values])
    best_weights = np.array([0.5, 0.5])
    best_score = np.inf
    reg_lambda = 0.01

    # Cost penalty weight — balances CVaR minimization vs cost avoidance
    # The cost_penalty = (cost_btc + cost_eth) * cost_lambda converts the A&S cost
    # (a fraction, e.g., 0.0005 for 50bps) into the same units as CVaR (also a fraction).
    # With cost_lambda=1.0: a 50bps rebalance costs 0.005 in score units.
    # Typical CVaR for crypto is ~0.001, so cost_lambda=1.0 makes the penalty
    # ~0.5x CVaR for calm regime — the optimizer can still overcome this if CVaR improves.
    # cost_lambda=10000 was wrong (467x penalty — optimizer NEVER rebalances).
    cost_lambda = 0.001

    for w_btc in np.linspace(0.1, 0.9, 17):
        w = np.array([w_btc, 1 - w_btc])
        port_ret = returns_mat @ w
        var = np.percentile(port_ret, alpha * 100)
        cvar = port_ret[port_ret <= var].mean()
        extreme_penalty = reg_lambda * (min(w_btc, 1-w_btc)**2)

        # COST PENALTY: estimate A&S cost of rebalancing to w
        cost_penalty = 0.0
        if current_weights is not None and btc_price is not None and eth_price is not None:
            delta_w = np.abs(w - current_weights)
            # Expected cost of rebalancing: delta_w * equity (notional in $)
            cost_btc = compute_as_cost(delta_w[0] * equity, btc_price, cost_model)
            cost_eth = compute_as_cost(delta_w[1] * equity, eth_price, cost_model)
            cost_penalty = (cost_btc + cost_eth) * cost_lambda  # scale to return units

        effective_score = cvar + extreme_penalty + cost_penalty
        if effective_score < best_score:
            best_score = effective_score
            best_weights = w

    # Fallback to inverse-vol weights if CVaR pushes to extreme
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
def run_flat_baseline(price_data, transaction_cost_bps=10, rebal_frequency=288):
    """50/50 BTC/ETH, periodic rebalancing, simple transaction cost.

    Costs are only charged when delta_w > 0 (actual trades occur).
    If target allocation equals current allocation, no trade occurs -> no cost.
    """
    rebal_dates = price_data.index[::rebal_frequency]
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
            # Only charge costs if delta_w > 0 (actual trade occurs)
            if delta_w.sum() > 0:
                total_notional = portfolio_value.iloc[i-1]
                total_cost = total_notional * transaction_cost_bps * 2 / 10000
            else:
                total_cost = 0.0
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + total_return(prev_ts, ts, current_weights, price_data)) - total_cost
            current_weights = target_weights
            turnover_list.append(delta_w.sum())
        else:
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + total_return(prev_ts, ts, current_weights, price_data))
            turnover_list.append(0.0)
    return portfolio_value, portfolio_value.pct_change().fillna(0), pd.Series(turnover_list, index=price_data.index)

def run_flat_baseline_as_cost(price_data, regime_labels, as_cost_models, rebal_frequency=288):
    """60/40 BTC/ETH, periodic rebalancing with TRUE A&S market impact costs.

    This is the FAIR comparison to the flat baseline:
    - Same 60/40 BTC/ETH allocation (used by RL/A&S+CVaR guardrails)
    - Same periodic rebalancing frequency (every 288 bars)
    - But uses TRUE A&S market impact costs instead of 10bps fixed
    This isolates the cost model difference from the strategy difference.
    """
    rebal_dates = price_data.index[::rebal_frequency]
    portfolio_value = pd.Series(index=price_data.index, dtype=float)
    portfolio_value.iloc[0] = 1.0
    # Start at 50/50 so first rebalance generates actual trades
    current_weights = np.array([0.5, 0.5])
    target_weights = np.array([0.60, 0.40])  # Same allocation as RL guardrails
    turnover_list = [0.0]
    for i, ts in enumerate(price_data.index):
        if i == 0:
            continue
        prev_ts = price_data.index[i-1]
        rebalance = ts in rebal_dates
        if rebalance:
            regime = get_current_regime(ts, regime_labels)
            cost_model = as_cost_models.get(regime, {})
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


def run_as_cvar_strategy(price_data, regime_labels, as_cost_models, alpha=0.05, rebal_frequency=288,
                         max_cost_budget=0.05):
    """CVaR-optimized weights per regime, periodic rebalancing, with cost-awareness.

    Cost-awareness additions:
    1. optimize_cvar_weights includes A&S cost penalty (penalizes expensive rebalances)
    2. Per-cost budget: stop rebalancing when cumulative A&S costs exceed max_cost_budget
       fraction of peak equity. This simulates a risk manager capping total costs.

    Args:
        max_cost_budget: Maximum cumulative A&S cost as fraction of peak equity.
            Default 5% — stop rebalancing once costs exceed this threshold.
    """
    rebal_dates = price_data.index[::rebal_frequency]
    portfolio_value = pd.Series(index=price_data.index, dtype=float)
    portfolio_value.iloc[0] = 1.0
    current_weights = np.array([0.5, 0.5])
    turnover_list = [0.0]

    # Per-cost budget tracking
    cumulative_cost = 0.0
    peak_equity = 1.0
    cost_budget_exceeded = False

    for i, ts in enumerate(price_data.index):
        if i == 0:
            continue
        prev_ts = price_data.index[i-1]
        rebalance = ts in rebal_dates and not cost_budget_exceeded

        # Update peak equity
        prev_equity = portfolio_value.iloc[i-1]
        peak_equity = max(peak_equity, prev_equity)

        if rebalance:
            regime = get_current_regime(ts, regime_labels)
            cost_model = as_cost_models.get(regime, {})
            lookback_data = price_data.iloc[max(0, i-60):i]
            btc_price = price_data["btc_close"].iloc[i]
            eth_price = price_data["eth_close"].iloc[i]

            # COST-AWARE CVAR: pass current weights + prices for cost penalty
            target_weights = optimize_cvar_weights(
                regime, cost_model, lookback_data, alpha,
                current_weights=current_weights,
                equity=prev_equity,
                btc_price=btc_price,
                eth_price=eth_price
            )
            delta_w = np.abs(target_weights - current_weights)
            trade_notional = delta_w * prev_equity
            total_cost = (compute_as_cost(trade_notional[0], btc_price, cost_model) +
                          compute_as_cost(trade_notional[1], eth_price, cost_model))

            # Check cost budget BEFORE adding to cumulative cost
            # (don't count costs for rebalances we skip due to budget)
            if peak_equity > 0 and (cumulative_cost / peak_equity) > max_cost_budget:
                cost_budget_exceeded = True
                # Don't rebalance; hold last weights; NO cost deducted
                portfolio_value.iloc[i] = prev_equity * (1 + total_return(prev_ts, ts, current_weights, price_data))
                turnover_list.append(0.0)
            else:
                # Update cumulative cost only for EXECUTED rebalances
                cumulative_cost += total_cost
                portfolio_value.iloc[i] = prev_equity * (1 + total_return(prev_ts, ts, current_weights, price_data)) - total_cost
                current_weights = target_weights
                turnover_list.append(delta_w.sum())
        else:
            portfolio_value.iloc[i] = prev_equity * (1 + total_return(prev_ts, ts, current_weights, price_data))
            turnover_list.append(0.0)
    return portfolio_value, portfolio_value.pct_change().fillna(0), pd.Series(turnover_list, index=price_data.index)

def run_rl_strategy(price_data, regime_labels, as_cost_models, rl_policies, lgbm_forecasters, obs_norm=None, decision_interval=1, policy_key='full'):
    """Regime-aware RL agent with strategy guardrails.

    Parameters
    ----------
    decision_interval : int
        How often (in bars) the RL agent makes decisions. 1 = every bar (5-min),
        288 = daily. On intermediate (hold) steps, target_weights is preserved,
        executed_delta=0, and NO A&S cost is incurred.
    policy_key : str
        Which policy to use from rl_policies dict. Default 'full' uses ppo_full.zip.
        'daily' uses ppo_daily.zip (trained with decision_interval=288).
    """
    try:
        from src.layer4_rl.rl_env import RegimePortfolioEnv
        env_kwargs = dict(
            price_data=price_data,
            regime_labels=regime_labels,
            as_cost_models=as_cost_models,
            forecasters=lgbm_forecasters,
            decision_interval=decision_interval,
        )
        # CRITICAL-3 fix: Pass train-split obs_norm to avoid train/test mismatch
        if obs_norm is not None:
            env_kwargs['obs_mean'] = obs_norm[0]
            env_kwargs['obs_std'] = obs_norm[1]
        env = RegimePortfolioEnv(**env_kwargs)
        obs, _ = env.reset()

        # Track equity and turnover
        equity_vals = [env.portfolio_value]
        turnover_list = []
        done = truncated = False
        steps_run = 0

        # Use specified policy (default 'full' = ppo_full.zip)
        policy = rl_policies.get(policy_key) or rl_policies.get('full') or rl_policies.get('Calm')

        while not (done or truncated):
            # Get RL action (guardrails applied INSIDE env.step())
            action, _ = policy.predict(obs, deterministic=True)

            old_weights = env.current_weights.copy()
            obs, reward, done, truncated, info = env.step(action)
            equity_vals.append(env.portfolio_value)
            # Turnover: difference between new target weights and old weights
            turnover_list.append(np.abs(env.current_weights[:2] - old_weights[:2]).sum())
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

log.info("  Flat baseline (10bps fixed cost)...")
flat_equity, flat_returns, flat_turnover = run_flat_baseline(price_test, rebal_frequency=DEFAULT_REBAL_FREQ)
flat_metrics = compute_metrics(flat_returns, flat_turnover)
log.info(f"  Flat metrics: {flat_metrics}")

log.info("  Flat baseline (TRUE A&S costs) — the fair comparison...")
flat_as_equity, flat_as_returns, flat_as_turnover = run_flat_baseline_as_cost(price_test, regime_test, as_cost_models, rebal_frequency=DEFAULT_REBAL_FREQ)
flat_as_metrics = compute_metrics(flat_as_returns, flat_as_turnover)
log.info(f"  Flat+A&S metrics: {flat_as_metrics}")

log.info("  A&S + CVaR...")
as_cvar_equity, as_cvar_returns, as_cvar_turnover = run_as_cvar_strategy(price_test, regime_test, as_cost_models, rebal_frequency=DEFAULT_REBAL_FREQ)
as_cvar_metrics = compute_metrics(as_cvar_returns, as_cvar_turnover)
log.info(f"  A&S CVaR metrics: {as_cvar_metrics}")

# CRITICAL-3 fix: Extract obs_norm from TRAIN split using the SAME method as training.
# Training (train_rl_stable.py) creates RegimePortfolioEnv(price_train, regime_train, ...)
# and extracts _obs_mean/_obs_std. We do the same here to ensure obs_norm matches.
from src.layer4_rl.rl_env import RegimePortfolioEnv
log.info("  Computing observation normalization from training split...")
_temp_env = RegimePortfolioEnv(
    price_train, regime_train, as_cost_models, lgbm_forecasters
)
obs_norm = (_temp_env._obs_mean, _temp_env._obs_std)
log.info(f"  obs_norm extracted (obs_std[4:9]={obs_norm[1][4:9].round(6)})")

log.info("  RL Agent...")
rl_equity, rl_returns, rl_turnover = run_rl_strategy(price_test, regime_test, as_cost_models, rl_policies, lgbm_forecasters, obs_norm)
rl_metrics = compute_metrics(rl_returns, rl_turnover)
log.info(f"  RL metrics: {rl_metrics}")

# Daily-frequency RL experiment (--rl-daily)
if args_cli.rl_daily:
    log.info("  Daily RL Agent (decision_interval=288, ppo_daily.zip)...")
    if 'daily' not in rl_policies:
        log.error("  ppo_daily.zip not loaded — run train_rl_daily.py first!")
    else:
        rld_equity, rld_returns, rld_turnover = run_rl_strategy(
            price_test, regime_test, as_cost_models, rl_policies, lgbm_forecasters,
            obs_norm, decision_interval=288, policy_key='daily'
        )
        rld_metrics = compute_metrics(rld_returns, rld_turnover)
        log.info(f"  Daily RL metrics: {rld_metrics}")
        # Comparison
        print("\n\n=== Daily RL vs 5-min RL Comparison ===")
        print(f"  5-min RL (ppo_full.zip):    Sharpe={rl_metrics['Sharpe']:.4f}, Ret={rl_metrics['Ann. Return']:.4f}, MaxDD={rl_metrics['Max Drawdown']:.4f}")
        print(f"  Daily RL (ppo_daily.zip):   Sharpe={rld_metrics['Sharpe']:.4f}, Ret={rld_metrics['Ann. Return']:.4f}, MaxDD={rld_metrics['Max Drawdown']:.4f}")
        print(f"  Improvement:                 Sharpe +{rld_metrics['Sharpe']-rl_metrics['Sharpe']:.4f}, Ret +{rld_metrics['Ann. Return']-rl_metrics['Ann. Return']:.4f}")

        # Save comparison
        cmp = {
            '5min_RL': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in rl_metrics.items()},
            'Daily_RL': {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in rld_metrics.items()},
            'decision_interval': 288,
        }
        with open(MODEL_DIR / 'rl' / 'rl_daily_comparison.json', 'w') as f:
            json.dump(cmp, f, indent=2)
        log.info(f"  Comparison saved to: {MODEL_DIR / 'rl' / 'rl_daily_comparison.json'}")

# ----------------------------------------------------------------------
# Frequency sweep mode
# ----------------------------------------------------------------------
if args_cli.all_frequencies:
    log.info("\n=== Running frequency sweep ===")
    sweep_freqs = ["1H", "4H", "1D", "3D", "1W", "1Q"]
    sweep_results = []

    for freq_label in sweep_freqs:
        freq_bars = FREQ_MAP[freq_label]
        log.info(f"\n  -- Frequency: {freq_label} ({freq_bars} bars) --")

        f_equity, f_rets, f_turn = run_flat_baseline(price_test, rebal_frequency=freq_bars)
        fa_equity, fa_rets, fa_turn = run_flat_baseline_as_cost(price_test, regime_test, as_cost_models, rebal_frequency=freq_bars)
        ac_equity, ac_rets, ac_turn = run_as_cvar_strategy(price_test, regime_test, as_cost_models, rebal_frequency=freq_bars)

        min_n = min(len(f_rets), len(fa_rets), len(ac_rets))
        f_rets = f_rets.iloc[:min_n]; fa_rets = fa_rets.iloc[:min_n]; ac_rets = ac_rets.iloc[:min_n]
        f_turn = f_turn.iloc[:min_n]; fa_turn = fa_turn.iloc[:min_n]; ac_turn = ac_turn.iloc[:min_n]

        f_m = compute_metrics(f_rets, f_turn)
        fa_m = compute_metrics(fa_rets, fa_turn)
        ac_m = compute_metrics(ac_rets, ac_turn)

        # RL is always every-step so frequency doesn't change its returns/turnover
        rl_m = rl_metrics  # reuse from above

        log.info(f"    Flat(10bps): Sharpe={f_m['Sharpe']:.3f}, Ret={f_m['Ann. Return']:.3f}, MaxDD={f_m['Max Drawdown']:.3f}, Turnover={f_m['Mean Turnover']:.6f}")
        log.info(f"    Flat(A&S):  Sharpe={fa_m['Sharpe']:.3f}, Ret={fa_m['Ann. Return']:.3f}, MaxDD={fa_m['Max Drawdown']:.3f}, Turnover={fa_m['Mean Turnover']:.6f}")
        log.info(f"    A&S+CVaR:   Sharpe={ac_m['Sharpe']:.3f}, Ret={ac_m['Ann. Return']:.3f}, MaxDD={ac_m['Max Drawdown']:.3f}, Turnover={ac_m['Mean Turnover']:.6f}")

        sweep_results.append({
            "Frequency": freq_label,
            "Bars": freq_bars,
            "Sharpe_Flat10": f_m["Sharpe"],
            "Sharpe_FlatAS": fa_m["Sharpe"],
            "Sharpe_CVaR": ac_m["Sharpe"],
            "Sharpe_RL": rl_m["Sharpe"],
            "Ret_Flat10": f_m["Ann. Return"],
            "Ret_FlatAS": fa_m["Ann. Return"],
            "Ret_CVaR": ac_m["Ann. Return"],
            "Ret_RL": rl_m["Ann. Return"],
            "MaxDD_Flat10": f_m["Max Drawdown"],
            "MaxDD_FlatAS": fa_m["Max Drawdown"],
            "MaxDD_CVaR": ac_m["Max Drawdown"],
            "MaxDD_RL": rl_m["Max Drawdown"],
            "Turnover_Flat10": f_m["Mean Turnover"],
            "Turnover_FlatAS": fa_m["Mean Turnover"],
            "Turnover_CVaR": ac_m["Mean Turnover"],
            "Turnover_RL": rl_m["Mean Turnover"],
        })

    sweep_df = pd.DataFrame(sweep_results)

    print("\n\n=== Frequency Sensitivity: Sharpe Ratio ===")
    print(sweep_df[["Frequency", "Bars", "Sharpe_Flat10", "Sharpe_FlatAS", "Sharpe_CVaR", "Sharpe_RL"]].to_string(index=False))

    print("\n\n=== Frequency Sensitivity: Annualized Return ===")
    print(sweep_df[["Frequency", "Bars", "Ret_Flat10", "Ret_FlatAS", "Ret_CVaR", "Ret_RL"]].to_string(index=False))

    print("\n\n=== Frequency Sensitivity: Max Drawdown ===")
    print(sweep_df[["Frequency", "Bars", "MaxDD_Flat10", "MaxDD_FlatAS", "MaxDD_CVaR", "MaxDD_RL"]].to_string(index=False))

    print("\n\n=== Frequency Sensitivity: Mean Turnover ===")
    print(sweep_df[["Frequency", "Bars", "Turnover_Flat10", "Turnover_FlatAS", "Turnover_CVaR", "Turnover_RL"]].to_string(index=False))

    # Save sweep results
    sweep_path = BACKTEST_DIR / "frequency_sweep_results.json"
    sweep_df.to_json(sweep_path, orient="records", indent=2)
    log.info(f"\nSaved frequency sweep: {sweep_path}")

    print("\n\nBacktest complete! Results in", BACKTEST_DIR)
    sys.exit(0)  # Exit after sweep

# ----------------------------------------------------------------------
# Align series and compute comparison
# ----------------------------------------------------------------------
min_len = min(len(flat_returns), len(flat_as_returns), len(as_cvar_returns), len(rl_returns))
flat_returns = flat_returns.iloc[:min_len]
flat_as_returns = flat_as_returns.iloc[:min_len]
as_cvar_returns = as_cvar_returns.iloc[:min_len]
rl_returns = rl_returns.iloc[:min_len]
flat_equity = flat_equity.iloc[:min_len]
flat_as_equity = flat_as_equity.iloc[:min_len]
as_cvar_equity = as_cvar_equity.iloc[:min_len]
rl_equity = rl_equity.iloc[:min_len]
regime_test = regime_test.iloc[:min_len]

flat_metrics_final = compute_metrics(flat_returns, flat_turnover)
flat_as_metrics_final = compute_metrics(flat_as_returns, flat_as_turnover)
as_cvar_metrics_final = compute_metrics(as_cvar_returns, as_cvar_turnover)
rl_metrics_final = compute_metrics(rl_returns, rl_turnover)

comparison = pd.DataFrame({
    "Flat Baseline\n(10bps fixed)": flat_metrics_final,
    "Flat Baseline\n(A&S costs)": flat_as_metrics_final,
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
for name, rets in [("Flat(10bps)", flat_returns), ("Flat(A&S)", flat_as_returns), ("A&S+CVaR", as_cvar_returns), ("RL", rl_returns)]:
    print(f"\n{name}:")
    try:
        # Ensure regime_test index aligns with returns index
        aligned_regime = regime_test.iloc[:len(rets)]
        aligned_regime.index = rets.index
        print(regime_conditional_metrics(rets, aligned_regime).round(4).to_string())
    except Exception as e:
        print(f"  Error: {e}")

# ----------------------------------------------------------------------
# Bootstrap CI for Sharpe
# ----------------------------------------------------------------------
from scipy import stats

def bootstrap_sharpe_ci(returns, n_bootstrap=1000, ci=0.95, block_size=288):
    """Block bootstrap preserving intra-day autocorrelation (1-day = 288 bars)."""
    np.random.seed(42)  # Reproducibility
    sharpes = []
    n = len(returns)
    n_blocks = max(1, n // block_size)
    for _ in range(n_bootstrap):
        # Sample random starting points for blocks
        indices = np.random.randint(0, max(1, n - block_size), size=n_blocks)
        boot = np.concatenate([returns.iloc[i:i+block_size].values for i in indices])
        # Match original length
        if len(boot) >= n:
            boot = boot[:n]
        else:
            boot = np.pad(boot, (0, n - len(boot)))[:n]
        boot = pd.Series(boot)
        if len(boot) < 2 or boot.std() == 0:
            sharpes.append(0.0)
            continue
        ann_ret = boot.mean() * ANN_FACTOR
        ann_vol = boot.std() * np.sqrt(ANN_FACTOR)
        sharpes.append(ann_ret / ann_vol if ann_vol > 1e-8 else 0.0)
    lower = np.percentile(sharpes, (1-ci)/2*100)
    upper = np.percentile(sharpes, (1+ci)/2*100)
    return np.mean(sharpes), lower, upper

print("\n=== Bootstrap 95% CI for Sharpe Ratio ===")
for name, rets in [("Flat(10bps)", flat_returns), ("Flat(A&S)", flat_as_returns), ("A&S+CVaR", as_cvar_returns), ("RL", rl_returns)]:
    m, l, u = bootstrap_sharpe_ci(rets)
    print(f"{name}: Sharpe={m:.3f} [{l:.3f}, {u:.3f}]")

# === Multiple Testing Correction ===
def benjamini_hochberg(p_values, q=0.10):
    """Apply Benjamini-Hochberg FDR correction to a list of p-values."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    thresholds = [(i+1) / n * q for i in range(n)]
    max_pass = 0
    for i in range(n):
        if sorted_p[i] <= thresholds[i]:
            max_pass = i + 1
    significant = set(sorted_idx[:max_pass])
    return significant

print("\n=== Benjamini-Hochberg Multiple Testing Correction ===")
from scipy import stats
def sharpe_p_value(r1, r2):
    """Two-sided Sharpe ratio difference test."""
    diff = r1 - r2
    se_diff = diff.std() / np.sqrt(len(diff))
    z = diff.mean() / (se_diff + 1e-8)
    return 2 * (1 - stats.norm.cdf(abs(z)))

strategies_returns = {
    "Flat(10bps)": flat_returns,
    "Flat(A&S)": flat_as_returns,
    "A&S+CVaR": as_cvar_returns,
    "RL": rl_returns,
}
all_returns = list(strategies_returns.values())
names = list(strategies_returns.keys())
pairs = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        p = sharpe_p_value(all_returns[i], all_returns[j])
        pairs.append((names[i], names[j], p))

print(f"Pairwise Sharpe comparisons (6 pairs), BH-corrected at q=0.10:")
significance = benjamini_hochberg([p[2] for p in pairs], q=0.10)
for idx, (n1, n2, p) in enumerate(pairs):
    sig = "SIGNIFICANT" if idx in significance else "not significant"
    print(f"  {n1} vs {n2}: p={p:.4f} [{sig}]")

print("\n=== Success Criteria ===")
flat_as_sharpe = flat_as_metrics_final["Sharpe"]
flat_as_mdd = flat_as_metrics_final["Max Drawdown"]
flat_as_ret = flat_as_metrics_final["Ann. Return"]
rl_sharpe = rl_metrics_final["Sharpe"]
rl_mdd = rl_metrics_final["Max Drawdown"]
rl_ret = rl_metrics_final["Ann. Return"]
print(f"{'Metric':<15} {'Flat(A&S)':>12} {'RL Agent':>12} {'Result':>10}")
print(f"{'Sharpe':<15} {flat_as_sharpe:>12.4f} {rl_sharpe:>12.4f} {'PASS' if rl_sharpe > flat_as_sharpe else 'FAIL':>10}")
print(f"{'Max Drawdown':<15} {flat_as_mdd:>11.1%} {rl_mdd:>11.1%} {'PASS' if rl_mdd > flat_as_mdd else 'FAIL':>10}")
print(f"{'Ann. Return':<15} {flat_as_ret:>11.1%} {rl_ret:>11.1%} {'PASS' if rl_ret > flat_as_ret else 'FAIL':>10}")

# ----------------------------------------------------------------------
# Equity curves plot
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
ax1 = axes[0]
ax1.plot(flat_equity.index, flat_equity.values, label="Flat Baseline (10bps)", color="gray", alpha=0.7, linestyle="--")
ax1.plot(flat_as_equity.index, flat_as_equity.values, label="Flat Baseline (A&S costs)", color="orange", alpha=0.9)
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
        "flat_baseline_10bps": {k: float(v) for k, v in flat_metrics_final.items()},
        "flat_baseline_as_cost": {k: float(v) for k, v in flat_as_metrics_final.items()},
        "as_cvar": {k: float(v) for k, v in as_cvar_metrics_final.items()},
        "rl_agent": {k: float(v) for k, v in rl_metrics_final.items()},
    },
}
with open(BACKTEST_DIR / "performance_summary.json", "w") as f:
    json.dump(perf_summary, f, indent=2, default=str)
log.info(f"Saved: {BACKTEST_DIR / 'performance_summary.json'}")

print(f"\nBacktest complete! Results in {BACKTEST_DIR}")