"""
Layer 2: Avellaneda-Stoikov Per-Regime Cost Model — Calibration

Calibrates A&S microstructure cost parameters separately for each HMM regime
(Calm, Volatile, Stressed) using Binance trade tick data.

A&S Model:
    Cost(q, σ, s, δ, γ) = σ·√(q/(2δ)) + s/2 + γ·q²/(2δ)

where:
    q   = order size (in asset units)
    σ   = asset volatility (annualized)
    s   = bid-ask spread (in price units)
    δ   = market depth per price unit
    γ   = risk aversion parameter

Usage:
    python -m src.layer2_as.as_calibrate --regime_csv models/hmm/regime_labels.csv \
           --trades data/processed/trades_processed.parquet --output models/as_cost/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
import warnings

warnings.filterwarnings("ignore")


def estimate_spread_from_trades(trades_df: pd.DataFrame) -> float:
    """
    Estimate bid-ask spread from trade direction clustering.
    Uses the standard way to infer effective spread from trade ticks:
        spread ≈ 2 * |trade_price - mid_price| weighted by volume
    """
    trades = trades_df.copy()
    trades["mid_estimate"] = trades["price"].rolling(5, min_periods=1).mean()
    trades["signed_trade"] = (trades["side"] == "buy").astype(int) * 2 - 1
    trades["half_spread"] = trades["signed_trade"] * (trades["price"] - trades["mid_estimate"])
    effective_spread = 2 * trades["half_spread"].abs().mean()
    return effective_spread


def estimate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Estimate realized volatility from 5-min returns."""
    rv = returns.std()
    if annualize:
        rv = rv * np.sqrt(288 * 365)  # 5-min bars, annualize
    return rv


def estimate_depth(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    window: str = "1H",
) -> float:
    """
    Estimate market depth δ (average volume per price unit) over a rolling window.
    δ = total volume / (price_range in window)
    """
    trades_agg = (
        trades_df.set_index("timestamp")
        .resample(window)["volume"]
        .sum()
    )
    price_agg = (
        price_df.set_index("timestamp")
        .resample(window)
        .agg({"high": "max", "low": "min", "close": "last"})
    )
    price_range = price_agg["high"] - price_agg["low"]
    depth = trades_agg.reindex(price_agg.index).dropna() / price_range.dropna()
    return depth.mean()


def calibrate_regime(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    regime: str,
    regime_idx: int,
    regime_labels: pd.Series,
) -> dict:
    """
    Calibrate A&S parameters for a single regime.

    Parameters
    ----------
    regime_labels : pd.Series
        Regime time series from HMM (indexed by timestamp)
    regime_idx : int
        Integer label for this regime (0, 1, or 2)
    """
    # Filter trades to this regime
    regime_mask = regime_labels == regime
    if regime_mask.sum() == 0:
        log.warning(f"No trades found for regime: {regime}")
        return None

    regime_trades = trades_df[regime_mask]
    regime_prices = price_df[regime_mask]

    params = {
        "regime": regime,
        "regime_idx": regime_idx,
        "n_trades": len(regime_trades),
        "spread": estimate_spread_from_trades(regime_trades),
        "volatility": estimate_volatility(np.log(regime_prices["close"] / regime_prices["close"].shift(1)).dropna()),
        "depth": estimate_depth(regime_trades, regime_prices),
        "gamma": 1e-6,  # default; calibrate via MLE if time permits
    }

    log.info(
        f"  {regime}: spread={params['spread']:.6f}, "
        f"vol={params['volatility']:.4f}, depth={params['depth']:.2f}"
    )
    return params


def compute_cost(
    q: float,
    params: dict,
    current_price: float = 1.0,
) -> float:
    """
    Compute A&S execution cost for order size q.

    Returns cost in absolute currency units.

    Cost = σ·√(q/(2δ)) + s/2 + γ·q²/(2δ)
    """
    sigma = params["volatility"]
    s = params["spread"]
    delta = params["depth"]
    gamma = params["gamma"]

    market_impact = sigma * np.sqrt(q / (2 * delta)) * current_price
    spread_cost = (s / 2) * current_price
    impact_cost = gamma * (q**2) / (2 * delta) * current_price

    return market_impact + spread_cost + impact_cost


def compute_cost_bps(
    q: float,
    params: dict,
    current_price: float = 1.0,
) -> float:
    """Compute A&S execution cost in basis points."""
    cost_abs = compute_cost(q, params, current_price)
    notional = q * current_price
    return (cost_abs / notional) * 10_000


def main():
    parser = argparse.ArgumentParser(description="Layer 2: A&S Per-Regime Calibration")
    parser.add_argument("--regime_csv", type=str, default="models/hmm/regime_labels.csv")
    parser.add_argument("--trades", type=str, default="data/processed/trades_processed.parquet")
    parser.add_argument("--prices", type=str, default="data/processed/price_features.parquet")
    parser.add_argument("--output", type=str, default="models/as_cost")
    args = parser.parse_args()

    import joblib

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    regime_labels = pd.read_csv(args.regime_csv, index_col=0, squeeze=True)
    regime_labels.index = pd.to_datetime(regime_labels.index)
    trades_df = pd.read_parquet(args.trades)
    price_df = pd.read_parquet(args.prices)

    # Align all data to common index
    trades_df = trades_df[trades_df["timestamp"] <= regime_labels.index.max()]
    regime_map = {label: idx for idx, label in enumerate(["Calm", "Volatile", "Stressed"])}

    results = {}
    for regime in ["Calm", "Volatile", "Stressed"]:
        log.info(f"Calibrating {regime}...")
        params = calibrate_regime(
            trades_df,
            price_df,
            regime,
            regime_map[regime],
            regime_labels,
        )
        if params:
            results[regime] = params

    # Save per-regime cost models
    for regime, params in results.items():
        regime_clean = regime.lower()
        joblib.dump(params, output_dir / f"as_cost_{regime_clean}.pkl")

    # Validate: stressed cost should be >> calm cost
    log.info("\n=== A&S Cost Validation (50 bps trade) ===")
    trade_size_bps = 0.0050  # 50 bps notional
    for regime, params in results.items():
        cost_bps = compute_cost_bps(trade_size_bps, params)
        log.info(f"  {regime}: {cost_bps:.1f} bps")

    stressed_cost = results["Stressed"]["volatility"] / results["Calm"]["volatility"] if "Calm" in results and "Stressed" in results else None
    if stressed_cost:
        log.info(f"  Volatility ratio (Stressed/Calm): {stressed_cost:.1f}x — should be ~10-100x")

    print(f"\nCalibration complete. Models saved to {output_dir}")


if __name__ == "__main__":
    main()
