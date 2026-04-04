"""
Layer 3: LightGBM Return Forecaster — Training

Trains regime-conditional LightGBM return forecasters for BTC and ETH.
Separate model per regime (Calm / Volatile / Stressed) and per asset.

Features:
    - Lagged returns (1-, 3-, 6-period)
    - Realized volatility (20-period rolling)
    - Order flow imbalance (OFI)
    - Spread proxy
    - Regime indicator (one-hot)
    - Cross-asset correlation (BTC-ETH)

Hyperparameters (per Problem Statement):
    num_leaves=31, learning_rate=0.05, n_estimators=500,
    early_stopping_rounds=50, chronological validation split

Key fixes vs skeleton:
    - OFI: properly compute buy-volume minus sell-volume per time bin
    - Cross-asset: BTC model -> ETH returns, ETH model -> BTC returns
    - Regime alignment: reindex regime_labels to match price_df index

Usage:
    python -m src.layer3_lightgbm.lgbm_train --data data/processed \
           --regime models/hmm/regime_labels.csv --output models/lgbm/
"""

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_features(
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    regime_labels: pd.Series,
    asset: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y for LightGBM forecaster.

    Parameters
    ----------
    asset : str
        'BTC' or 'ETH'

    Key fixes vs skeleton:
    - OFI: properly groupby timestamp_bin and side to compute signed volume
    - Cross-asset: BTC model uses ETH returns, ETH model uses BTC returns
    - Regime alignment: reindex regime_labels to match price_df index
    """
    asset_lower = asset.lower()
    close_col = f"{asset_lower}_close" if f"{asset_lower}_close" in price_df.columns else 'close'

    df = price_df.copy()

    # Lagged returns
    for lag in [1, 3, 6]:
        df[f"return_lag_{lag}"] = df[close_col].pct_change(lag)

    # Realized volatility
    df["realized_vol"] = df[close_col].pct_change().rolling(20).std()

    # Spread proxy
    df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]

    # OFI: FIXED - properly compute buy volume - sell volume per time bin
    trades_df = trades_df.copy()
    trades_df["timestamp_bin"] = pd.to_datetime(trades_df["timestamp"]).dt.floor("5min")
    trades_df["signed_volume"] = trades_df["volume"] * (trades_df["side"] == "buy").astype(int).replace(0, -1)
    ofi = (
        trades_df.groupby("timestamp_bin")["signed_volume"]
        .sum()
        .rolling(5).mean()
    )
    df["ofi"] = ofi.reindex(df.index).fillna(0)

    # Cross-asset return: FIXED - use OTHER asset's returns
    if asset == "BTC":
        # BTC model uses ETH returns
        cross_close = df.get("eth_close", df[close_col])
    else:
        # ETH model uses BTC returns
        cross_close = df.get("btc_close", df[close_col])
    df["cross_asset_return"] = cross_close.pct_change()

    # Ensure price_df has datetime index for regime alignment
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('timestamp', inplace=True)

    # Regime indicators: align regime_labels to price_df index
    regime_aligned = regime_labels.reindex(df.index)
    for r in ["Calm", "Volatile", "Stressed"]:
        df[f"regime_{r}"] = (regime_aligned == r).astype(int)

    # Target: next-period return (shifted -1 avoids look-ahead bias)
    y = df[close_col].pct_change().shift(-1)

    feature_cols = [
        "return_lag_1", "return_lag_3", "return_lag_6",
        "realized_vol", "spread_proxy", "ofi",
        "cross_asset_return",
        "regime_Calm", "regime_Volatile", "regime_Stressed",
    ]
    X = df[feature_cols]

    # Filter to valid rows (no NaN in features or target)
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid]
    y = y[valid]

    # Re-apply regime indicators after filtering to ensure alignment
    regime_aligned_valid = regime_aligned[valid]
    for r in ["Calm", "Volatile", "Stressed"]:
        X[f"regime_{r}"] = (regime_aligned_valid == r).astype(int)

    log.info(f"{asset} features: {X.shape}, target non-null: {valid.sum()}")
    return X, y


def train_regime_model(
    X: pd.DataFrame,
    y: pd.Series,
    regime: str,
    asset: str,
    val_start_date: pd.Timestamp = None,
    val_frac: float = 0.2,
) -> lgb.LGBMRegressor:
    """
    Train LightGBM for a specific regime.
    Uses chronological validation split (no shuffling).
    """
    regime_mask = X[f"regime_{regime}"] == 1
    X_r = X[regime_mask]
    y_r = y[regime_mask]

    # Regime-specific minimum sample thresholds (stressed has fewer observations)
    min_samples = 20 if regime == "Stressed" else 100
    if len(X_r) < min_samples:
        log.warning(f"Very few samples for {asset}/{regime}: {len(X_r)}, need {min_samples}")
        return None

    # Chronological split - use datetime index if available, otherwise fraction
    # For stressed regime, if date-based split yields no training data, fall back to fraction split
    use_date_split = (
        val_start_date is not None
        and isinstance(X_r.index, pd.DatetimeIndex)
        and len(X_r.index) > 0
        and isinstance(X_r.index[0], pd.Timestamp)
        and regime != "Stressed"  # Stressed regime uses fraction split due to small size
    )
    if use_date_split:
        train_mask = X_r.index < val_start_date
        val_mask = X_r.index >= val_start_date
        X_train, X_val = X_r[train_mask], X_r[val_mask]
        y_train, y_val = y_r[train_mask], y_r[val_mask]
    else:
        split = int(len(X_r) * (1 - val_frac))
        X_train, X_val = X_r.iloc[:split], X_r.iloc[split:]
        y_train, y_val = y_r.iloc[:split], y_r.iloc[split:]

    # Lower train/val requirements for stressed regime (only 20 total samples)
    min_train = 8 if regime == "Stressed" else 50
    min_val = 3 if regime == "Stressed" else 20
    if len(X_train) < min_train or len(X_val) < min_val:
        log.warning(f"Insufficient train/val samples for {asset}/{regime}: train={len(X_train)}, val={len(X_val)}")
        return None

    model = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        early_stopping_rounds=50,
        verbose=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val) if len(X_val) > 0 else float("nan")
    log.info(f"  {asset}/{regime}: train R2={train_score:.4f}, val R2={val_score:.4f}, best_iter={model.best_iteration_}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Layer 3: LightGBM Return Forecaster Training")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--regime", type=str, default="models/hmm/regime_labels.csv")
    parser.add_argument("--output", type=str, default="models/lgbm")
    parser.add_argument("--val_start", type=str, default="2024-07-01",
                        help="Validation start date (YYYY-MM-DD)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_df = pd.read_parquet(Path(args.data) / "price_features.parquet")
    if 'timestamp' in price_df.columns and not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.set_index('timestamp', inplace=True)
    trades_df = pd.read_parquet(Path(args.data) / "trades_processed.parquet")
    regime_labels = pd.read_csv(args.regime, index_col=0).iloc[:, 0]
    regime_labels.index = pd.to_datetime(regime_labels.index)

    val_start_date = pd.Timestamp(args.val_start)

    for asset in ["BTC", "ETH"]:
        log.info(f"Training LightGBM for {asset}...")
        X, y = build_features(price_df, trades_df, regime_labels, asset)

        for regime in ["Calm", "Volatile", "Stressed"]:
            model = train_regime_model(X, y, regime, asset, val_start_date=val_start_date)
            if model is not None:
                fname = f"lgbm_{asset.lower()}_{regime.lower()}.pkl"
                joblib.dump(model, output_dir / fname)
                log.info(f"  Saved: {fname}")

    log.info(f"\nLightGBM training complete. Models saved to {output_dir}")


if __name__ == "__main__":
    main()
