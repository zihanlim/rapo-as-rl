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
    early_stopping_rounds=50, 20% validation holdout

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
from sklearn.model_selection import TimeSeriesSplit

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
    """
    close_col = f"{asset.lower()}_close" if f"{asset.lower()}_close" in price_df.columns else asset.lower()
    df = price_df.copy()

    # Lagged returns
    for lag in [1, 3, 6]:
        df[f"return_lag_{lag}"] = df[close_col].pct_change(lag)

    # Realized volatility
    df["realized_vol"] = df[close_col].pct_change().rolling(20).std()

    # Spread proxy
    df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]

    # Order flow imbalance
    trades_df["timestamp_bin"] = trades_df["timestamp"].floor("5min")
    ofi = (
        trades_df.groupby("timestamp_bin")["volume"]
        .apply(lambda x: (x * (trades_df.loc[x.index, "side"] == "buy").astype(int) * 2 - 1).sum())
    )
    df["ofi"] = ofi.reindex(df.index).fillna(0)

    # Cross-asset: BTC return for ETH model and vice versa
    if asset == "ETH":
        btc_close = df.get("btc_close", df[close_col])  # fallback
        df["cross_asset_return"] = btc_close.pct_change()
    else:
        eth_close = df.get("eth_close", df[close_col])
        df["cross_asset_return"] = eth_close.pct_change()

    # Regime indicator (one-hot)
    for r in ["Calm", "Volatile", "Stressed"]:
        df[f"regime_{r}"] = (regime_labels == r).astype(int)

    # Target: next-period return
    y = df[close_col].pct_change().shift(-1)

    feature_cols = [
        "return_lag_1", "return_lag_3", "return_lag_6",
        "realized_vol", "spread_proxy", "ofi",
        "cross_asset_return",
        "regime_Calm", "regime_Volatile", "regime_Stressed",
    ]
    X = df[feature_cols]
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid]
    y = y[valid]

    log.info(f"{asset} features: {X.shape}, target non-null: {valid.sum()}")
    return X, y


def train_regime_model(
    X: pd.DataFrame,
    y: pd.Series,
    regime: str,
    asset: str,
    val_frac: float = 0.2,
) -> lgb.LGBMRegressor:
    """
    Train LightGBM for a specific regime.
    Uses last val_frac of data as time-series validation set.
    """
    regime_mask = X[f"regime_{regime}"] == 1
    X_r = X[regime_mask]
    y_r = y[regime_mask]

    if len(X_r) < 100:
        log.warning(f"Very few samples for {asset}/{regime}: {len(X_r)}")
        return None

    split = int(len(X_r) * (1 - val_frac))
    X_train, X_val = X_r.iloc[:split], X_r.iloc[split:]
    y_train, y_val = y_r.iloc[:split], y_r.iloc[split:]

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
    log.info(f"  {asset}/{regime}: train R²={train_score:.4f}, val R²={val_score:.4f}, best_iter={model.best_iteration_}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Layer 3: LightGBM Return Forecaster Training")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--regime", type=str, default="models/hmm/regime_labels.csv")
    parser.add_argument("--output", type=str, default="models/lgbm")
    args = parser.parse_args()

    import joblib

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    price_df = pd.read_parquet(Path(args.data) / "price_features.parquet")
    trades_df = pd.read_parquet(Path(args.data) / "trades_processed.parquet")
    regime_labels = pd.read_csv(args.regime, index_col=0, squeeze=True)
    regime_labels.index = pd.to_datetime(regime_labels.index)

    for asset in ["BTC", "ETH"]:
        log.info(f"Training LightGBM for {asset}...")
        X, y = build_features(price_df, trades_df, regime_labels, asset)

        for regime in ["Calm", "Volatile", "Stressed"]:
            model = train_regime_model(X, y, regime, asset)
            if model is not None:
                fname = f"lgbm_{asset.lower()}_{regime.lower()}.pkl"
                joblib.dump(model, output_dir / fname)
                log.info(f"  Saved: {fname}")

    print(f"\nLightGBM training complete. Models saved to {output_dir}")


if __name__ == "__main__":
    main()
