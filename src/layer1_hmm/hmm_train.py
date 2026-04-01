"""
Layer 1: HMM Regime Classifier — Training and Evaluation

Trains a Gaussian HMM with 3 states (Calm / Volatile / Stressed) on
Binance OHLCV and trade tick data. Validates state count using BIC/AIC.

Usage:
    python -m src.layer1_hmm.hmm_train --data_dir data/processed --output models/hmm/
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Feature engineering for HMM
# ----------------------------------------------------------------------


def build_hmm_features(price_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for HMM regime classification.

    Features:
    - 5-min log return
    - Realized volatility (20-period rolling)
    - Order flow imbalance (OFI)
    - Spread proxy (from trade direction clustering)

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV data with columns: timestamp, open, high, low, close, volume
    trades_df : pd.DataFrame
        Trade tick data with columns: timestamp, side (buy/sell), volume

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by timestamp, shape (T, n_features)
    """
    df = price_df.copy()
    df["return"] = np.log(df["close"] / df["close"].shift(1))
    df["realized_vol"] = df["return"].rolling(20).std()

    # Spread proxy: high - low relative to close
    df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]

    # Order flow imbalance from trade ticks
    # Align trades to 5-min bars
    trades_df["timestamp_bin"] = trades_df["timestamp"].floor("5min")
    ofi = (
        trades_df.groupby("timestamp_bin")["volume"]
        .apply(lambda x: (x * (trades_df.loc[x.index, "side"] == "buy").astype(int) * 2 - 1).sum())
    )
    df["ofi"] = ofi.reindex(df.index).fillna(0).rolling(5).mean()

    feature_cols = ["return", "realized_vol", "spread_proxy", "ofi"]
    features = df[feature_cols].dropna()
    log.info(f"HMM feature matrix built: {features.shape}")

    return features


# ----------------------------------------------------------------------
# Model selection: BIC / AIC
# ----------------------------------------------------------------------


def select_states(features: np.ndarray, candidate_states: list[int] = None) -> dict:
    """
    Select optimal number of HMM states using BIC and AIC.

    Returns dict with scores per state count.
    """
    if candidate_states is None:
        candidate_states = [2, 3, 4]

    scores = {}
    for n_states in candidate_states:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=500,
            random_state=42,
        )
        model.fit(features)
        scores[n_states] = {
            "bic": model.bic(features),
            "aic": model.aic(features),
            "log_likelihood": model.score(features),
        }
        log.info(f"  n_states={n_states}: BIC={scores[n_states]['bic']:.2f}, AIC={scores[n_states]['aic']:.2f}")

    best = min(scores, key=lambda k: scores[k]["bic"])
    log.info(f"Best state count (BIC): {best}")
    return scores, best


# ----------------------------------------------------------------------
# Main training
# ----------------------------------------------------------------------


def train_hmm(
    features: np.ndarray,
    n_states: int = 3,
    date_index: pd.DatetimeIndex = None,
) -> GaussianHMM:
    """
    Train final HMM with n_states and label states as Calm / Volatile / Stressed.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=500,
        random_state=42,
    )
    model.fit(features)
    log.info(f"HMM trained: {n_states} states, log-likelihood={model.score(features):.2f}")

    # Label states based on mean volatility (lowest vol = Calm, highest = Stressed)
    state_vols = model.means_[:, 1]  # column 1 = realized_vol feature
    sorted_states = np.argsort(state_vols)  # ascending: lowest vol first
    state_labels = {sorted_states[0]: "Calm", sorted_states[1]: "Volatile", sorted_states[2]: "Stressed"}
    model.state_labels_ = state_labels
    log.info(f"State labels: {state_labels}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Layer 1: HMM Regime Classifier Training")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    parser.add_argument("--output", type=str, default="models/hmm", help="Output model directory")
    parser.add_argument("--n_states", type=int, default=3, help="Number of HMM states")
    args = parser.parse_args()

    import joblib

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processed data
    price_df = pd.read_parquet(Path(args.data_dir) / "price_features.parquet")
    trades_df = pd.read_parquet(Path(args.data_dir) / "trades_processed.parquet")

    # Build features
    features = build_hmm_features(price_df, trades_df)
    X = features.values

    # State selection (BIC/AIC comparison)
    log.info("Running state selection (BIC/AIC)...")
    scores, best_n = select_states(X)
    n_states = args.n_states if args.n_states != 3 else best_n

    # Train final model
    model = train_hmm(X, n_states=n_states)

    # Predict regime sequence
    hidden_states = model.predict(X)
    regime_labels = pd.Series(hidden_states).map(model.state_labels_)
    regime_labels.index = features.index
    regime_labels.to_csv(output_dir / "regime_labels.csv")
    log.info(f"Regime labels saved. Distribution:\n{regime_labels.value_counts()}")

    # Save model
    joblib.dump(model, output_dir / "hmm_model.pkl")
    log.info(f"Model saved to {output_dir / 'hmm_model.pkl'}")


if __name__ == "__main__":
    main()
