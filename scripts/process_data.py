"""
Data Processing Layer 0: Feature Engineering and Alignment

Loads raw Binance data and produces analysis-ready parquet files.

Usage:
    python scripts/process_data.py
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/process_data.log"),
    ],
)
log = logging.getLogger(__name__)


def load_raw_data(data_dir: Path) -> dict:
    """Load raw OHLCV and trade data from parquet files."""
    raw_dir = Path(data_dir) / "raw"
    metadata_path = raw_dir / "fetch_metadata.json"

    data = {}

    # Load metadata
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"synthetic_data": False, "symbols": []}
        log.warning("No metadata found, assuming real data")

    # Load OHLCV data
    # NOTE: Use _5m_batch.parquet files instead of _ohlcv_5m.parquet
    # The batch files have correct price levels; the original files have stale prices
    # before 2026-03-28 10:35:00 causing impossible returns (110% BTC, 31% ETH)
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        # Prefer new 4-year files (BTC_USDT_ohlcv_5m.parquet) over batch files
        symbol_upper = symbol.replace("/", "_")
        ohlcv_new = raw_dir / f"{symbol_upper}_ohlcv_5m.parquet"
        ohlcv_batch = raw_dir / f"{symbol.lower()[:3]}_5m_batch.parquet"
        trades_path = raw_dir / f"{symbol_upper}_trades_15m.parquet"

        # Use new 4-year files if available, otherwise fall back to batch
        if ohlcv_new.exists():
            ohlcv_path = ohlcv_new
            log.info(f"Using new 4-year file: {ohlcv_path.name}")
        elif ohlcv_batch.exists():
            ohlcv_path = ohlcv_batch
            log.info(f"Using batch file: {ohlcv_path.name}")
        else:
            log.warning(f"No OHLCV data found for {symbol}")
            continue

        if ohlcv_path.exists():
            data[symbol] = {
                "ohlcv": pd.read_parquet(ohlcv_path),
                "trades": pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame(),
            }
            log.info(f"Loaded {symbol}: {len(data[symbol]['ohlcv'])} OHLCV bars, {len(data[symbol]['trades'])} trades")
        else:
            log.warning(f"No data found for {symbol}")

    return data, metadata


def compute_price_features(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price features for combined BTC/ETH dataset.

    Output columns:
    - timestamp, open, high, low, close, volume (from aligned data)
    - btc_close, eth_close, btc_return, eth_return
    - realized_vol (20-period rolling)
    - spread_proxy ((high-low)/close)
    - return_lag_1, return_lag_3, return_lag_6 for BTC and ETH
    - cross_asset_return (BTC return for ETH model, ETH return for BTC model)
    """
    # Start with BTC OHLCV as base
    df = btc_df.copy()
    df = df.rename(columns={c: f"btc_{c}" if c != "timestamp" else c for c in df.columns})

    # Merge ETH close prices (align to BTC timestamps)
    eth_aligned = eth_df[["timestamp", "close"]].rename(columns={"close": "eth_close"})
    df = df.merge(eth_aligned, on="timestamp", how="left")

    # Compute log returns
    df["btc_return"] = np.log(df["btc_close"] / df["btc_close"].shift(1))
    df["eth_return"] = np.log(df["eth_close"] / df["eth_close"].shift(1))

    # Realized volatility (20-period rolling std of returns)
    df["realized_vol"] = df["btc_return"].rolling(20).std()

    # Spread proxy
    df["spread_proxy"] = (df["btc_high"] - df["btc_low"]) / df["btc_close"]

    # Lagged returns for BTC
    for lag in [1, 3, 6]:
        df[f"btc_return_lag_{lag}"] = df["btc_return"].shift(lag)

    # Lagged returns for ETH
    for lag in [1, 3, 6]:
        df[f"eth_return_lag_{lag}"] = df["eth_return"].shift(lag)

    # Cross-asset returns
    df["btc_return_for_eth_model"] = df["btc_return"]  # BTC return for ETH model
    df["eth_return_for_btc_model"] = df["eth_return"]  # ETH return for BTC model

    # Rename BTC OHLCV columns to standard names
    df = df.rename(
        columns={
            "btc_open": "open",
            "btc_high": "high",
            "btc_low": "low",
            "btc_volume": "volume",
        }
    )
    # close = BTC close (base asset). Keep btc_close separately for cross-asset features.
    # btc_close is renamed to close below; save it first
    df["close"] = df["btc_close"].copy()  # btc_close still exists here
    df["btc_close"] = df["close"].copy()  # btc_close = close = BTC close

    # Select final columns in order
    final_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "btc_close",
        "eth_close",
        "btc_return",
        "eth_return",
        "realized_vol",
        "spread_proxy",
        "btc_return_lag_1",
        "btc_return_lag_3",
        "btc_return_lag_6",
        "eth_return_lag_1",
        "eth_return_lag_3",
        "eth_return_lag_6",
        "btc_return_for_eth_model",
        "eth_return_for_btc_model",
    ]

    return df[final_cols]


def process_trades(btc_trades: pd.DataFrame, eth_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Process and align trade data to common 5-minute timestamps.

    Returns trades with columns: timestamp, side, volume, price
    """
    # Concatenate trades with asset indicator
    btc_trades = btc_trades.copy()
    eth_trades = eth_trades.copy()

    btc_trades["asset"] = "BTC"
    eth_trades["asset"] = "ETH"

    all_trades = pd.concat([btc_trades, eth_trades], ignore_index=True)

    if all_trades.empty:
        return pd.DataFrame(columns=["timestamp", "side", "volume", "price"])

    # Sort by timestamp
    all_trades = all_trades.sort_values("timestamp").reset_index(drop=True)

    # Aggregate to 5-minute bins
    all_trades["timestamp_bin"] = all_trades["timestamp"].dt.floor("5min")

    # For each bin, compute volume-weighted average price and dominant side
    def dominant_side(series):
        buy_count = (series == "buy").sum()
        sell_count = (series == "sell").sum()
        return "buy" if buy_count >= sell_count else "sell"

    def weighted_avg_price(series):
        volumes = all_trades.loc[series.index, "volume"].values
        prices = series.values
        total_vol = volumes.sum()
        if total_vol == 0:
            return np.nan
        return np.average(prices, weights=volumes)

    aggregated = (
        all_trades.groupby("timestamp_bin")
        .agg(
            side=("side", dominant_side),
            volume=("volume", "sum"),
            price=("price", weighted_avg_price),
        )
        .reset_index()
        .rename(columns={"timestamp_bin": "timestamp"})
    )

    return aggregated


def align_to_common_index(
    price_df: pd.DataFrame, trades_df: pd.DataFrame, freq: str = "5min"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align both DataFrames to a common time index.
    """
    # Create complete time index spanning the data range
    if price_df.empty:
        raise ValueError("Price DataFrame is empty")

    start_time = price_df["timestamp"].min()
    end_time = price_df["timestamp"].max()
    common_index = pd.date_range(start=start_time, end=end_time, freq=freq)

    # Reindex price data
    price_df = price_df.set_index("timestamp")
    price_df = price_df.reindex(common_index)
    price_df.index.name = "timestamp"
    price_df = price_df.reset_index()

    # Reindex trades data
    if not trades_df.empty:
        trades_df = trades_df.set_index("timestamp")
        trades_df = trades_df.reindex(common_index)
        trades_df.index.name = "timestamp"
        # Forward-fill gaps: bins with no trades inherit last known values
        trades_df["side"] = trades_df["side"].ffill()
        trades_df["volume"] = trades_df["volume"].fillna(0.0)
        trades_df["price"] = trades_df["price"].ffill()
        trades_df = trades_df.reset_index()
    else:
        trades_df = pd.DataFrame(columns=["timestamp", "side", "volume", "price"])

    return price_df, trades_df


def log_data_quality(df: pd.DataFrame, name: str):
    """Log data quality metrics."""
    log.info(f"=== Data Quality Report: {name} ===")
    log.info(f"  Shape: {df.shape}")
    log.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    log.info(f"  Null counts:\n{df.isnull().sum().to_string()}")
    log.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Process raw Binance data into features")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    log.info("Loading raw data...")
    raw_data, metadata = load_raw_data(data_dir)

    if not raw_data:
        log.error("No raw data found. Run fetch_binance_data.py first.")
        return

    btc_df = raw_data.get("BTC/USDT", {}).get("ohlcv", pd.DataFrame())
    eth_df = raw_data.get("ETH/USDT", {}).get("ohlcv", pd.DataFrame())
    btc_trades = raw_data.get("BTC/USDT", {}).get("trades", pd.DataFrame())
    eth_trades = raw_data.get("ETH/USDT", {}).get("trades", pd.DataFrame())

    # Compute price features
    log.info("Computing price features...")
    price_features = compute_price_features(btc_df, eth_df)

    # Process trades
    log.info("Processing trades...")
    trades_processed = process_trades(btc_trades, eth_trades)

    # Align to common 5-minute index
    log.info("Aligning to common 5-minute index...")
    price_features_aligned, trades_aligned = align_to_common_index(price_features, trades_processed)

    # Log data quality
    log_data_quality(price_features_aligned, "price_features")
    log_data_quality(trades_aligned, "trades_processed")

    # Save processed data
    price_path = processed_dir / "price_features.parquet"
    price_features_aligned.to_parquet(price_path, index=False)
    log.info(f"Saved price features to {price_path}")

    trades_path = processed_dir / "trades_processed.parquet"
    trades_aligned.to_parquet(trades_path, index=False)
    log.info(f"Saved trades to {trades_path}")

    # Save metadata about processing
    processing_metadata = {
        "processed_at": datetime.now().isoformat(),
        "synthetic_data": metadata.get("synthetic_data", False),
        "source_symbols": metadata.get("symbols", []),
        "n_price_records": len(price_features_aligned),
        "n_trade_records": len(trades_aligned),
        "date_range": {
            "start": str(price_features_aligned["timestamp"].min()),
            "end": str(price_features_aligned["timestamp"].max()),
        },
    }

    metadata_path = processed_dir / "processing_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(processing_metadata, f, indent=2)
    log.info(f"Saved processing metadata to {metadata_path}")

    log.info("Processing complete!")


if __name__ == "__main__":
    main()
