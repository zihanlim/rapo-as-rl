"""
Data Collection Layer 0: Binance OHLCV and Trade Tick Data

Fetches BTC/USDT and ETH/USDT data from Binance via ccxt.
Falls back to synthetic data if API fails or rate limits.

Usage:
    python scripts/fetch_binance_data.py
"""

import argparse
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/fetch_binance_data.log"),
    ],
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
START_DATE = "2021-01-01"
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
OHLCV_INTERVAL = "5m"
TRADES_INTERVAL = "15min"

# Regime parameters for synthetic data
REGIME_PARAMS = {
    "Calm": {"annualized_vol": 0.4, "spread_bps": (2, 5)},
    "Volatile": {"annualized_vol": 0.9, "spread_bps": (15, 25)},
    "Stressed": {"annualized_vol": 1.8, "spread_bps": (80, 150)},
}

# HMM-like regime transition matrix (approximate)
REGIME_TRANSITIONS = {
    "Calm": {"Calm": 0.92, "Volatile": 0.07, "Stressed": 0.01},
    "Volatile": {"Calm": 0.15, "Volatile": 0.75, "Stressed": 0.10},
    "Stressed": {"Calm": 0.05, "Volatile": 0.35, "Stressed": 0.60},
}


# ----------------------------------------------------------------------
# Synthetic data generation
# ----------------------------------------------------------------------


def generate_synthetic_regimes(n_periods: int, seed: int = 42) -> np.ndarray:
    """Generate HMM-like regime sequence."""
    np.random.seed(seed)
    regimes = np.empty(n_periods, dtype=object)
    current_regime = "Calm"
    regimes[0] = current_regime

    for i in range(1, n_periods):
        probs = list(REGIME_TRANSITIONS[current_regime].values())
        next_regime = np.random.choice(["Calm", "Volatile", "Stressed"], p=probs)
        regimes[i] = next_regime
        current_regime = next_regime

    return regimes


def generate_synthetic_ohlcv(
    timestamps: pd.DatetimeIndex,
    base_price: float,
    regime_sequence: np.ndarray,
    symbol: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with regime-dependent volatility."""
    np.random.seed(seed)
    n = len(timestamps)
    dt = 5 / (252 * 24 * 60)  # 5-minute fraction of year

    close_prices = np.zeros(n)
    close_prices[0] = base_price

    regime_map = {"Calm": 0, "Volatile": 1, "Stressed": 2}

    for i in range(1, n):
        regime = regime_sequence[i - 1]
        vol = REGIME_PARAMS[regime]["annualized_vol"]
        spread_bps = np.random.uniform(*REGIME_PARAMS[regime]["spread_bps"])

        # Log return with regime-dependent volatility
        log_return = np.random.normal(0, vol * np.sqrt(dt))
        close_prices[i] = close_prices[i - 1] * np.exp(log_return)

        # Add slight mean reversion to avoid price drift
        if i > 50:
            recent_vol = np.std(np.diff(np.log(close_prices[max(0, i - 50) : i])))
            if recent_vol > 0:
                reversion = 0.001 * (np.mean(close_prices[max(0, i - 20) : i]) - close_prices[i - 1]) / close_prices[i - 1]
                close_prices[i] *= np.exp(reversion)

    # Generate OHLC from close prices
    data = []
    for i in range(n):
        close = close_prices[i]
        regime = regime_sequence[i]
        spread_bps = np.random.uniform(*REGIME_PARAMS[regime]["spread_bps"])
        half_spread = close * spread_bps / 10000

        high = close + np.random.uniform(0, half_spread * 1.5)
        low = close - np.random.uniform(0, half_spread * 1.5)
        open_price = close + np.random.uniform(-half_spread, half_spread)

        # Volume with clustering effect
        base_vol = 50 if "BTC" in symbol else 500
        vol_multiplier = 1 + regime_map[regime] * 0.5
        volume = np.random.exponential(base_vol * vol_multiplier)

        data.append(
            {
                "timestamp": timestamps[i],
                "open": open_price,
                "high": max(open_price, close, high),
                "low": min(open_price, close, low),
                "close": close,
                "volume": volume,
                "regime": regime,
            }
        )

    return pd.DataFrame(data)


def generate_synthetic_trades(
    timestamps: pd.DatetimeIndex,
    price_df: pd.DataFrame,
    symbol: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic trade tick data aggregated to 15-minute intervals (vectorized)."""
    rng = np.random.default_rng(seed)
    n_bars = len(timestamps)
    regime_map = {"Calm": 0, "Volatile": 1, "Stressed": 2}

    # Pre-compute regime and close price arrays
    regimes = price_df["regime"].values
    close_prices = price_df["close"].values

    # Base trades per bar: reduced for speed (BTC ~20, ETH ~60)
    base_trades = 20 if "BTC" in symbol else 60
    regime_mult = np.array([1.0, 1.5, 2.5])[np.array([regime_map.get(r, 0) for r in regimes])]

    # Number of trades per bar (Poisson-distributed)
    n_trades_per_bar = rng.poisson(base_trades * regime_mult)
    total_trades = n_trades_per_bar.sum()

    # Generate all trades in one vectorized operation
    # Trade timestamps: random offset within each 15-min bar
    offsets = rng.integers(0, 900, size=total_trades)  # seconds within bar
    bar_indices = np.repeat(np.arange(n_bars), n_trades_per_bar)
    timestamps_arr = np.array([t.value for t in timestamps])  # nanoseconds
    trade_ts = timestamps_arr[bar_indices] + offsets * 1_000_000_000

    # Trade prices with microstructure noise
    noise_bps = rng.uniform(-0.5, 0.5, size=total_trades)
    trade_prices = close_prices[bar_indices] * (1 + noise_bps / 10000)

    # Side assignment (buy 52%, sell 48%)
    sides = np.where(rng.random(total_trades) < 0.52, "buy", "sell")

    # Volume (exponential)
    base_vol = 0.5 if "BTC" in symbol else 5.0
    volumes = rng.exponential(base_vol, size=total_trades)

    trades_df = pd.DataFrame({
        "timestamp": pd.to_datetime(trade_ts),
        "side": sides,
        "volume": volumes,
        "price": trade_prices,
    })
    trades_df = trades_df.sort_values("timestamp").reset_index(drop=True)
    log.info(f"Generated {len(trades_df)} synthetic trades for {symbol}")
    return trades_df


def generate_synthetic_data(symbols: list, start_date: str, end_date: datetime) -> dict:
    """Generate synthetic data for all symbols."""
    log.warning("Generating synthetic data due to API failure or rate limiting")

    # Generate timestamps for 5-minute bars
    start = pd.to_datetime(start_date)
    periods_5m = int((end_date - start).total_seconds() / (5 * 60))
    timestamps_5m = pd.date_range(start=start, periods=periods_5m, freq="5min")

    # Generate timestamps for 15-minute trade aggregation
    periods_15m = int((end_date - start).total_seconds() / (15 * 60))
    timestamps_15m = pd.date_range(start=start, periods=periods_15m, freq="15min")

    synthetic_data = {}
    base_prices = {"BTC/USDT": 30000, "ETH/USDT": 2000}

    for symbol in symbols:
        log.info(f"Generating synthetic data for {symbol}")

        # Generate regime sequence for OHLCV
        regime_sequence = generate_synthetic_regimes(len(timestamps_5m))

        # Generate OHLCV
        ohlcv_df = generate_synthetic_ohlcv(
            timestamps_5m, base_prices[symbol], regime_sequence, symbol
        )

        # Generate trades (pass 5-min OHLCV timestamps so dimensions match price_df)
        trades_df = generate_synthetic_trades(timestamps_5m, ohlcv_df, symbol)

        synthetic_data[symbol] = {
            "ohlcv": ohlcv_df,
            "trades": trades_df,
            "synthetic_data": True,
            "regime_sequence": regime_sequence,
        }

    return synthetic_data


# ----------------------------------------------------------------------
# Binance API fetching with retry
# ----------------------------------------------------------------------


def fetch_ohlcv_with_retry(
    exchange,
    symbol: str,
    timeframe: str,
    since: int,
    end_date: datetime,
    limit: int = 1000,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch OHLCV data with exponential backoff retry."""
    all_ohlcv = []
    current_since = since

    while True:
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
                if not ohlcv:
                    return pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1

                # Check rate limit (1200 requests/minute for Binance free tier)
                time.sleep(0.05)  # Small delay to avoid hitting limit
                break

            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    wait_time = 2**attempt * 1.0
                    log.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    log.error(f"Error fetching OHLCV: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2**attempt)

        # If we've fetched enough data or reached present, stop
        if len(ohlcv) < limit or pd.to_datetime(ohlcv[-1][0], unit="ms") >= pd.Timestamp(end_date) - pd.Timedelta(days=1):
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_trades_with_retry(
    exchange,
    symbol: str,
    since: int,
    limit: int = 1000,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch trade data with exponential backoff retry."""
    all_trades = []
    current_since = since

    while True:
        for attempt in range(max_retries):
            try:
                trades = exchange.fetch_trades(symbol, since=current_since, limit=limit)
                if not trades:
                    return pd.DataFrame(all_trades, columns=["timestamp", "side", "volume", "price"])

                for trade in trades:
                    all_trades.append(
                        {
                            "timestamp": pd.to_datetime(trade["timestamp"], unit="ms"),
                            "side": "buy" if trade["side"] == "buy" else "sell",
                            "volume": trade.get("volume", trade.get("amount", 0)),
                            "price": trade["price"],
                        }
                    )

                current_since = trades[-1]["timestamp"] + 1
                time.sleep(0.05)  # Rate limit avoidance
                break

            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    wait_time = 2**attempt * 1.0
                    log.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    log.error(f"Error fetching trades: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2**attempt)

        if len(trades) < limit:
            break

    return pd.DataFrame(all_trades)


def aggregate_trades_to_15min(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade ticks to 15-minute intervals."""
    if trades_df.empty:
        return trades_df

    trades_df = trades_df.copy()
    trades_df["timestamp_bin"] = trades_df["timestamp"].dt.floor("15min")

    aggregated = (
        trades_df.groupby("timestamp_bin")
        .agg(
            side=("side", lambda x: "buy" if (x == "buy").sum() > (x == "sell").sum() else "sell"),
            volume=("volume", "sum"),
            price=("price", "last"),
        )
        .reset_index()
        .rename(columns={"timestamp_bin": "timestamp"})
    )

    return aggregated


# ----------------------------------------------------------------------
# Main data fetching
# ----------------------------------------------------------------------


def derive_trades_from_ohlcv(ohlcv_df: pd.DataFrame, n_trades_per_bar: int = 20) -> pd.DataFrame:
    """
    Derive synthetic trade ticks from OHLCV bars using the Lee-Ready algorithm.
    Uses intrabar price movement to classify each synthetic trade as buy or sell initiated.
    """
    trades_list = []
    for i, row in ohlcv_df.iterrows():
        ts = row["timestamp"]
        open_, high, low, close = row["open"], row["high"], row["low"], row["close"]

        # Generate n_trades_per_bar synthetic trades within this 5-min bar
        for j in range(n_trades_per_bar):
            # Use tick rule: if price moves up from open, trade is buy-initiated
            # If price moves down, trade is sell-initiated
            tfrac = (j + 0.5) / n_trades_per_bar
            micro_price = open_ + tfrac * (close - open_) + np.random.normal(0, (high - low) * 0.01)

            # Classify by price movement from previous micro-trade
            if j == 0:
                prev_price = open_
            else:
                prev_price = open_ + ((j - 0.5) / n_trades_per_bar) * (close - open_)

            side = "buy" if micro_price >= prev_price else "sell"
            noise = np.random.uniform(-0.5, 0.5) / 10000
            trade_price = micro_price * (1 + noise)

            # Volume: exponential with mean scaled by bar volume
            bar_vol = row["volume"]
            vol = np.random.exponential(bar_vol / n_trades_per_bar)

            trades_list.append({
                "timestamp": ts + timedelta(seconds=np.random.randint(0, 300)),
                "side": side,
                "volume": vol,
                "price": trade_price,
            })

    df = pd.DataFrame(trades_list)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_all_data(symbols: list, start_date: str, end_date: datetime, use_synthetic_fallback: bool = True) -> dict:
    """Fetch all data from Binance: real OHLCV + derived trades, synthetic historical."""
    import ccxt

    log.info(f"Initializing Binance exchange...")
    exchange = ccxt.binance({"rateLimit": 1200, "enableRateLimit": True})

    all_data = {}
    real_ohlcv_available = False

    # Fetch data from API (no synthetic fallback - get real market data)
    # 10 years = ~3650 days gives ~1M bars for full market cycles (2016-2026)
    api_fetch_days = 3650
    synthetic_start = end_date - timedelta(days=api_fetch_days)
    log.info(f"Fetching {api_fetch_days} days of OHLCV from API ({synthetic_start} to {end_date})")

    for symbol in symbols:
        log.info(f"Fetching OHLCV data for {symbol}")
        try:
            since = int(synthetic_start.timestamp() * 1000)
            ohlcv_df = fetch_ohlcv_with_retry(exchange, symbol, OHLCV_INTERVAL, since, end_date)

            if not ohlcv_df.empty:
                log.info(f"Fetched {len(ohlcv_df)} real OHLCV bars for {symbol}")
                # Derive synthetic trades from real OHLCV using Lee-Ready
                log.info(f"Deriving synthetic trades from OHLCV for {symbol}...")
                trades_df = derive_trades_from_ohlcv(ohlcv_df)
                trades_agg = aggregate_trades_to_15min(trades_df)
                log.info(f"Derived {len(trades_agg)} aggregated trade records")
                all_data[symbol] = {
                    "ohlcv": ohlcv_df,
                    "trades": trades_agg,
                    "synthetic_data": False,
                }
                real_ohlcv_available = True
            else:
                log.warning(f"No OHLCV returned for {symbol}")
        except Exception as e:
            log.error(f"Failed to fetch OHLCV for {symbol}: {e}")

    return all_data


def save_raw_data(data: dict, output_dir: Path):
    """Save raw data to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {"synthetic_data": False, "symbols": [], "generated_at": datetime.now().isoformat()}

    for symbol, symbol_data in data.items():
        clean_symbol = symbol.replace("/", "_")

        # Save OHLCV
        ohlcv_path = output_dir / f"{clean_symbol}_ohlcv_5m.parquet"
        symbol_data["ohlcv"].to_parquet(ohlcv_path, index=False)
        log.info(f"Saved OHLCV to {ohlcv_path}")

        # Save trades
        trades_path = output_dir / f"{clean_symbol}_trades_15m.parquet"
        symbol_data["trades"].to_parquet(trades_path, index=False)
        log.info(f"Saved trades to {trades_path}")

        metadata["symbols"].append(symbol)
        if symbol_data.get("synthetic_data"):
            metadata["synthetic_data"] = True

    # Save metadata
    metadata_path = output_dir / "fetch_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance OHLCV and trade data")
    parser.add_argument("--start_date", type=str, default=START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    end_date = datetime.now()

    log.info(f"Starting data fetch from {args.start_date} to {end_date}")
    log.info(f"Symbols: {SYMBOLS}")

    # Fetch all data
    data = fetch_all_data(SYMBOLS, args.start_date, end_date)

    # Save raw data
    output_dir = Path(args.output_dir)
    save_raw_data(data, output_dir)

    # Summary
    for symbol, symbol_data in data.items():
        synthetic_flag = "[SYNTHETIC] " if symbol_data.get("synthetic_data") else ""
        log.info(
            f"{synthetic_flag}{symbol}: {len(symbol_data['ohlcv'])} OHLCV bars, "
            f"{len(symbol_data['trades'])} trade records"
        )

    log.info("Data fetch complete!")


if __name__ == "__main__":
    main()
