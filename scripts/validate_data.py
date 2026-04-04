"""
Data Validation Layer 0: Quality Checks and Statistics

Validates processed data quality and reports statistics.

Usage:
    python scripts/validate_data.py
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
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/validate_data.log")],
)
log = logging.getLogger(__name__)

START_DATE = "2021-01-01"


def check_file_exists(path: Path, name: str) -> bool:
    """Check if a file exists and log result."""
    if path.exists():
        log.info(f"Found {name}: {path}")
        return True
    else:
        log.error(f"Missing {name}: {path}")
        return False


def validate_date_range(df: pd.DataFrame, name: str, min_date: str = START_DATE) -> bool:
    """Validate that data spans expected date range."""
    if df.empty:
        log.error(f"{name}: DataFrame is empty")
        return False

    min_dt = pd.to_datetime(min_date)
    actual_min = df["timestamp"].min()
    actual_max = df["timestamp"].max()

    log.info(f"{name} date range: {actual_min} to {actual_max}")

    if actual_min > min_dt:
        log.warning(f"{name}: Data starts after {min_date} (starts at {actual_min})")
        return False

    return True


def check_nulls(df: pd.DataFrame, name: str) -> dict:
    """Check for null values and NaN patterns."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)

    result = {
        "total_rows": len(df),
        "null_counts": null_counts.to_dict(),
        "null_pct": null_pct.to_dict(),
    }

    log.info(f"{name} null analysis:")
    log.info(f"  Total rows: {len(df)}")
    for col in df.columns:
        if null_counts[col] > 0:
            log.warning(f"  {col}: {null_counts[col]} nulls ({null_pct[col]}%)")

    # Check for NaN patterns specifically
    nan_counts = df.isna().sum()
    if nan_counts.any():
        log.warning(f"{name} contains NaN values in: {nan_counts[nan_counts > 0].to_dict()}")

    return result


def compute_return_stats(df: pd.DataFrame) -> dict:
    """Compute return statistics."""
    stats = {}

    for col in ["btc_return", "eth_return"]:
        if col in df.columns:
            returns = df[col].dropna()
            if len(returns) > 0:
                stats[col] = {
                    "mean": float(returns.mean()),
                    "std": float(returns.std()),
                    "min": float(returns.min()),
                    "max": float(returns.max()),
                    "skew": float(returns.skew()),
                    "kurtosis": float(returns.kurtosis()),
                }

                # Annualized volatility
                annualized_vol = returns.std() * np.sqrt(252 * 24 * 12)  # 5-min bars
                stats[col]["annualized_vol"] = float(annualized_vol)

    return stats


def compute_volatility_regimes(df: pd.DataFrame) -> dict:
    """Detect and characterize volatility regimes."""
    if "realized_vol" not in df.columns:
        return {}

    vol = df["realized_vol"].dropna()
    if len(vol) == 0:
        return {}

    # Define regime thresholds based on percentiles
    q33 = vol.quantile(0.33)
    q67 = vol.quantile(0.67)

    regimes = {
        "Calm": (vol <= q33).sum(),
        "Volatile": ((vol > q33) & (vol <= q67)).sum(),
        "Stressed": (vol > q67).sum(),
    }

    regime_stats = {
        "Calm": {"count": regimes["Calm"], "pct": regimes["Calm"] / len(vol) * 100},
        "Volatile": {"count": regimes["Volatile"], "pct": regimes["Volatile"] / len(vol) * 100},
        "Stressed": {"count": regimes["Stressed"], "pct": regimes["Stressed"] / len(vol) * 100},
        "thresholds": {"q33": float(q33), "q67": float(q67)},
    }

    log.info(f"Volatility regime distribution:")
    for regime, info in regime_stats.items():
        if regime != "thresholds":
            log.info(f"  {regime}: {info['count']} ({info['pct']:.1f}%)")

    return regime_stats


def compute_correlation(df: pd.DataFrame) -> dict:
    """Compute BTC/ETH return correlation."""
    if "btc_return" not in df.columns or "eth_return" not in df.columns:
        return {}

    valid = df[["btc_return", "eth_return"]].dropna()
    if len(valid) == 0:
        return {}

    corr = valid["btc_return"].corr(valid["eth_return"])
    log.info(f"BTC/ETH return correlation: {corr:.4f}")

    # Correlation by regime
    if "realized_vol" in df.columns:
        correlations_by_regime = {}
        vol = df["realized_vol"]
        q33 = vol.quantile(0.33)
        q67 = vol.quantile(0.67)

        for regime, mask in [
            ("Calm", vol <= q33),
            ("Volatile", (vol > q33) & (vol <= q67)),
            ("Stressed", vol > q67),
        ]:
            reg_data = valid.loc[mask]
            if len(reg_data) > 10:
                correlations_by_regime[regime] = float(reg_data["btc_return"].corr(reg_data["eth_return"]))

        return {"overall": float(corr), "by_regime": correlations_by_regime}

    return {"overall": float(corr)}


def main():
    parser = argparse.ArgumentParser(description="Validate processed Binance data")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    log.info("=" * 60)
    log.info("DATA VALIDATION REPORT")
    log.info("=" * 60)

    validation_results = {"validated_at": datetime.now().isoformat(), "checks": {}, "stats": {}}

    # Check files exist
    price_path = data_dir / "price_features.parquet"
    trades_path = data_dir / "trades_processed.parquet"
    metadata_path = data_dir / "processing_metadata.json"

    price_exists = check_file_exists(price_path, "price_features.parquet")
    trades_exists = check_file_exists(trades_path, "trades_processed.parquet")

    if not price_exists:
        log.error("price_features.parquet not found. Run process_data.py first.")
        validation_results["checks"]["files"] = "FAILED"
        return

    # Load data
    log.info("Loading processed data...")
    price_df = pd.read_parquet(price_path)
    trades_df = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    # Load metadata
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        log.info(f"Data source: {'SYNTHETIC' if metadata.get('synthetic_data') else 'REAL Binance API'}")
        validation_results["synthetic_data"] = metadata.get("synthetic_data", False)

    # Run validations
    log.info("\n--- File Existence Checks ---")
    validation_results["checks"]["files"] = "PASSED" if (price_exists and trades_exists) else "FAILED"

    log.info("\n--- Date Range Checks ---")
    price_date_check = validate_date_range(price_df, "price_features")
    trades_date_check = validate_date_range(trades_df, "trades_processed")
    validation_results["checks"]["date_range"] = "PASSED" if price_date_check else "PARTIAL"

    log.info("\n--- Null Value Checks ---")
    price_nulls = check_nulls(price_df, "price_features")
    trades_nulls = check_nulls(trades_df, "trades_processed")
    validation_results["checks"]["nulls"] = (
        "PASSED" if (sum(price_nulls["null_counts"].values()) == 0) else "WARNING"
    )
    validation_results["stats"]["price_nulls"] = price_nulls
    validation_results["stats"]["trades_nulls"] = trades_nulls

    log.info("\n--- Return Statistics ---")
    return_stats = compute_return_stats(price_df)
    validation_results["stats"]["returns"] = return_stats

    for col, stats in return_stats.items():
        log.info(f"{col}:")
        log.info(f"  Mean: {stats['mean']:.6f}")
        log.info(f"  Std: {stats['std']:.6f}")
        log.info(f"  Annualized Vol: {stats.get('annualized_vol', 'N/A'):.4f}")

    log.info("\n--- Volatility Regime Analysis ---")
    regime_stats = compute_volatility_regimes(price_df)
    validation_results["stats"]["regimes"] = regime_stats

    log.info("\n--- Correlation Analysis ---")
    corr_stats = compute_correlation(price_df)
    validation_results["stats"]["correlation"] = corr_stats

    # Overall summary
    log.info("\n" + "=" * 60)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 60)

    all_passed = all(v in ["PASSED", "WARNING"] for v in validation_results["checks"].values())

    if all_passed:
        log.info("All checks PASSED")
        validation_results["overall"] = "PASSED"
    else:
        log.error("Some checks FAILED - review logs above")
        validation_results["overall"] = "FAILED"

    log.info(f"Total price records: {len(price_df)}")
    log.info(f"Total trade records: {len(trades_df)}")
    log.info(f"Date range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")

    # Save validation report (convert numpy types to native Python for JSON serialization)
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(x) for x in obj]
        elif hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        else:
            return obj

    validation_results = convert_to_native(validation_results)
    report_path = data_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    log.info(f"\nValidation report saved to {report_path}")

    return validation_results


if __name__ == "__main__":
    main()
