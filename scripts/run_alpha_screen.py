"""
Run Alpha Screen — CLI Driver
=============================
Two-step pipeline:
  Step 1: Compute all alpha features from existing data
  Step 2: Screen features for predictive power + A&S cost survival

Usage:
    # Full run
    python scripts/run_alpha_screen.py

    # Feature computation only
    python scripts/run_alpha_screen.py --step features

    # Screening only (requires features already computed)
    python scripts/run_alpha_screen.py --step screen

    # Custom thresholds
    python scripts/run_alpha_screen.py --r2_threshold 0.005 --ic_threshold 0.02
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_step(step: str, **kwargs):
    """Run a single pipeline step as a subprocess."""
    cmd = []
    if sys.executable:
        cmd = [sys.executable]

    if step == "features":
        cmd += ["-m", "src.layer3b_alpha.feature_library"]
        cmd += ["--prices", kwargs.get("prices", "data/processed/price_features.parquet")]
        cmd += ["--trades", kwargs.get("trades", "data/processed/trades_processed.parquet")]
        cmd += ["--output", kwargs.get("features_output", "data/processed/alpha_features.parquet")]
        print(f"\n[Step 1/2] Computing alpha features...")
        print(f"Command: {' '.join(cmd)}")

    elif step == "screen":
        cmd += ["-m", "src.layer3b_alpha.signal_screen"]
        cmd += ["--features", kwargs.get("features", "data/processed/alpha_features.parquet")]
        cmd += ["--output", kwargs.get("screen_output", "results/alpha_screen_results.json")]
        cmd += ["--r2_threshold", str(kwargs.get("r2_threshold", 0.001))]
        cmd += ["--ic_threshold", str(kwargs.get("ic_threshold", 0.01))]
        print(f"\n[Step 2/2] Screening signals...")
        print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run alpha feature library + screen pipeline")
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "features", "screen"],
        default="all",
        help="'all' = compute features then screen; 'features' = features only; 'screen' = screening only"
    )
    parser.add_argument("--prices", type=str, default="data/processed/price_features.parquet")
    parser.add_argument("--trades", type=str, default="data/processed/trades_processed.parquet")
    parser.add_argument("--features_output", type=str, default="data/processed/alpha_features.parquet")
    parser.add_argument("--screen_output", type=str, default="results/alpha_screen_results.json")
    parser.add_argument("--r2_threshold", type=float, default=0.001)
    parser.add_argument("--ic_threshold", type=float, default=0.01)
    args = parser.parse_args()

    if args.step in ("all", "features"):
        ok = run_step("features",
                      prices=args.prices,
                      trades=args.trades,
                      features_output=args.features_output)
        if not ok:
            print("Feature computation failed!")
            return

    if args.step in ("all", "screen"):
        # Verify features exist
        features_path = Path(args.features_output)
        if not features_path.exists():
            print(f"Error: Features file not found at {features_path}")
            print("Run with --step all or --step features first")
            return

        ok = run_step("screen",
                      features=args.features_output,
                      screen_output=args.screen_output,
                      r2_threshold=args.r2_threshold,
                      ic_threshold=args.ic_threshold)
        if not ok:
            print("Signal screening failed!")
            return

    print(f"\n{'='*60}")
    print("ALPHA SCREEN PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Features: {args.features_output}")
    print(f"Results:  {args.screen_output}")


if __name__ == "__main__":
    main()
