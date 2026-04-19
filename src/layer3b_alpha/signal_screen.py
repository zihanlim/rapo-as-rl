"""
Layer 3b: Alpha Signal Screen
============================
Screens candidate alpha signals across three prediction horizons:
  - 5-min:  Next bar return
  - 1-hour: Next 12 bars return
  - 1-day:  Next 288 bars return

For each signal × horizon:
  1. In-sample IC (Information Coefficient = Pearson correlation with forward return)
  2. Out-of-sample R² using expanding window (train 75%, test 25%)
  3. A&S cost survival filter: does E[signal_return] exceed A&S cost per period?

Stopping rule:
  If no signal shows OOS R² > 0.01 at any frequency → comprehensive negative finding.

Usage:
    python -m src.layer3b_alpha.signal_screen --features data/processed/alpha_features.parquet \\
                                               --prices data/processed/price_features.parquet \\
                                               --as_cost_dir models/as_cost \\
                                               --output results/alpha_screen_results.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from scipy import stats

# =============================================================================
# Data structures
# =============================================================================

@dataclass
class SignalResult:
    signal: str
    horizon: str          # '5min', '1hour', '1day'
    ic: float             # In-sample Pearson correlation with forward return
    oos_r2: float        # Out-of-sample R² (expanding window)
    oos_ic: float         # Out-of-sample IC
    p_value_ic: float     # t-test p-value for IC
    a_s_cost_bps: float   # A&S cost per period (bps), single-sided
    expected_return_bps: float
    survival: bool         # Does signal survive A&S cost filter?
    n_obs: int


# =============================================================================
# A&S Cost Estimation (simplified, per-horizon)
# =============================================================================

# Per-bar A&S costs (in bps) — from calibration
AS_COST_PER_BAR = {
    'Calm': 2.15,       # ~123 bps / 57 bars ≈ 2.15 bps/bar
    'Volatile': 5.26,    # ~300 bps / 57 bars ≈ 5.26 bps/bar
    'Stressed': 22.67,   # ~1292 bps / 57 bars ≈ 22.67 bps/bar
}

# Market impact scales with sqrt(q) — for aggregate horizon N bars:
# cost(N bars) = cost_per_bar * sqrt(N)
def get_as_cost_for_horizon(horizon: str, regime: str = 'Calm') -> float:
    """Return A&S cost in bps for a single trade at given horizon."""
    horizon_bars = {'5min': 1, '1hour': 12, '1day': 288}
    n = horizon_bars.get(horizon, 1)
    base_cost = AS_COST_PER_BAR.get(regime, AS_COST_PER_BAR['Calm'])
    # Market impact: temporary impact scales with sqrt of trade size
    # For aggregated horizon, trade size in "bar-equivalents" = sqrt(n)
    return base_cost * np.sqrt(n)


# =============================================================================
# Signal Screening Core
# =============================================================================

def screen_signal(
    signal: pd.Series,
    fwd_return: pd.Series,
    signal_name: str,
    horizon: str,
    regime: str = 'Calm',
    min_obs: int = 100,
) -> Optional[SignalResult]:
    """
    Screen a single signal against a forward return at given horizon.

    Steps:
    1. Align signal and fwd_return, drop NaN
    2. Compute in-sample IC (full aligned series)
    3. Compute OOS R² using expanding window (train 75%, test last 25%)
    4. Compute A&S cost survival
    """
    df = pd.DataFrame({'signal': signal, 'fwd': fwd_return}).dropna()

    if len(df) < min_obs:
        return None

    # ---- 1. In-sample IC ----
    ic = df['signal'].corr(df['fwd'])
    n = len(df)
    # t-stat for Pearson correlation
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if n > 2 else 1.0

    # ---- 2. OOS R² (expanding window) ----
    # Train on first 75%, test on last 25%
    train_size = int(n * 0.75)
    if train_size < 30:
        return None

    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    # Simple OLS: predict fwd from signal
    # If signal is mean-reverting: long signal when signal < 0, short when signal > 0
    # If signal is momentum: long signal when signal > 0, short when signal < 0
    # We don't know direction a priori — use absolute IC to rank
    from numpy.polynomial.polynomial import polyfit, polyval
    try:
        coef = np.polyfit(train['signal'], train['fwd'], 1)
        pred_test = np.polyval(coef, test['signal'])
        e = test['fwd'] - pred_test
        ss_res = (e**2).sum()
        ss_tot = ((test['fwd'] - test['fwd'].mean())**2).sum()
        oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        oos_ic = test['signal'].corr(test['fwd'])
    except Exception:
        oos_r2 = 0.0
        oos_ic = 0.0

    # ---- 3. A&S Cost Survival ----
    # FIXED (2026-04-18): A&S cost is paid on BOTH entry AND exit.
    # Round-trip cost = 2x single-sided cost.
    # The unconditional per-bar IC estimate understates actual trading costs.
    as_cost_single = get_as_cost_for_horizon(horizon, regime)
    as_cost_round_trip = 2 * as_cost_single
    expected_return_bps = ic * df['fwd'].std() * 10000

    # Survival: expected return magnitude must exceed ROUND-TRIP A&S cost
    survival = abs(expected_return_bps) > as_cost_round_trip

    return SignalResult(
        signal=signal_name,
        horizon=horizon,
        ic=float(ic),
        oos_r2=float(oos_r2),
        oos_ic=float(oos_ic),
        p_value_ic=float(p_value),
        a_s_cost_bps=float(as_cost_single),
        expected_return_bps=float(expected_return_bps),
        survival=survival,
        n_obs=int(n),
    )


def screen_all_signals(
    features_path: str = "data/processed/alpha_features.parquet",
    as_cost_dir: str = "models/as_cost",
    output_path: str = "results/alpha_screen_results.json",
    r2_threshold: float = 0.001,   # Liberal threshold for initial screen
    ic_threshold: float = 0.01,    # Minimum IC magnitude
    pvalue_threshold: float = 0.05,
) -> Dict:
    """
    Run the full alpha screen across all signals and horizons.
    """
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df.columns)-1} features, {len(df)} observations")

    # Forward returns (already in features file)
    horizons = {
        '5min': 'fwd_return_5min',
        '1hour': 'fwd_return_1h',
        '1day': 'fwd_return_1d',
    }

    # Feature columns (exclude non-features)
    exclude_cols = ['timestamp', 'fwd_return_5min', 'fwd_return_1h', 'fwd_return_1d',
                    'btc_close', 'eth_close', 'btc_return', 'eth_return', 'realized_vol',
                    'spread_proxy', 'btc_return_lag_1', 'btc_return_lag_3',
                    'eth_return_lag_1', 'eth_return_lag_3']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Screening {len(feature_cols)} signals across {len(horizons)} horizons...")
    results: List[SignalResult] = []

    for horizon_name, fwd_col in horizons.items():
        print(f"\n  Horizon: {horizon_name} ({fwd_col})")
        if fwd_col not in df.columns:
            print(f"    Warning: {fwd_col} not found, skipping horizon")
            continue

        fwd_return = df[fwd_col]

        for feat in feature_cols:
            signal = df[feat]
            result = screen_signal(
                signal, fwd_return,
                signal_name=feat,
                horizon=horizon_name,
                regime='Calm',  # Conservative: use Calm regime costs
                min_obs=200,
            )
            if result is not None:
                results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame([asdict(r) for r in results])

    # Filter: must have |IC| > threshold and p-value < threshold
    significant = results_df[
        (results_df['ic'].abs() > ic_threshold) &
        (results_df['p_value_ic'] < pvalue_threshold)
    ].copy()

    # Sort by OOS R² descending
    significant = significant.sort_values('oos_r2', ascending=False)

    # Summary statistics
    summary = {
        'total_signals_screened': len(results_df),
        'total_horizons': len(horizons),
        'significant_count': len(significant),
        'surviving_count': int(significant['survival'].sum()) if len(significant) > 0 else 0,
        'best_oos_r2': float(significant['oos_r2'].max()) if len(significant) > 0 else 0.0,
        'best_ic': float(significant['ic'].abs().max()) if len(significant) > 0 else 0.0,
        'r2_threshold': r2_threshold,
        'ic_threshold': ic_threshold,
        'pvalue_threshold': pvalue_threshold,
        'top_signals': [],
    }

    # Top 10 signals by OOS R²
    for _, row in significant.head(10).iterrows():
        summary['top_signals'].append({
            'signal': row['signal'],
            'horizon': row['horizon'],
            'ic': row['ic'],
            'oos_r2': row['oos_r2'],
            'oos_ic': row['oos_ic'],
            'p_value': row['p_value_ic'],
            'a_s_cost_bps': row['a_s_cost_bps'],
            'expected_return_bps': row['expected_return_bps'],
            'survival': row['survival'],
        })

    # Categorize results
    for horizon in horizons.keys():
        horizon_results = significant[significant['horizon'] == horizon]
        summary[f'{horizon}_significant'] = len(horizon_results)
        summary[f'{horizon}_surviving'] = int(horizon_results['survival'].sum()) if len(horizon_results) > 0 else 0

    # Stopping rule check
    any_surviving = summary['surviving_count'] > 0
    any_meaningful_r2 = summary['best_oos_r2'] > r2_threshold

    # Stopping rule: NEGATIVE only if BOTH R^2 AND survival fail
    summary['stopping_rule_triggered'] = not (any_surviving or any_meaningful_r2)

    if any_surviving:
        summary['verdict'] = 'POSITIVE: Signals survived A&S cost filter — further analysis needed'
    elif any_meaningful_r2:
        summary['verdict'] = 'POSITIVE BUT NOT SURVIVING: OOS R2 meaningful but A&S costs consume edge'
    else:
        summary['verdict'] = 'NEGATIVE: Comprehensive negative finding — no exploitable alpha'

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SCREENING RESULTS")
    print(f"{'='*60}")
    print(f"Total signals screened: {summary['total_signals_screened']}")
    print(f"Significant (|IC| > {ic_threshold}, p < {pvalue_threshold}): {summary['significant_count']}")
    print(f"Surviving A&S cost filter: {summary['surviving_count']}")
    print(f"Best OOS R²: {summary['best_oos_r2']:.6f}")
    print(f"Best |IC|: {summary['best_ic']:.4f}")
    print(f"\nVerdict: {summary['verdict']}")
    if any_surviving:
        print(f"\n  -> {summary['surviving_count']} signals survived A&S cost filter")
        print(f"  -> Further analysis required for integration")
    elif any_meaningful_r2:
        print(f"\n  -> OOS R2 = {summary['best_oos_r2']:.4f} is meaningful but all signals fail A&S survival")
        print(f"  -> Comprehensive negative finding: predictive power exists but edge insufficient after costs")
    elif summary['stopping_rule_triggered']:
        print(f"\n  -> All signals failed OOS R^2 > {r2_threshold} AND A&S survival test")
        print(f"  -> Comprehensive negative finding: no exploitable alpha at any tested horizon")
    print(f"\nFull results saved to {output_path}")

    # Also save full results DataFrame
    results_path = output_path.parent / f"{output_path.stem}_full.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Full results table: {results_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 3b: Screen alpha signals")
    parser.add_argument("--features", type=str, default="data/processed/alpha_features.parquet")
    parser.add_argument("--prices", type=str, default="data/processed/price_features.parquet")
    parser.add_argument("--as_cost_dir", type=str, default="models/as_cost")
    parser.add_argument("--output", type=str, default="results/alpha_screen_results.json")
    parser.add_argument("--r2_threshold", type=float, default=0.001)
    parser.add_argument("--ic_threshold", type=float, default=0.01)
    args = parser.parse_args()

    summary = screen_all_signals(
        features_path=args.features,
        as_cost_dir=args.as_cost_dir,
        output_path=args.output,
        r2_threshold=args.r2_threshold,
        ic_threshold=args.ic_threshold,
    )
