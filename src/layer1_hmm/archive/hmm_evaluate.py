"""
Layer 1: HMM Regime Classifier — Evaluation and Validation

Validates HMM regime assignments against known market events
(e.g., Luna collapse, FTX collapse, COVID crash).

Usage:
    python -m src.layer1_hmm.hmm_evaluate --model models/hmm/hmm_model.pkl --data data/processed
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def plot_regime_returns(
    returns: pd.Series,
    regimes: pd.Series,
    title: str = "HMM Regime Classification vs Returns",
    save_path: str = None,
):
    """
    Plot cumulative returns coloured by regime.
    Verifies economic intuition: Stressed = high vol, Calm = low vol.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Cumulative returns
    cumret = (1 + returns).cumprod() - 1
    colors = {"Calm": "green", "Volatile": "orange", "Stressed": "red"}
    for regime, color in colors.items():
        mask = regimes == regime
        if mask.sum() == 0:
            continue
        axes[0].scatter(
            cumret.index[mask],
            cumret.values[mask],
            c=color,
            s=2,
            alpha=0.5,
            label=regime,
        )
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend(title="Regime")
    axes[0].set_title(title)

    # Panel 2: Regime over time
    regime_map = {"Calm": 0, "Volatile": 1, "Stressed": 2}
    regime_numeric = regimes.map(regime_map)
    axes[1].fill_between(
        regime_numeric.index,
        regime_numeric.values,
        step="pre",
        alpha=0.4,
        color="steelblue",
    )
    axes[1].set_ylabel("Regime")
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Calm", "Volatile", "Stressed"])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def validate_against_events(
    regimes: pd.Series,
    returns: pd.Series,
    events: dict[str, str] = None,
) -> pd.DataFrame:
    """
    Validate that HMM regime assignments align with known market events.

    Parameters
    ----------
    regimes : pd.Series
        Regime label time series
    returns : pd.Series
        5-min return series
    events : dict
        Mapping of event name to approximate date string
        e.g. {"FTX Collapse": "2022-11-09", "Luna Collapse": "2022-05-09"}

    Returns
    -------
    pd.DataFrame
        Summary table of average realized volatility per regime
    """
    if events is None:
        events = {
            "Luna Collapse": "2022-05-09",
            "FTX Collapse": "2022-11-09",
            "COVID Crash": "2020-03-12",
        }

    results = []
    for event_name, date_str in events.items():
        event_date = pd.to_datetime(date_str)
        window = regimes.loc[event_date : event_date + pd.Timedelta(days=7)]
        ret_window = returns.loc[event_date : event_date + pd.Timedelta(days=7)]
        if len(window) == 0:
            continue
        dominant = window.value_counts().index[0]
        avg_vol = ret_window.std() * np.sqrt(288)  # annualize 5-min vol
        results.append(
            {
                "Event": event_name,
                "Dominant Regime": dominant,
                "Avg Annualized Vol (bps/day)": round(avg_vol * 10000, 1),
                "N Periods in Window": len(window),
            }
        )
        print(f"{event_name}: Dominant regime = {dominant}, Ann.Vol = {avg_vol*10000:.1f} bps/day")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Layer 1: HMM Evaluation")
    parser.add_argument("--model", type=str, default="models/hmm/hmm_model.pkl")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="models/hmm/figures")
    args = parser.parse_args()

    model = joblib.load(args.model)
    price_df = pd.read_parquet(Path(args.data) / "price_features.parquet")

    returns = np.log(price_df["close"] / price_df["close"].shift(1)).dropna()

    # Build features (same as training)
    from src.layer1_hmm.hmm_train import build_hmm_features

    trades_df = pd.read_parquet(Path(args.data) / "trades_processed.parquet")
    features = build_hmm_features(price_df, trades_df)
    hidden_states = model.predict(features.values)
    regimes = pd.Series(hidden_states).map(model.state_labels_)
    regimes.index = features.index

    # Validate against known events
    print("\n=== Event-Based Validation ===")
    event_results = validate_against_events(regimes, returns)

    # Plot
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_regime_returns(
        returns,
        regimes,
        save_path=output_dir / "hmm_regime_returns.png",
    )

    event_results.to_csv(output_dir / "event_validation.csv", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
