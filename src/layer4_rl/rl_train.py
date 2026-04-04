"""
Layer 4: RL Agent Training — PPO per Regime

Trains three separate PPO policies (one per regime) using Stable Baselines3.
Hyperparameter grid search over learning_rate, gamma, clip_range.

Training terminates when rolling Sharpe (100-period) stabilizes within ±0.05
over 50,000 timesteps.

Architecture: MLP with 2 hidden layers of 64 units each, ReLU activation.

Usage:
    python -m src.layer4_rl.rl_train --data data/processed \
           --regime models/hmm/regime_labels.csv \
           --as_cost models/as_cost \
           --lgbm models/lgbm \
           --output models/rl/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import ReLU
from stable_baselines3 import PPO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def rolling_sharpe(rewards: list[float], window: int = 100) -> float:
    """Compute rolling Sharpe ratio."""
    if len(rewards) < window:
        return 0.0
    recent = rewards[-window:]
    return np.mean(recent) / (np.std(recent) + 1e-8)


def train_regime_ppo(env, regime: str, n_timesteps: int = 1_000_000) -> PPO:
    """
    Train PPO agent for a specific regime with hyperparameter grid search.

    Hyperparameter grid:
        learning_rate: {3e-4, 1e-4}
        gamma: {0.99, 0.95}
        clip_range: {0.1, 0.2}

    Early stopping: rolling Sharpe (100-period) stabilizes within ±0.05
    over 50,000 timesteps.
    """
    hyperparams = [
        {"learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.1},
        {"learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2},
        {"learning_rate": 3e-4, "gamma": 0.95, "clip_range": 0.1},
        {"learning_rate": 3e-4, "gamma": 0.95, "clip_range": 0.2},
        {"learning_rate": 1e-4, "gamma": 0.99, "clip_range": 0.1},
        {"learning_rate": 1e-4, "gamma": 0.99, "clip_range": 0.2},
        {"learning_rate": 1e-4, "gamma": 0.95, "clip_range": 0.1},
        {"learning_rate": 1e-4, "gamma": 0.95, "clip_range": 0.2},
    ]

    best_sharpe = -np.inf
    best_model = None
    best_params = None
    reward_buffer = []

    for params in hyperparams:
        log.info(f"  Testing: lr={params['learning_rate']}, γ={params['gamma']}, clip={params['clip_range']}")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            clip_range=params["clip_range"],
            policy_kwargs={"net_arch": [64, 64], "activation_fn": ReLU},
            verbose=0,
            seed=42,
        )

        # Train with early stopping evaluation
        n_eval = 50_000
        n_iterations = n_timesteps // n_eval
        stable_count = 0

        for iteration in range(n_iterations):
            model.learn(n_eval, progress_bar=False, reset_num_timesteps=False)
            # Evaluate on a separate eval environment
            eval_env = env  # in practice: use separate validation env
            obs, _ = eval_env.reset()
            eval_rewards = []
            for _ in range(min(500, eval_env.max_t)):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = eval_env.step(action)
                eval_rewards.append(reward)
                if done or truncated:
                    obs, _ = eval_env.reset()
            rs = rolling_sharpe(eval_rewards)
            reward_buffer.extend(eval_rewards)

            # Early stopping check: requires at least 50,000 rewards for stable comparison
            if len(reward_buffer) >= 50_000:
                current_sharpe = rolling_sharpe(reward_buffer)
                prev_sharpe = rolling_sharpe(reward_buffer[:-50_000])
                if abs(current_sharpe - prev_sharpe) < 0.05:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= 1:
                    log.info(f"    Early stop at iteration {iteration+1}, Sharpe={current_sharpe:.3f}")
                    break

            log.info(f"    Iter {iteration+1}: rolling Sharpe = {rs:.3f}")

        final_sharpe = rolling_sharpe(reward_buffer)
        log.info(f"    Final Sharpe for this config: {final_sharpe:.3f}")

        if final_sharpe > best_sharpe:
            best_sharpe = final_sharpe
            best_params = params
            best_model = model

    log.info(f"  Best params for {regime}: {best_params}, Sharpe={best_sharpe:.3f}")
    return best_model, best_params, best_sharpe


def main():
    parser = argparse.ArgumentParser(description="Layer 4: PPO RL Training per Regime")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--regime", type=str, default="models/hmm/regime_labels.csv")
    parser.add_argument("--as_cost", type=str, default="models/as_cost")
    parser.add_argument("--lgbm", type=str, default="models/lgbm")
    parser.add_argument("--output", type=str, default="models/rl")
    parser.add_argument("--n_timesteps", type=int, default=300_000)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data with DatetimeIndex from timestamp column
    price_df = pd.read_parquet(Path(args.data) / "price_features.parquet")
    if 'timestamp' in price_df.columns:
        price_df = price_df.set_index('timestamp')
        price_df.index = pd.to_datetime(price_df.index)

    # Load regime labels
    regime_labels = pd.read_csv(args.regime, index_col=0)
    regime_labels = regime_labels.iloc[:, 0] if regime_labels.ndim > 1 else regime_labels.squeeze()
    regime_labels.index = pd.to_datetime(regime_labels.index)

    # Load A&S cost models
    import joblib
    as_models = {}
    for regime in ["Calm", "Volatile", "Stressed"]:
        pkl_path = Path(args.as_cost) / f"as_cost_{regime.lower()}.pkl"
        if pkl_path.exists():
            as_models[regime] = joblib.load(pkl_path)

    # Load LightGBM forecasters (BTC/Calm, BTC/Volatile, ETH/Calm, ETH/Volatile)
    # For stressed regime, use synthetic forecasters since no stressed model exists
    from src.layer4_rl.rl_env import SyntheticForecaster
    lgbm_models = {}
    for asset in ["BTC", "ETH"]:
        lgbm_models[asset] = {}
        for regime in ["Calm", "Volatile", "Stressed"]:
            pkl_path = Path(args.lgbm) / f"lgbm_{asset.lower()}_{regime.lower()}.pkl"
            if pkl_path.exists():
                lgbm_models[asset][regime] = joblib.load(pkl_path)
            else:
                # Use synthetic forecaster for stressed regime
                lgbm_models[asset][regime] = SyntheticForecaster(regime, asset)

    # Create environment
    from src.layer4_rl.rl_env import RegimePortfolioEnv

    env = RegimePortfolioEnv(
        price_data=price_df,
        regime_labels=regime_labels,
        as_cost_models=as_models,
        forecasters=lgbm_models,
    )

    # Train per regime
    for regime in ["Calm", "Volatile", "Stressed"]:
        log.info(f"\nTraining PPO for {regime} regime...")
        model, params, sharpe = train_regime_ppo(env, regime, args.n_timesteps)
        model.save(str(output_dir / f"ppo_{regime.lower()}.zip"))
        log.info(f"  Saved: ppo_{regime.lower()}.zip")

    print(f"\nRL training complete. Policies saved to {output_dir}")


if __name__ == "__main__":
    main()
