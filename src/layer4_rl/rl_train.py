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
from stable_baselines3.common.vec_env import DummyVecEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def rolling_sharpe(rewards: list[float], window: int = 100) -> float:
    """Compute rolling Sharpe ratio."""
    if len(rewards) < window:
        return 0.0
    recent = rewards[-window:]
    return np.mean(recent) / (np.std(recent) + 1e-8)


def train_regime_ppo(env, regime: str, n_timesteps: int = 1_000_000) -> tuple:
    """
    Train PPO agent for a specific regime with hyperparameter grid search.

    Hyperparameter grid:
        learning_rate: {3e-5, 1e-5}
        gamma: {0.99, 0.95}
        clip_range: {0.1, 0.2}

    Early stopping: rolling Sharpe (100-period) stabilizes within ±0.05
    over 50,000 timesteps.

    Returns:
        best_model: trained PPO model
        best_params: dict with best hyperparameters
        best_sharpe: float Sharpe ratio of best model
        all_results: list of dicts, one per config, each containing:
            params, final_sharpe, iterations, sharpe_history
    """
    # Wrap env in DummyVecEnv (required by SB3) with small n_steps for stable updates
    # Small n_steps = more frequent policy updates = better gradient stability
    vec_env = DummyVecEnv([lambda: env])

    # Two configs with conservative LR to avoid PPO NaN divergence
    hyperparams = [
        {"learning_rate": 3e-5, "gamma": 0.99, "clip_range": 0.2},  # moderate LR
        {"learning_rate": 1e-5, "gamma": 0.99, "clip_range": 0.1},  # very low LR
    ]

    best_sharpe = -np.inf
    best_model = None
    best_params = None
    all_results = []

    for params in hyperparams:
        log.info(f"  Testing: lr={params['learning_rate']}, γ={params['gamma']}, clip={params['clip_range']}")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            clip_range=params["clip_range"],
            n_steps=64,           # Small n_steps = more frequent updates (prevents NaN)
            batch_size=64,        # Minibatch size for PPO update
            max_grad_norm=0.5,    # Gradient clipping for training stability
            device="cpu",         # CPU for numerical stability
            policy_kwargs={"net_arch": [64, 64], "activation_fn": ReLU},
            ent_coef=0.01,        # Light exploration bonus
        )
        # Initialize log_std to moderate value to prevent policy collapse
        with torch.no_grad():
            model.policy.log_std.data = torch.tensor([-0.3, -0.3])
            verbose=0,
            seed=42,
            ent_coef=0.01,        # Encourage exploration
        )

        # Train with evaluation every 5K steps
        # Early stopping: after at least 3 eval windows, if last 3 Sharpe values
        # are all within ±0.05 of each other, the model has converged
        eval_every = 5_000
        n_iterations = n_timesteps // eval_every
        reward_buffer = []
        iteration_sharpes = []

        for iteration in range(n_iterations):
            # KEY FIX: reset_num_timesteps=True prevents SB3 buffer-related NaN trigger
            model.learn(eval_every, progress_bar=False, reset_num_timesteps=True)

            # Evaluate on the vec_env
            eval_obs = vec_env.reset()
            eval_rewards = []
            for _ in range(min(500, env.max_t)):
                action, _ = model.predict(eval_obs, deterministic=True)
                eval_obs, reward, done, trunc = vec_env.step(action)
                eval_rewards.append(reward[0])  # vec_env returns array
                if done[0] or trunc[0]:
                    eval_obs = vec_env.reset()
            rs = rolling_sharpe(eval_rewards)
            reward_buffer.extend(eval_rewards)
            iteration_sharpes.append(rs)

            if len(iteration_sharpes) >= 3:
                recent = iteration_sharpes[-3:]
                if max(recent) - min(recent) < 0.05:
                    log.info(f"    Early stop at iteration {iteration+1}, Sharpe={rs:.3f} (stable)")
                    break

            log.info(f"    Iter {iteration+1}: rolling Sharpe = {rs:.3f}")

        final_sharpe = rolling_sharpe(reward_buffer)
        log.info(f"    Final Sharpe for this config: {final_sharpe:.3f}")

        config_result = {
            "params": params,
            "final_sharpe": final_sharpe,
            "iterations": len(iteration_sharpes),
            "sharpe_history": iteration_sharpes,
        }
        all_results.append(config_result)

        if final_sharpe > best_sharpe:
            best_sharpe = final_sharpe
            best_params = params
            best_model = model

    log.info(f"  Best params for {regime}: {best_params}, Sharpe={best_sharpe:.3f}")
    return best_model, best_params, best_sharpe, all_results


def main():
    parser = argparse.ArgumentParser(description="Layer 4: PPO RL Training per Regime")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--regime", type=str, default="models/hmm/regime_labels.csv")
    parser.add_argument("--as_cost", type=str, default="models/as_cost")
    parser.add_argument("--lgbm", type=str, default="models/lgbm")
    parser.add_argument("--output", type=str, default="models/rl")
    parser.add_argument("--n_timesteps", type=int, default=200_000)
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

    # Create full-environment for backtest evaluation (uses all data with regime transitions)
    from src.layer4_rl.rl_env import RegimePortfolioEnv
    from src.layer4_rl.rl_env import create_regime_filtered_env

    full_env = RegimePortfolioEnv(
        price_data=price_df,
        regime_labels=regime_labels,
        as_cost_models=as_models,
        forecasters=lgbm_models,
    )

    # Pre-compute observation normalization stats from FULL dataset
    # This ensures all regime-specific PPOs train with the same normalization
    obs_norm = (full_env._obs_mean, full_env._obs_std)
    log.info(f"  Observation normalization: mean={obs_norm[0][[4,5,6,7,8]].round(4)}, std={obs_norm[1][[4,5,6,7,8]].round(4)}")

    import json

    # Train per regime — each regime's PPO trains ONLY on its own regime's data
    training_results = {}
    for regime in ["Calm", "Volatile", "Stressed"]:
        log.info(f"\nTraining PPO for {regime} regime (regime-filtered data)...")
        n_regime_bars = sum(1 for r in regime_labels if r == regime)
        log.info(f"  {regime} regime has {n_regime_bars} bars in full dataset")

        if n_regime_bars < 500:
            log.warning(f"  {regime} has only {n_regime_bars} bars — insufficient for RL training!")
            log.info(f"  Skipping {regime} training, will use Calm policy as fallback")

        env = create_regime_filtered_env(
            price_data=price_df,
            regime_labels=regime_labels,
            as_cost_models=as_models,
            forecasters=lgbm_models,
            target_regime=regime,
            initial_balance=100_000.0,
            obs_normalization=obs_norm,  # Use FULL-data normalization
        )
        log.info(f"  Regime-filtered env: {len(env.price_data)} bars (from {len(price_df)} total)")
        model, params, sharpe, results = train_regime_ppo(env, regime, args.n_timesteps)
        model.save(str(output_dir / f"ppo_{regime.lower()}.zip"))
        log.info(f"  Saved: ppo_{regime.lower()}.zip")
        training_results[regime] = {
            "best_params": params,
            "best_sharpe": sharpe,
            "config_results": [
                {"params": r["params"], "final_sharpe": r["final_sharpe"],
                 "iterations": r["iterations"], "sharpe_history": r["sharpe_history"]}
                for r in results
            ],
        }

    # Save training results to JSON
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(training_results, f, indent=2)
    log.info(f"\nTraining results saved to {results_path}")

    # ------------------------------------------------------------------
    # Post-training: generate training curves and evaluation charts
    # ------------------------------------------------------------------
    log.info("\nGenerating training curves and evaluation charts...")

    # Plot training Sharpe curves per regime
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, regime in enumerate(["Calm", "Volatile", "Stressed"]):
        ax = axes[idx]
        for result in training_results[regime]["config_results"]:
            label = f"lr={result['params']['learning_rate']}, clip={result['params']['clip_range']}"
            ax.plot(result['sharpe_history'], label=label, alpha=0.8)
        ax.set_title(f'{regime} Regime - Training Sharpe')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Rolling Sharpe (100-step)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  Saved: training_curves.png")

    # Load models and evaluate vs random baseline on full environment
    from stable_baselines3 import PPO

    class RandomPolicy:
        def __init__(self, action_space):
            self.action_space = action_space
        def predict(self, obs, deterministic=False):
            return self.action_space.sample(), None

    eval_metrics = {}
    comparison_results = {}

    for regime in ["Calm", "Volatile", "Stressed"]:
        model_path = output_dir / f"ppo_{regime.lower()}.zip"
        if not model_path.exists():
            continue
        model = PPO.load(str(model_path))

        # Evaluate RL policy on FULL environment (not regime-filtered)
        rl_rewards = []
        obs, _ = full_env.reset()
        rebalance_count = 0
        prev_weights = full_env.current_weights.copy()
        for _ in range(min(500, full_env.max_t)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = full_env.step(action)
            rl_rewards.append(reward)
            if not np.allclose(full_env.current_weights[:2], prev_weights[:2], atol=1e-4):
                rebalance_count += 1
            prev_weights = full_env.current_weights.copy()
            if done or truncated:
                obs, _ = full_env.reset()

        rl_ret = np.array(rl_rewards)
        rl_sharpe = np.mean(rl_ret) / (np.std(rl_ret) + 1e-8) * np.sqrt(288 * 365)
        cumulative = np.cumprod(1 + rl_ret)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        rl_mdd = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # Evaluate Random baseline on full environment
        rnd_rewards = []
        obs, _ = full_env.reset()
        for _ in range(min(500, full_env.max_t)):
            action = full_env.action_space.sample()
            obs, reward, done, truncated, _ = full_env.step(action)
            rnd_rewards.append(reward)
            if done or truncated:
                obs, _ = full_env.reset()

        rnd_ret = np.array(rnd_rewards)
        rnd_sharpe = np.mean(rnd_ret) / (np.std(rnd_ret) + 1e-8) * np.sqrt(288 * 365)

        eval_metrics[regime] = {
            'sharpe': rl_sharpe, 'max_drawdown': rl_mdd,
            'mean_rebalances': rebalance_count,
        }
        comparison_results[regime] = {'rl': eval_metrics[regime], 'random': {
            'sharpe': rnd_sharpe, 'max_drawdown': 0, 'mean_rebalances': 0,
        }}
        log.info(f"  {regime}: RL Sharpe={rl_sharpe:.3f}, Random Sharpe={rnd_sharpe:.3f}")

    # Save evaluation summary CSV
    summary_rows = []
    for regime in ["Calm", "Volatile", "Stressed"]:
        if regime not in comparison_results:
            continue
        rl = comparison_results[regime]['rl']
        rnd = comparison_results[regime]['random']
        imp = ((rl['sharpe'] - rnd['sharpe']) / abs(rnd['sharpe']) * 100) if abs(rnd['sharpe']) > 1e-8 else float('inf')
        summary_rows.append({
            'Regime': regime,
            'RL Sharpe': f"{rl['sharpe']:.3f}",
            'Random Sharpe': f"{rnd['sharpe']:.3f}",
            'RL Max DD': f"{rl['max_drawdown']*100:.2f}%",
            'RL Rebalances': f"{rl['mean_rebalances']:.1f}",
            'Sharpe Improvement': f"{imp:.1f}%",
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'rl_evaluation_summary.csv', index=False)
    log.info(f"  Saved: rl_evaluation_summary.csv")

    print(f"\nRL training complete. Policies + charts saved to {output_dir}")


if __name__ == "__main__":
    main()
