"""
Stable RL Training Script — full-environment training with validation-based selection.

Fixes from regime-conditional approach:
1. Full environment training: single PPO learns across all regimes
2. log_std=-0.5: better exploration
3. Validation-based early stopping: prevent overfitting
4. Longer training: 100k steps
5. Market context features in observations (from rl_env.py fixes)
6. Reward shaping with churn + drawdown penalties (from rl_env.py fixes)

Key design:
- Single PPO policy trained on FULL RegimePortfolioEnv
- Regime context provided via obs[3] (regime index: 0=Calm, 1=Volatile, 2=Stressed)
- Train/Val/Test: 20 days train, 5 days val, rest test
- Early stopping on validation Sharpe ratio
- Best validation model saved to models/rl/ppo_full.zip
"""
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.layer4_rl.rl_env import RegimePortfolioEnv
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Annualization factor for 5-min bars
ANN_FACTOR = 288 * 365

# PPO hyperparameters — ultra-conservative for stability
REGIME_PARAMS = {
    'full': {'net_arch': [32, 32], 'lr': 3e-5, 'clip': 0.1, 'n_steps': 64, 'batch_size': 16, 'mgn': 0.5, 'log_std': -0.5},
}


def compute_sharpe(returns):
    """Compute annualized Sharpe ratio from returns series."""
    if len(returns) < 2:
        return 0.0
    ann_ret = returns.mean() * ANN_FACTOR
    ann_vol = returns.std() * np.sqrt(ANN_FACTOR)
    return ann_ret / (ann_vol + 1e-8)


def make_model(vec_env):
    """Create PPO model with ultra-conservative hyperparameters."""
    p = REGIME_PARAMS['full']
    model = PPO(
        'MlpPolicy', vec_env,
        learning_rate=p['lr'],
        gamma=0.99,
        clip_range=p['clip'],
        n_steps=p['n_steps'],
        batch_size=p['batch_size'],
        max_grad_norm=p['mgn'],
        device='cpu',
        policy_kwargs={
            'net_arch': p['net_arch'],
            'activation_fn': nn.ReLU,
        },
        verbose=0,
        seed=42,
        n_epochs=1,
        ent_coef=0.0,
    )
    # Initialize log_std to -0.5 for better exploration
    model.policy.log_std.data = torch.tensor([p['log_std'], p['log_std']])
    return model


def check_nan(model):
    """Check if model weights contain NaN or Inf."""
    w = model.policy.action_net.weight.data
    return not torch.isfinite(w).all()


def collect_episode_returns(env, model, deterministic=True, max_steps=2000):
    """Collect returns from one complete episode for Sharpe evaluation."""
    obs, _ = env.reset()
    episode_rewards = []
    done = truncated = False
    steps = 0

    while not (done or truncated) and steps < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards.append(reward)
        steps += 1

    return np.array(episode_rewards)


def evaluate_on_env(env, model, n_episodes=3):
    """Evaluate model on environment, return mean Sharpe of episodes."""
    all_episode_sharpes = []

    for _ in range(n_episodes):
        returns = collect_episode_returns(env, model, deterministic=True)
        if len(returns) >= 2:
            sharpe = compute_sharpe(pd.Series(returns))
            all_episode_sharpes.append(sharpe)

    return np.mean(all_episode_sharpes) if all_episode_sharpes else 0.0


def main():
    DATA_DIR = Path('data/processed')
    REGIME_PATH = Path('models/hmm/regime_labels.csv')
    AS_DIR = Path('models/as_cost')
    LGBM_DIR = Path('models/lgbm')
    OUTPUT_DIR = Path('models/rl')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    log.info('Loading data...')
    price_df = pd.read_parquet(DATA_DIR / 'price_features.parquet')
    if 'timestamp' in price_df.columns:
        price_df = price_df.set_index('timestamp')
    price_df.index = pd.to_datetime(price_df.index)
    log.info(f'  Price data: {price_df.shape}, range: {price_df.index.min()} to {price_df.index.max()}')

    regime_labels = pd.read_csv(REGIME_PATH, index_col=0).iloc[:, 0]
    regime_labels.index = pd.to_datetime(regime_labels.index)
    log.info(f'  Regime labels: {len(regime_labels)} entries, range: {regime_labels.index.min()} to {regime_labels.index.max()}')

    # Align regime_labels to price_df index (forward-fill missing at start)
    regime_labels_aligned = regime_labels.reindex(price_df.index)
    regime_labels_aligned = regime_labels_aligned.fillna('Calm')
    log.info(f'  Regime labels aligned to price_df: {len(regime_labels_aligned)} entries')

    # -------------------------------------------------------------------------
    # Train/Val/Test split: 20 days train, 5 days val, rest test
    # -------------------------------------------------------------------------
    data_min = price_df.index.min()
    TRAIN_END = data_min + pd.Timedelta(days=20)
    VAL_END = data_min + pd.Timedelta(days=25)

    train_mask = price_df.index <= TRAIN_END
    val_mask = (price_df.index > TRAIN_END) & (price_df.index <= VAL_END)
    test_mask = price_df.index > VAL_END

    price_train = price_df[train_mask]
    price_val = price_df[val_mask]
    price_test = price_df[test_mask]

    regime_train = regime_labels_aligned[train_mask]
    regime_val = regime_labels_aligned[val_mask]
    regime_test = regime_labels_aligned[test_mask]

    log.info(f'  Train: {len(price_train)} bars ({price_train.index.min().date()} to {price_train.index.max().date()})')
    log.info(f'  Val:   {len(price_val)} bars ({price_val.index.min().date()} to {price_val.index.max().date()})')
    log.info(f'  Test:  {len(price_test)} bars ({price_test.index.min().date()} to {price_test.index.max().date()})')

    # -------------------------------------------------------------------------
    # Load cost models
    # -------------------------------------------------------------------------
    log.info('Loading A&S cost models...')
    as_models = {}
    for regime in ['Calm', 'Volatile', 'Stressed']:
        as_models[regime] = joblib.load(AS_DIR / f'as_cost_{regime.lower()}.pkl')

    # -------------------------------------------------------------------------
    # Load LGBM forecasters
    # -------------------------------------------------------------------------
    log.info('Loading LightGBM forecasters...')
    from src.layer4_rl.rl_env import SyntheticForecaster
    lgbm_models = {}
    for asset in ['BTC', 'ETH']:
        lgbm_models[asset] = {}
        for regime in ['Calm', 'Volatile', 'Stressed']:
            pkl_path = LGBM_DIR / f'lgbm_{asset.lower()}_{regime.lower()}.pkl'
            if pkl_path.exists():
                lgbm_models[asset][regime] = joblib.load(pkl_path)
            else:
                lgbm_models[asset][regime] = SyntheticForecaster(regime, asset)
    log.info('  Loaded LGBM forecasters for BTC/ETH x Calm/Volatile/Stressed')

    # -------------------------------------------------------------------------
    # Create training-only environment for observation normalization
    # FIX CRITICAL-1: Compute normalization from TRAINING split ONLY to avoid
    # look-ahead bias. The normalization must NOT see validation/test data.
    # -------------------------------------------------------------------------
    log.info('Computing observation normalization from TRAINING split only...')
    train_env_for_norm = RegimePortfolioEnv(
        price_train, regime_train, as_models, lgbm_models
    )
    obs_norm = (train_env_for_norm._obs_mean, train_env_for_norm._obs_std)
    log.info(f'  obs_std[4:10]: {obs_norm[1][[4,5,6,7,8,9]].round(6)}')
    log.info(f'  (Using train split only to avoid look-ahead bias)')

    # -------------------------------------------------------------------------
    # Create training and validation environments (TRAIN SPLIT ONLY for training)
    # -------------------------------------------------------------------------
    log.info('Creating training environment (train split, all regimes)...')
    train_env = RegimePortfolioEnv(
        price_train, regime_train, as_models, lgbm_models
    )
    train_env._obs_mean, train_env._obs_std = obs_norm
    vec_train = DummyVecEnv([lambda: train_env])

    log.info('Creating validation environment (val split)...')
    val_env = RegimePortfolioEnv(
        price_val, regime_val, as_models, lgbm_models
    )
    val_env._obs_mean, val_env._obs_std = obs_norm

    # -------------------------------------------------------------------------
    # Training loop with validation-based early stopping
    # -------------------------------------------------------------------------
    TOTAL_STEPS = 100_000
    CHUNK = 500  # Train in chunks
    EVAL_EVERY = 2500  # Evaluate on validation every 2500 steps
    PATIENCE = 3  # Stop if no improvement for 3 consecutive evaluations

    log.info(f'\n=== Training PPO on FULL environment: {TOTAL_STEPS} steps ===')
    log.info(f'  Chunk size: {CHUNK}, Eval every: {EVAL_EVERY}, Patience: {PATIENCE}')

    model = make_model(vec_train)

    best_model_path = OUTPUT_DIR / 'ppo_full.zip'
    total_timesteps = 0
    reinit_count = 0
    eval_count = 0

    # Validation tracking
    best_val_sharpe = -np.inf
    best_model_state = None
    no_improve_count = 0

    while total_timesteps < TOTAL_STEPS:
        # Train for CHUNK steps
        try:
            model.learn(CHUNK, progress_bar=False, reset_num_timesteps=True)
            total_timesteps += CHUNK
            reinit_count = 0
        except (ValueError, RuntimeError) as e:
            if 'NaN' in str(e) or 'invalid values' in str(e) or 'inf' in str(e):
                reinit_count += 1
                if reinit_count > 3:
                    log.error(f'  Too many NaN errors ({reinit_count}), stopping training.')
                    break
                log.warning(f'  NaN/Invalid error at {total_timesteps} steps! Reinitializing...')
                del model
                model = make_model(vec_train)
                no_improve_count = 0  # Reset patience on reinit
                continue
            else:
                raise

        # Check for NaN after training
        if check_nan(model):
            log.warning(f'  NaN detected after training at {total_timesteps} steps! Reinitializing...')
            del model
            model = make_model(vec_train)
            no_improve_count = 0  # Reset patience on reinit
            continue

        # Log progress every 10 chunks (5000 steps)
        if (total_timesteps // CHUNK) % 10 == 0:
            w = model.policy.action_net.weight.data
            log.info(f'  Steps {total_timesteps}: weight_mean={w.mean().item():.6f}, NaN={not torch.isfinite(w).all()}')

        # Evaluate on validation set every EVAL_EVERY steps
        if total_timesteps % EVAL_EVERY == 0 and total_timesteps > 0:
            eval_count += 1
            val_sharpe = evaluate_on_env(val_env, model, n_episodes=3)
            log.info(f'  [Eval #{eval_count}] Steps {total_timesteps}: Val Sharpe = {val_sharpe:.4f} (best={best_val_sharpe:.4f})')

            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                best_model_state = {k: v.cpu().clone() for k, v in model.policy.state_dict().items()}
                no_improve_count = 0
                log.info(f'    *** New best model! Val Sharpe = {best_val_sharpe:.4f} ***')
            else:
                no_improve_count += 1
                log.info(f'    No improvement ({no_improve_count}/{PATIENCE})')
                if no_improve_count >= PATIENCE:
                    log.info(f'  Early stopping: no improvement for {PATIENCE} consecutive evaluations.')
                    break

    # -------------------------------------------------------------------------
    # Restore best model or save current
    # -------------------------------------------------------------------------
    if best_model_state is not None:
        log.info(f'  Restoring best model with Val Sharpe = {best_val_sharpe:.4f}')
        model.policy.load_state_dict(best_model_state)

    model.save(str(best_model_path))
    log.info(f'\n=== Training complete! ===')
    log.info(f'  Total steps trained: {total_timesteps}')
    log.info(f'  Total evaluations: {eval_count}')
    log.info(f'  Best validation Sharpe: {best_val_sharpe:.4f}')
    log.info(f'  Model saved to: {best_model_path}')

    # -------------------------------------------------------------------------
    # Save training summary
    # -------------------------------------------------------------------------
    summary = {
        'total_steps': int(total_timesteps),
        'best_val_sharpe': float(best_val_sharpe),
        'final_eval_count': eval_count,
        'train_split': {
            'start': str(price_train.index.min().date()),
            'end': str(price_train.index.max().date()),
            'n_bars': len(price_train),
        },
        'val_split': {
            'start': str(price_val.index.min().date()),
            'end': str(price_val.index.max().date()),
            'n_bars': len(price_val),
        },
        'hyperparameters': REGIME_PARAMS['full'],
    }
    with open(OUTPUT_DIR / 'training_full_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f'  Training summary saved to: {OUTPUT_DIR / "training_full_summary.json"}')

    print(f'\nTraining complete! Best validation Sharpe: {best_val_sharpe:.4f}')
    print(f'Model saved to: {best_model_path}')


if __name__ == '__main__':
    main()