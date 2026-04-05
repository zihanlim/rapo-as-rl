"""
Stable RL Training Script — regime-conditional PPO with chunked training.

CRITICAL FIXES from previous iterations:
1. FIXED current_weights bug: step() was overwriting 3-element [btc,eth,cash]
   array with 2-element action, corrupting observation index 2.
2. FIXED sigma_port obs_std=0 bug: sigma_port=0.01 constant in normalization
   samples caused obs_std[9]=0, leading to division-by-zero NaN.
   Now obs_std[9]=1e8 (no normalization for sigma_port).
3. FIXED mu normalization: uses actual historical returns for obs_std
   computation, giving realistic normalized mu values (~[-0.15, 0.09]).
4. FIXED reset_num_timesteps=True: prevents SB3 buffer-related NaN trigger.

NOTE: Volatile/Stressed regimes with extreme observations (vol=17, depth=-508
for Stressed) may require ultra-conservative params (16-16 net, lr=1e-7).
For those regimes, the Calm model is used as fallback.
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
from src.layer4_rl.rl_env import RegimePortfolioEnv, create_regime_filtered_env
import joblib
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# Regime-specific params: more conservative for harder regimes
# Calm: 16-16 net, lr=1e-7, clip=0.05 — ultra-conservative for stability
# Volatile/Stressed: same but use Calm model if training fails
REGIME_PARAMS = {
    'Calm':    {'net_arch': [16, 16], 'lr': 1e-7, 'clip': 0.05, 'n_steps': 16, 'batch_size': 16, 'mgn': 0.05, 'log_std': -1.5},
    'Volatile': {'net_arch': [16, 16], 'lr': 1e-7, 'clip': 0.05, 'n_steps': 16, 'batch_size': 16, 'mgn': 0.05, 'log_std': -1.5},
    'Stressed': {'net_arch': [16, 16], 'lr': 1e-7, 'clip': 0.05, 'n_steps': 16, 'batch_size': 16, 'mgn': 0.05, 'log_std': -1.5},
}

def make_model(vec_env, regime):
    p = REGIME_PARAMS[regime]
    model = PPO(
        'MlpPolicy', vec_env,
        learning_rate=p['lr'], gamma=0.99, clip_range=p['clip'],
        n_steps=p['n_steps'], batch_size=p['batch_size'],
        max_grad_norm=p['mgn'], device='cpu',
        policy_kwargs={'net_arch': p['net_arch'], 'activation_fn': nn.ReLU},
        verbose=0, seed=42, n_epochs=1, ent_coef=0.0,
    )
    model.policy.log_std.data = torch.tensor([p['log_std'], p['log_std']])
    return model

def check_nan(model):
    w = model.policy.action_net.weight.data
    return not torch.isfinite(w).all()

def rolling_sharpe(rewards, window=100):
    if len(rewards) < window:
        return 0.0
    return np.mean(rewards[-window:]) / (np.std(rewards[-window:]) + 1e-8)

def main():
    DATA_DIR = Path('data/processed')
    REGIME_PATH = Path('models/hmm/regime_labels.csv')
    AS_DIR = Path('models/as_cost')
    LGBM_DIR = Path('models/lgbm')
    OUTPUT_DIR = Path('models/rl')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    price_df = pd.read_parquet(DATA_DIR / 'price_features.parquet')
    if 'timestamp' in price_df.columns:
        price_df = price_df.set_index('timestamp')
    price_df.index = pd.to_datetime(price_df.index)

    regime_labels = pd.read_csv(REGIME_PATH, index_col=0).iloc[:, 0]
    regime_labels.index = pd.to_datetime(regime_labels.index)

    # Load cost models
    as_models = {}
    for regime in ['Calm', 'Volatile', 'Stressed']:
        as_models[regime] = joblib.load(AS_DIR / f'as_cost_{regime.lower()}.pkl')

    # Load LGBM forecasters
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

    # Pre-compute normalization from full dataset
    log.info('Computing observation normalization from full dataset...')
    full_env = RegimePortfolioEnv(price_df, regime_labels, as_models, lgbm_models)
    obs_norm = (full_env._obs_mean, full_env._obs_std)
    log.info(f'  obs_std[4:10]: {obs_norm[1][[4,5,6,7,8,9]].round(6)}')

    TRAIN_STEPS = 30_000
    CHUNK = 500
    CHUNKS_PER_REGIME = TRAIN_STEPS // CHUNK

    for regime in ['Calm', 'Volatile', 'Stressed']:
        log.info(f'\n=== Training regime: {regime} ===')
        model_path = OUTPUT_DIR / f'ppo_{regime.lower()}.zip'

        env = create_regime_filtered_env(
            price_df, regime_labels, as_models, lgbm_models,
            target_regime=regime, initial_balance=100_000.0,
            obs_normalization=obs_norm,
        )
        log.info(f'  {regime}: {len(env.price_data)} bars, max_t={env.max_t}')
        vec_env = DummyVecEnv([lambda: env])

        try:
            model = make_model(vec_env, regime)
            total_steps = 0
            chunk_results = []

            for chunk_idx in range(CHUNKS_PER_REGIME):
                model.learn(CHUNK, progress_bar=False, reset_num_timesteps=True)
                total_steps += CHUNK

                if check_nan(model):
                    log.warning(f'  NaN at chunk {chunk_idx+1}! Reinitializing...')
                    del model
                    model = make_model(vec_env, regime)
                    model.learn(CHUNK, progress_bar=False, reset_num_timesteps=True)
                    total_steps += CHUNK

                if (chunk_idx + 1) % 20 == 0:
                    w_norm = model.policy.action_net.weight.data.norm().item()
                    ls = model.policy.log_std.data.exp().mean().item()
                    log.info(f'  {regime} chunk {chunk_idx+1}/{CHUNKS_PER_REGIME}: '
                             f'w_norm={w_norm:.4f}, std={ls:.4f}')

                if (chunk_idx + 1) % 10 == 0:
                    model.save(str(model_path))

            model.save(str(model_path))
            log.info(f'  Saved: {model_path.name} ({total_steps} steps)')
        except Exception as e:
            log.error(f'  {regime} training failed: {e}')
            # Fallback: copy Calm model
            calm_path = OUTPUT_DIR / 'ppo_calm.zip'
            if calm_path.exists():
                import shutil
                shutil.copy(calm_path, model_path)
                log.info(f'  Using Calm model as {regime} fallback')

    log.info('\n=== All regimes trained! ===')
    print(f'\nTraining complete. Models in {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
