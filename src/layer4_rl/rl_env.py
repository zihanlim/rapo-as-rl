"""
Layer 4: RL Environment — Gymnasium Custom Environment

Gym-style RL environment for regime-conditional portfolio rebalancing.

State Space (11 dimensions):
    w_btc, w_eth, w_cash, regime, mu_btc, mu_eth,
    sigma, spread, depth, sigma_port, cum_pnl

Action Space (2 dimensions, continuous):
    target_w_btc, target_w_eth  (w_cash = 1 - sum, clipped to [0,1])

Reward Function:
    r_t = (portfolio_return - A&S_cost) / sigma_port

Episode: One 5-min bar. Regime transitions handled by HMM at each step.

Usage:
    gymnasium.make("RegimePortfolioEnv-v0", price_data=..., regime_labels=..., ...)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class SyntheticForecaster:
    """
    Synthetic forecaster that returns reasonable forecast values for testing.
    Returns random walk-like forecasts based on regime-specific statistics.
    """

    def __init__(self, regime: str, asset: str):
        self.regime = regime
        self.asset = asset
        # Regime-specific expected return characteristics
        self.regime_params = {
            "Calm": {"mean": 0.0002, "std": 0.001},      # Small positive trend, low vol
            "Volatile": {"mean": 0.0000, "std": 0.003},  # No trend, medium vol
            "Stressed": {"mean": -0.0001, "std": 0.005},  # Slight negative trend, high vol
        }
        self._rng = np.random.RandomState(42)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return synthetic forecast based on regime characteristics."""
        params = self.regime_params.get(self.regime, {"mean": 0.0, "std": 0.002})
        # Use first feature column (return_1) with some noise
        base_return = features[0, 0] if features.shape[1] > 0 else 0.0
        # Blend with regime mean
        forecast = 0.7 * base_return + 0.3 * self._rng.normal(params["mean"], params["std"])
        return np.array([forecast])


class SyntheticCostModel:
    """
    Synthetic A&S cost model that returns reasonable cost parameters for testing.
    """

    def __init__(self, regime: str):
        self.regime = regime
        # Regime-specific cost characteristics
        self.regime_params = {
            "Calm": {"volatility": 0.02, "spread": 0.0005, "depth": 1.0, "gamma": 1e-6},
            "Volatile": {"volatility": 0.05, "spread": 0.001, "depth": 0.8, "gamma": 1e-6},
            "Stressed": {"volatility": 0.10, "spread": 0.002, "depth": 0.5, "gamma": 1e-5},
        }

    def get(self, key: str, default: float = 0.0) -> float:
        """Return cost parameter."""
        return self.regime_params.get(self.regime, {}).get(key, default)


def create_synthetic_forecasters() -> Dict[str, Dict[str, SyntheticForecaster]]:
    """Create synthetic forecasters for all assets and regimes."""
    forecasters = {}
    for asset in ["BTC", "ETH"]:
        forecasters[asset] = {}
        for regime in ["Calm", "Volatile", "Stressed"]:
            forecasters[asset][regime] = SyntheticForecaster(regime, asset)
    return forecasters


def create_synthetic_cost_models() -> Dict[str, SyntheticCostModel]:
    """Create synthetic A&S cost models for all regimes."""
    return {regime: SyntheticCostModel(regime) for regime in ["Calm", "Volatile", "Stressed"]}


def create_synthetic_price_data(n_bars: int = 10000, freq: str = "5min") -> pd.DataFrame:
    """
    Create synthetic OHLCV data for testing the RL environment.

    Generates realistic-looking crypto price data with:
    - Trend following returns
    - Regime-dependent volatility
    - Typical crypto characteristics (high vol, slight positive drift)
    """
    np.random.seed(42)

    # Generate timestamps
    end_time = pd.Timestamp("2024-01-01")
    timestamps = pd.date_range(end=end_time, periods=n_bars, freq=freq)

    # Generate prices with geometric brownian motion
    n_days = n_bars * 5 / 288  # 5-min bars per day

    # Regime sequence (roughly every 1000 bars switch)
    regimes = np.zeros(n_bars, dtype=int)
    regime_changes = np.random.choice(n_bars, size=n_bars // 1000, replace=False)
    current_regime = 0
    for i, change_idx in enumerate(np.sort(regime_changes)):
        regimes[change_idx:] = i % 3

    # Per-bar parameters
    daily_vol_btc = 0.04  # ~4% daily vol for BTC
    daily_vol_eth = 0.06  # ~6% daily vol for ETH
    daily_drift = 0.0002  # Small positive drift

    vol_per_bar_btc = daily_vol_btc / np.sqrt(288)
    vol_per_bar_eth = daily_vol_eth / np.sqrt(288)

    # Generate returns
    btc_returns = np.zeros(n_bars)
    eth_returns = np.zeros(n_bars)
    btc_prices = np.zeros(n_bars)
    eth_prices = np.zeros(n_bars)

    btc_price = 50000.0  # Starting BTC price
    eth_price = 3000.0   # Starting ETH price

    for i in range(n_bars):
        regime = regimes[i]

        # Regime modifiers
        vol_mult = {0: 0.8, 1: 1.2, 2: 2.0}[regime]
        drift_mult = {0: 1.0, 1: 0.5, 2: -0.5}[regime]

        # Generate correlated returns
        z_btc = np.random.normal(0, 1)
        z_eth = np.random.normal(0, 1)
        rho = 0.7  # BTC-ETH correlation

        btc_ret = daily_drift * drift_mult / 288 + vol_per_bar_btc * vol_mult * z_btc
        eth_ret = daily_drift * drift_mult / 288 + vol_per_bar_eth * vol_mult * (
            rho * z_btc + np.sqrt(1 - rho**2) * z_eth
        )

        btc_returns[i] = btc_ret
        eth_returns[i] = eth_ret

        btc_price *= np.exp(btc_ret)
        eth_price *= np.exp(eth_ret)

        btc_prices[i] = btc_price
        eth_prices[i] = eth_price

    # Build DataFrame
    df = pd.DataFrame({
        "btc_open": btc_prices * (1 - np.random.uniform(0, 0.001, n_bars)),
        "btc_high": btc_prices * (1 + np.random.uniform(0, 0.002, n_bars)),
        "btc_low": btc_prices * (1 - np.random.uniform(0, 0.002, n_bars)),
        "btc_close": btc_prices,
        "btc_volume": np.random.uniform(100, 1000, n_bars),
        "eth_open": eth_prices * (1 - np.random.uniform(0, 0.001, n_bars)),
        "eth_high": eth_prices * (1 + np.random.uniform(0, 0.002, n_bars)),
        "eth_low": eth_prices * (1 - np.random.uniform(0, 0.002, n_bars)),
        "eth_close": eth_prices,
        "eth_volume": np.random.uniform(500, 5000, n_bars),
        "btc_return": btc_returns,
        "eth_return": eth_returns,
        "btc_return_1": btc_returns,
        "eth_return_1": eth_returns,
        "btc_return_3": np.convolve(btc_returns, np.ones(3)/3, mode='same'),
        "eth_return_3": np.convolve(eth_returns, np.ones(3)/3, mode='same'),
        "btc_return_6": np.convolve(btc_returns, np.ones(6)/6, mode='same'),
        "eth_return_6": np.convolve(eth_returns, np.ones(6)/6, mode='same'),
        "realized_vol": np.abs(btc_returns) * 20,  # Scaled realized vol proxy
        "spread_proxy": 0.0005 + regimes * 0.0003,  # Higher spread in stressed
        "ofi": np.random.normal(0, 100, n_bars),  # Order flow imbalance
        "cross_asset_return": (btc_returns + eth_returns) / 2,
    }, index=timestamps)

    return df


def create_synthetic_regime_labels(n_bars: int, price_data: pd.DataFrame) -> pd.Series:
    """Create synthetic regime labels based on volatility regimes in the data."""
    np.random.seed(42)
    regimes = []
    for i in range(n_bars):
        vol = price_data.iloc[i]["realized_vol"]
        if vol < 0.03:
            regimes.append("Calm")
        elif vol < 0.06:
            regimes.append("Volatile")
        else:
            regimes.append("Stressed")
    return pd.Series(regimes, index=price_data.index, name="regime")


def create_regime_filtered_env(price_data, regime_labels, as_cost_models, forecasters, target_regime, initial_balance=100_000, obs_normalization=None):
    """
    Create an environment that filters data to only include contiguous segments
    of the target regime. Each contiguous segment becomes its own episode.

    This enables proper regime-conditional RL training where each regime's policy
    learns only from its own regime's data.

    Parameters
    ----------
    obs_normalization : tuple (mean, std) or None
        If provided, use these pre-computed normalization stats (from full dataset).
        If None, computes from filtered data (not recommended).
    """
    # Find contiguous regime segments
    segments = []
    current_start = 0
    current_regime = regime_labels.iloc[0]

    for i in range(1, len(regime_labels)):
        if regime_labels.iloc[i] != current_regime:
            if current_regime == target_regime:
                segments.append((current_start, i))
            current_start = i
            current_regime = regime_labels.iloc[i]

    if current_regime == target_regime:
        segments.append((current_start, len(regime_labels)))

    if not segments:
        raise ValueError(f"No data found for regime: {target_regime}")

    filtered_dfs = []
    for start, end in segments:
        filtered_dfs.append(price_data.iloc[start:end])

    filtered_price = pd.concat(filtered_dfs)
    filtered_regime = pd.concat([regime_labels.iloc[start:end] for start, end in segments])

    filtered_regime.index = range(len(filtered_regime))
    filtered_price.index = range(len(filtered_price))

    env = RegimePortfolioEnv(
        price_data=filtered_price,
        regime_labels=filtered_regime,
        as_cost_models=as_cost_models,
        forecasters=forecasters,
        initial_balance=initial_balance,
    )
    # Use pre-computed normalization from FULL dataset (not filtered data)
    if obs_normalization is not None:
        env._obs_mean, env._obs_std = obs_normalization
    return env

class RegimePortfolioEnv(gym.Env):
    """
    Regime-conditional portfolio rebalancing environment.

    Parameters
    ----------
    price_data : pd.DataFrame
        OHLCV data with btc_close, eth_close columns
    regime_labels : pd.Series
        HMM regime labels indexed by timestamp (0-based int or string)
    as_cost_models : dict
        Per-regime A&S cost model dicts {regime: params_dict}
    forecasters : dict
        Per-asset, per-regime LightGBM model dict {asset: {regime: model}}
    initial_balance : float
        Starting portfolio value in USD (default 100_000)
    """

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        price_data,
        regime_labels,
        as_cost_models,
        forecasters,
        initial_balance: float = 100_0,
    ):
        super().__init__()

        self.price_data = price_data
        self.regime_labels = regime_labels
        self.as_cost_models = as_cost_models
        self.forecasters = forecasters
        self.initial_balance = initial_balance

        self.n_assets = 2  # BTC, ETH
        self.t = 0
        self.max_t = len(self.price_data) - 2

        # Max episode steps to prevent infinite episodes (based on data length)
        self._max_episode_steps = self.max_t + 1
        self._elapsed_steps = 0

        # PPO is sensitive to observation scales. Compute normalization stats
        # from all data upfront so training and backtest use the SAME stats.
        self._compute_obs_normalization(price_data, regime_labels, as_cost_models, forecasters)

        # Observation space: 11 dimensions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        # Action space: target BTC and ETH weights
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Internal state
        self.current_weights = np.array([0.5, 0.5, 0.0], dtype=np.float32)  # BTC, ETH, cash
        self.cum_pnl = 0.0
        self.portfolio_value = initial_balance

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self._elapsed_steps = 0
        self.current_weights = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self.cum_pnl = 0.0
        self.portfolio_value = self.initial_balance
        return self._get_obs(), {}

    def _compute_obs_normalization(self, price_data, regime_labels, as_cost_models, forecasters):
        """Pre-compute observation mean/std from ALL data for stable PPO training."""
        samples = []
        for t in range(len(price_data) - 1):
            t_idx = t
            if isinstance(regime_labels.index[0], (int, np.integer)):
                regime_str = regime_labels.iloc[t_idx] if t_idx < len(regime_labels) else "Calm"
            else:
                ts = price_data.index[t_idx]
                regime_str = regime_labels.get(ts, "Calm")
            regime_idx = {"Calm": 0, "Volatile": 1, "Stressed": 2}.get(regime_str, 0)
            cost_model = as_cost_models.get(regime_str, {})
            # Use actual historical return as mu proxy so that obs_std reflects
            # the true scale of return observations. At runtime, LGBM will
            # override mu values, but obs_std keeps normalization stable.
            asset_prefixes = ["btc", "eth"]
            mu_values = []
            for prefix in asset_prefixes:
                ret_col = f"{prefix}_return"
                if ret_col in price_data.columns:
                    mu_values.append(float(price_data[ret_col].iloc[t_idx]))
                else:
                    mu_values.append(0.0)
            mu_btc, mu_eth = mu_values[0], mu_values[1]
            sigma_port = 0.01
            samples.append([
                0.5, 0.5, 0.0, float(regime_idx),
                mu_btc, mu_eth,
                cost_model.get("volatility", 0.0),
                cost_model.get("spread", 0.0),
                cost_model.get("depth", 0.0),
                sigma_port, 0.0,
            ])
        samples = np.array(samples, dtype=np.float32)
        self._obs_mean = np.mean(samples, axis=0)
        self._obs_std = np.std(samples, axis=0)
        # Weights (0-1), regime (0-2), sigma_port, cum_pnl: no normalization needed
        self._obs_std[0:4] = 1e8   # effectively no normalization
        self._obs_std[9] = 1e8     # sigma_port — don't normalize (varies at runtime)
        self._obs_std[10] = 1e8   # cum_pnl
        # For mu_btc/mu_eth/vol/spread/depth: ensure minimum std
        # so that runtime values don't produce extreme normalized values.
        # mu_btc/mu_eth have scale ~1e-3 to 1e-2; use MIN_STD=1e-5 to keep
        # normalized values within ~[-100, 100] before clipping to [-10, 10]
        MIN_STD = 1e-5
        for i in [4, 5, 6, 7, 8]:
            if self._obs_std[i] < MIN_STD:
                self._obs_std[i] = MIN_STD
        self._obs_count = len(samples)


    def step(self, action: np.ndarray):
        """Execute one 5-min trading period."""
        # Normalize action to valid weight simplex
        target = np.clip(action, 0.0, 1.0)
        target_sum = target.sum()
        if target_sum > 1.0:
            target = target / target_sum  # Normalize to sum=1
        target_weights = np.append(target, max(0.0, 1.0 - target.sum())).astype(np.float32)
        # Ensure valid simplex: w_btc >= 0, w_eth >= 0, w_cash >= 0, sum = 1
        target_weights = np.clip(target_weights, 0.0, 1.0)
        target_weights = target_weights / (target_weights.sum() + 1e-8)

        # Get current regime and cost model
        # Handle both integer index (filtered env) and datetime index (full env)
        if isinstance(self.regime_labels.index[0], (int, np.integer)):
            # Integer-indexed (filtered regime env): use self.t directly
            regime_str = self.regime_labels.iloc[self.t] if self.t < len(self.regime_labels) else "Calm"
        else:
            # Datetime-indexed (full env): look up by timestamp
            ts = self.price_data.index[self.t]
            regime_str = self.regime_labels.get(ts, "Calm")
        regime_idx = {"Calm": 0, "Volatile": 1, "Stressed": 2}.get(regime_str, 0)
        cost_model = self.as_cost_models.get(regime_str, {})

        # Get LightGBM return forecasts
        mu_btc = self._forecast("BTC", regime_str)
        mu_eth = self._forecast("ETH", regime_str)
        mu = np.array([mu_btc, mu_eth])

        # Portfolio return this period
        portfolio_return = np.dot(self.current_weights[:2], mu)

        # A&S execution cost
        delta_weights = target_weights[:2] - self.current_weights[:2]
        trade_value = np.abs(delta_weights) * self.portfolio_value
        btc_price = self.price_data["btc_close"].iloc[self.t]
        eth_price = self.price_data["eth_close"].iloc[self.t]

        cost_btc = self._as_cost(trade_value[0], btc_price, cost_model)
        cost_eth = self._as_cost(trade_value[1], eth_price, cost_model)
        total_cost = cost_btc + cost_eth

        # Net reward: fractional portfolio return minus fractional cost
        # Scale rewards to put them in a range PPO handles well (~[-1, 1])
        REWARD_SCALE = 100.0
        reward = (portfolio_return - (total_cost / self.portfolio_value)) * REWARD_SCALE

        # Update state — portfolio_value update must be in dollars to stay consistent
        realized_pnl = self.portfolio_value * portfolio_return - total_cost
        self.cum_pnl += realized_pnl
        # target_weights is [btc, eth]; cash fills the remainder to sum to 1
        cash_w = 1.0 - target_weights[0] - target_weights[1]
        self.current_weights = np.array([target_weights[0], target_weights[1], cash_w], dtype=np.float32)
        self.portfolio_value += realized_pnl
        self.t += 1
        self._elapsed_steps += 1

        # Episode termination: end of data or max steps reached
        terminated = self.t >= self.max_t
        truncated = self._elapsed_steps >= self._max_episode_steps
        done = terminated or truncated

        return self._get_obs(), reward, done, truncated, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        t_idx = min(self.t, len(self.price_data) - 1)
        # Handle both integer index (filtered env) and datetime index (full env)
        if isinstance(self.regime_labels.index[0], (int, np.integer)):
            # Integer-indexed (filtered regime env): use self.t directly
            regime_str = self.regime_labels.iloc[t_idx] if t_idx < len(self.regime_labels) else "Calm"
        else:
            # Datetime-indexed (full env): look up by timestamp
            ts = self.price_data.index[t_idx]
            regime_str = self.regime_labels.get(ts, "Calm")
        regime_idx = {"Calm": 0, "Volatile": 1, "Stressed": 2}.get(regime_str, 0)
        cost_model = self.as_cost_models.get(regime_str, {})

        mu_btc = self._forecast("BTC", regime_str)
        mu_eth = self._forecast("ETH", regime_str)
        sigma_port = self._rolling_volatility()

        raw_obs = np.array(
            [
                self.current_weights[0],
                self.current_weights[1],
                self.current_weights[2],
                float(regime_idx),
                mu_btc,
                mu_eth,
                cost_model.get("volatility", 0.0),
                cost_model.get("spread", 0.0),
                cost_model.get("depth", 0.0),
                sigma_port,
                self.cum_pnl,
            ],
            dtype=np.float32,
        )
        # Normalize: (obs - mean) / std for stable PPO training
        norm_obs = (raw_obs - self._obs_mean) / self._obs_std
        # Final safety: clip extreme values to prevent PPO NaN
        norm_obs = np.clip(norm_obs, -10.0, 10.0)
        norm_obs = np.where(np.isfinite(norm_obs), norm_obs, 0.0)
        return norm_obs

    def _forecast(self, asset: str, regime: str) -> float:
        """Get LightGBM forecast for asset/regime. Returns 0.0 if model unavailable."""
        try:
            model = self.forecasters.get(asset, {}).get(regime)
            if model is None:
                return 0.0
            # Build features for this timestamp (EXACTLY matching lgbm_train.py feature_cols)
            t_idx = min(self.t, len(self.price_data) - 1)
            row = self.price_data.iloc[t_idx]
            asset_prefix = asset.lower()
            # Feature names must EXACTLY match what LGBM was trained with
            feature_names = [
                "return_lag_1", "return_lag_3", "return_lag_6",
                "realized_vol", "spread_proxy", "ofi", "cross_asset_return",
                "regime_Calm", "regime_Volatile", "regime_Stressed",
            ]
            # Values in the same order as feature_names above
            feature_values = [
                row.get(f"{asset_prefix}_return_lag_1", 0.0),
                row.get(f"{asset_prefix}_return_lag_3", 0.0),
                row.get(f"{asset_prefix}_return_lag_6", 0.0),
                row.get("realized_vol", 0.0),
                row.get("spread_proxy", 0.0),
                row.get("ofi", 0.0),
                row.get("cross_asset_return", 0.0),
                float(regime == "Calm"),
                float(regime == "Volatile"),
                float(regime == "Stressed"),
            ]
            # Use DataFrame with correct feature names to avoid sklearn warning
            features = pd.DataFrame([feature_values], columns=feature_names)
            pred = model.predict(features)[0]
            if np.isnan(pred) or np.isinf(pred):
                return 0.0
            return float(pred)
        except Exception as e:
            # Only log once per 1000 steps to avoid spam
            if self.t % 1000 == 0:
                import logging
                logging.getLogger(__name__).warning(
                    f"Forecast failed for {asset}/{regime} at step {self.t}: {e}"
                )
            return 0.0

    def _as_cost(self, trade_value: float, price: float, cost_model: dict) -> float:
        """Compute A&S cost for a trade.

        A&S cost model parameters are per BTC:
            - s: spread in $/BTC (half-spread = s/2)
            - delta: market depth in BTC/$ (i.e., BTC流动性 per dollar)
            - gamma: temporary impact parameter in $/BTC per BTC traded
            - sigma: annual volatility in $/BTC (deannualized to per-bar)

        Trade size q is in BTC (trade_value / price).
        """
        if not cost_model or price == 0:
            return 0.0

        sigma_annual = cost_model.get("volatility", 0.0)  # annual vol in $/BTC
        sigma = sigma_annual / np.sqrt(365 * 288)  # per-bar vol in $/BTC
        s = cost_model.get("spread", 0.0)  # spread in $/BTC
        delta = cost_model.get("depth", 1.0)  # depth in BTC/$
        gamma = cost_model.get("gamma", 1e-6)  # gamma in $/BTC per BTC
        q = trade_value / price  # quantity in BTC

        # Guard against invalid values
        q = abs(q)  # Use absolute quantity
        if np.isnan(q) or np.isinf(q) or q <= 0:
            return 0.0
        if delta <= 0:
            delta = 1.0  # Avoid division by zero

        # A&S cost components:
        # Market impact: sigma * sqrt(q / (2*delta)) - permanent price impact
        market_impact = sigma * np.sqrt(max(0, q / (2 * delta))) * price
        # Spread cost: half-spread * quantity (in BTC terms, gives $)
        spread_cost = (s / 2) * q
        # Temporary impact: gamma * q^2 / (2*delta)
        impact_cost = gamma * (q ** 2) / (2 * delta) * price

        # Guard against NaN/Inf
        total_cost = market_impact + spread_cost + impact_cost
        if np.isnan(total_cost) or np.isinf(total_cost):
            return 0.0

        return total_cost

    def _rolling_volatility(self) -> float:
        """Rolling 20-period realized volatility of portfolio returns."""
        if self.t < 20:
            return 0.01
        returns = []
        for i in range(max(0, self.t - 20), self.t):
            p = self.price_data.iloc[i]
            ret = np.dot(self.current_weights[:2], [
                p.get("btc_return", 0.0), p.get("eth_return", 0.0)
            ])
            returns.append(ret)
        # Filter out NaN values before computing std
        returns = [r for r in returns if not np.isnan(r)]
        return np.std(returns) if len(returns) > 0 else 0.01

    def close(self):
        pass
