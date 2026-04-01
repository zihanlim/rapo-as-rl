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


class RegimePortfolioEnv(gym.Env):
    """
    Regime-conditional portfolio rebalancing environment.

    Parameters
    ----------
    price_data : pd.DataFrame
        OHLCV data with btc_close, eth_close columns
    regime_labels : pd.Series
        HMM regime labels indexed by timestamp
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
        self.current_weights = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self.cum_pnl = 0.0
        self.portfolio_value = self.initial_balance
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Execute one 5-min trading period."""
        # Normalize action to valid weight simplex
        target = np.clip(action, 0.0, 1.0)
        target = target / (target.sum() + 1e-8)
        target_weights = np.append(target, max(0.0, 1.0 - target.sum())).astype(np.float32)

        # Get current regime and cost model
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

        # Net reward: risk-adjusted return net of cost
        realized_pnl = self.portfolio_value * portfolio_return - total_cost
        sigma_port = self._rolling_volatility()
        reward = (realized_pnl - total_cost) / (sigma_port + 1e-8)

        # Update state
        self.cum_pnl += realized_pnl
        self.current_weights = target_weights
        self.portfolio_value += realized_pnl
        self.t += 1

        done = self.t >= self.max_t
        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        ts = self.price_data.index[min(self.t, len(self.price_data) - 1)]
        regime_str = self.regime_labels.get(ts, "Calm")
        regime_idx = {"Calm": 0, "Volatile": 1, "Stressed": 2}.get(regime_str, 0)
        cost_model = self.as_cost_models.get(regime_str, {})

        mu_btc = self._forecast("BTC", regime_str)
        mu_eth = self._forecast("ETH", regime_str)
        sigma_port = self._rolling_volatility()

        return np.array(
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

    def _forecast(self, asset: str, regime: str) -> float:
        """Get LightGBM forecast for asset/regime. Returns 0.0 if model unavailable."""
        try:
            model = self.forecasters.get(asset, {}).get(regime)
            if model is None:
                return 0.0
            # Build features for this timestamp (simplified)
            t_idx = min(self.t, len(self.price_data) - 1)
            row = self.price_data.iloc[t_idx]
            features = np.array([[
                row.get(f"{asset.lower()}_return_1", 0.0),
                row.get(f"{asset.lower()}_return_3", 0.0),
                row.get(f"{asset.lower()}_return_6", 0.0),
                row.get("realized_vol", 0.0),
                row.get("spread_proxy", 0.0),
                row.get("ofi", 0.0),
                row.get("cross_asset_return", 0.0),
                float(regime == "Calm"),
                float(regime == "Volatile"),
                float(regime == "Stressed"),
            ]], dtype=np.float32)
            return float(model.predict(features)[0])
        except Exception:
            return 0.0

    def _as_cost(self, trade_value: float, price: float, cost_model: dict) -> float:
        """Compute A&S cost for a trade."""
        if not cost_model or price == 0:
            return 0.0
        sigma = cost_model.get("volatility", 0.0) / np.sqrt(365 * 288)  # deannualize
        s = cost_model.get("spread", 0.0)
        delta = cost_model.get("depth", 1.0)
        gamma = cost_model.get("gamma", 1e-6)
        q = trade_value / price
        market_impact = sigma * np.sqrt(q / (2 * delta)) * price
        spread_cost = (s / 2) * price
        impact_cost = gamma * (q**2) / (2 * delta) * price
        return market_impact + spread_cost + impact_cost

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
        return np.std(returns) if returns else 0.01

    def close(self):
        pass
