"""Create all milestone notebooks for the RAPO-AS-RL capstone project."""
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from pathlib import Path

NOTEBOOKS = Path("notebooks")

def create_hmm_notebook():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell("# Layer 1: HMM Regime Classifier Evaluation\n\nTrains a 3-state Gaussian HMM to classify market regimes as Calm / Volatile / Stressed."),
        new_markdown_cell("## Setup"),
        new_code_cell("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\nfrom src.layer1_hmm.hmm_train import build_hmm_features, select_states, train_hmm\nimport joblib\nfrom pathlib import Path\n\nDATA_DIR = Path('data/processed')\nMODEL_DIR = Path('models/hmm')\nprint('Libraries loaded')"),
        new_markdown_cell("## Load Processed Data"),
        new_code_cell("price_df = pd.read_parquet(DATA_DIR / 'price_features.parquet')\ntrades_df = pd.read_parquet(DATA_DIR / 'trades_processed.parquet')\nprint(f'Price data: {price_df.shape}')\nprint(f'Trades data: {trades_df.shape}')\nprint(f'Date range: {price_df[\"timestamp\"].min()} to {price_df[\"timestamp\"].max()}')"),
        new_markdown_cell("## Build HMM Features"),
        new_code_cell("features = build_hmm_features(price_df, trades_df)\nX = features.values\nprint(f'Feature matrix shape: {X.shape}')\nprint(f'Features: {list(features.columns)}')"),
        new_markdown_cell("## State Selection (BIC/AIC)"),
        new_code_cell("scores, best_n = select_states(X)\nprint('BIC/AIC Results:')\nfor n, s in scores.items():\n    print(f'  n_states={n}: BIC={s[\"bic\"]:.2f}, AIC={s[\"aic\"]:.2f}')\nprint(f'BIC selects: {best_n} states (project uses 3)')"),
        new_markdown_cell("## Train Final HMM (3 states)"),
        new_code_cell("model = train_hmm(X, n_states=3)\njoblib.dump(model, MODEL_DIR / 'hmm_model.pkl')\nprint(f'Model saved. Log-likelihood: {model.score(X):.2f}')"),
        new_markdown_cell("## State Labeling Validation"),
        new_code_cell("hidden_states = model.predict(X)\nregime_series = pd.Series(hidden_states).map(model.state_labels_)\nregime_series.index = features.index\n\nprint('State means (return, realized_vol, spread_proxy, ofi):')\nfor state_id, label in sorted(model.state_labels_.items(), key=lambda x: model.means_[x[0], 1]):\n    means = model.means_[state_id]\n    print(f'  {label}: return={means[0]:.6f}, vol={means[1]:.6f}')\n\nprint('\\nRegime distribution:')\nprint(regime_series.value_counts())"),
        new_markdown_cell("## Visualization"),
        new_code_cell("fig, axes = plt.subplots(3, 1, figsize=(14, 10))\ncolors = {'Calm': 'green', 'Volatile': 'orange', 'Stressed': 'red'}\n\nax = axes[0]\nfor regime in ['Calm', 'Volatile', 'Stressed']:\n    mask = regime_series == regime\n    ax.scatter(features.index[mask], price_df.loc[features.index[mask], 'close'],\n               c=colors[regime], s=2, alpha=0.5, label=regime)\nax.set_ylabel('BTC Close')\nax.set_title('BTC Close with HMM Regime Classification')\nax.legend()\nax.grid(True, alpha=0.3)\n\nax = axes[1]\nfor regime in ['Calm', 'Volatile', 'Stressed']:\n    mask = regime_series == regime\n    ax.scatter(features.index[mask], features.loc[features.index[mask], 'realized_vol'],\n               c=colors[regime], s=2, alpha=0.5, label=regime)\nax.set_ylabel('Realized Vol')\nax.set_title('Realized Volatility by Regime')\nax.legend()\nax.grid(True, alpha=0.3)\n\nax = axes[2]\nregime_num = regime_series.map({'Calm': 0, 'Volatile': 1, 'Stressed': 2})\nax.plot(regime_series.index, regime_num, linewidth=0.5)\nax.set_ylabel('Regime')\nax.set_yticks([0, 1, 2])\nax.set_yticklabels(['Calm', 'Volatile', 'Stressed'])\nax.set_title('HMM Regime Sequence')\nax.grid(True, alpha=0.3)\n\nplt.tight_layout()\nplt.savefig(MODEL_DIR / 'hmm_regime_plots.png', dpi=150)\nplt.show()"),
        new_markdown_cell("## Transition Analysis"),
        new_code_cell("transition_df = pd.crosstab(\n    regime_series.shift(1).fillna(method='ffill'),\n    regime_series, normalize='index'\n)\nprint('Regime Transition Matrix (row=t-1, col=t):')\nprint(transition_df.round(3))\n\nfig, ax = plt.subplots(figsize=(6, 5))\nsns.heatmap(transition_df, annot=True, fmt='.3f', cmap='Blues', ax=ax,\n            xticklabels=['Calm', 'Volatile', 'Stressed'],\n            yticklabels=['Calm', 'Volatile', 'Stressed'])\nax.set_title('HMM Regime Transition Probabilities')\nplt.tight_layout()\nplt.savefig(MODEL_DIR / 'hmm_transitions.png', dpi=150)\nplt.show()"),
        new_markdown_cell("## Save Regime Labels"),
        new_code_cell("full_regime = pd.Series(index=price_df['timestamp'], dtype=object)\nfull_regime.loc[features.index] = regime_series\nfull_regime = full_regime.ffill().bfill()\nfull_regime.to_csv(MODEL_DIR / 'regime_labels.csv', header=True)\nprint(f'Regime labels saved.\\n{full_regime.value_counts()}')"),
        new_markdown_cell("## Limitations\n- BIC prefers 4 states; project constrains to 3 for interpretability\n- Stressed regime is rare due to limited crisis data\n- Non-convergence warnings; multiple restarts used\n- Data is hybrid: recent real Binance OHLCV + synthetic historical"),
    ]
    nbformat.write(nb, NOTEBOOKS / '01_hmm_regime_classification.ipynb')
    print("Notebook 01 created")


def create_data_notebook():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell("# Layer 0: Data Collection & Processing\n\nCollects Binance OHLCV and trade tick data via ccxt. Falls back to synthetic data with realistic microstructure when API is unavailable."),
        new_markdown_cell("## Pipeline Overview"),
        new_code_cell("# Data flow:\n# 1. scripts/fetch_binance_data.py  -> data/raw/\n# 2. scripts/process_data.py         -> data/processed/\n# 3. scripts/validate_data.py        -> validation report\n\n# Scripts use ccxt for Binance API, with synthetic microstructure fallback\n# Synthetic data: HMM-like regime transitions, realistic vol/spread per regime"),
        new_markdown_cell("## Raw Data Summary"),
        new_code_cell("import pandas as pd\nfrom pathlib import Path\n\nraw = Path('data/raw')\nprocessed = Path('data/processed')\n\n# Load metadata\nimport json\nwith open(raw / 'fetch_metadata.json') as f:\n    metadata = json.load(f)\nprint('Data source:', 'REAL Binance API' if not metadata.get('synthetic_data') else 'SYNTHETIC')\nprint('Symbols:', metadata.get('symbols'))\nprint('Generated at:', metadata.get('generated_at'))"),
        new_markdown_cell("## Processed Data Inspection"),
        new_code_cell("price_df = pd.read_parquet(processed / 'price_features.parquet')\ntrades_df = pd.read_parquet(processed / 'trades_processed.parquet')\nprint(f'Price features: {price_df.shape}')\nprint(f'Columns: {list(price_df.columns)}')\nprint(f'\\nTrades: {trades_df.shape}')\nprint(f'\\nDate range: {price_df[\"timestamp\"].min()} to {price_df[\"timestamp\"].max()}')\nprint(f'\\nReturn stats:\\n{price_df[[\"btc_return\", \"eth_return\"]].describe()}')"),
        new_markdown_cell("## Regime Distribution"),
        new_code_cell("import numpy as np\n\n# Use HMM-like classification based on realized_vol for quick inspection\nvol = price_df['realized_vol'].dropna()\nq33, q66 = vol.quantile([0.33, 0.66])\nprint(f'Vol quantiles: 33%={q33:.6f}, 66%={q66:.6f}')\n\nregime = pd.cut(vol, bins=[-np.inf, q33, q66, np.inf], labels=['Calm', 'Volatile', 'Stressed'])\nprint(f'\\nQuick regime distribution:\\n{regime.value_counts()}')"),
        new_markdown_cell("## Synthetic Data Validation\n\n[Executed if synthetic data was used]"),
    ]
    nbformat.write(nb, NOTEBOOKS / '00_data_collection.ipynb')
    print("Notebook 00 created")


create_data_notebook()
create_hmm_notebook()
print("All notebooks created")
