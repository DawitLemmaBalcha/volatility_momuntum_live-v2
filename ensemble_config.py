# ensemble_config.py
import os

# --- Connector Imports ---
try:
    from connectors.mt5_connector import MetaTraderConnector
    from connectors.bybit_connector import BybitConnector
except ImportError:
    MetaTraderConnector = BybitConnector = None

from connectors.bybit_testnet_connector import BybitTestnetConnector
from connectors.dummy_connector import DummyConnector

# --- !! DEBUGGING STEP !! ---
# Let's print the key right after trying to load it to see what's happening.
testnet_key = os.getenv("BYBIT_TESTNET_API_KEY")
print(f"--- DEBUG: Attempting to load BYBIT_TESTNET_API_KEY. Value found: '{testnet_key}' ---")
# --- !! END DEBUGGING STEP !! ---

VENUES = {
    # --- Backtest Venue ---
    "ensemble_backtest": {
        "mode": "backtest",
        "connector_class": DummyConnector,
        "connection_params": {},
        "strategies": ["strategies/a_aggressive_scalper.py", "strategies/b_conservative_trend.py"]
    },
    
    # --- Paper Trading Venue ---
    "bybit_paper_trading": {
        "mode": "live",
        "connector_class": BybitTestnetConnector,
        "connection_params": {
             "api_key": testnet_key, # Use the variable we loaded for debugging
             "api_secret": os.getenv("BYBIT_TESTNET_API_SECRET"),
        },
        "strategies": [
            "strategies/aggressive_1.py",
            "strategies/aggressive_2.py",
            "strategies/aggressive_3.py",
            "strategies/aggressive_4.py",
            "strategies/conservative_1.py",
            "strategies/conservative_2.py",
            "strategies/conservative_3.py",
            "strategies/conservative_4.py" 
        ]
    },
    
    # --- Live Trading Venues ---
    "live_prop_firm_A": {
        "mode": "live",
        "connector_class": MetaTraderConnector,
        "connection_params": {
            "login": 123456, "password": "YOUR_MT5_PASSWORD", "server": "PropFirm-Server"
        },
        "strategies": ["strategies/b_conservative_trend.py"]
    },
    "live_bybit_main": {
        "mode": "live",
        "connector_class": BybitConnector,
        "connection_params": {
             "api_key": os.getenv("BYBIT_MAINNET_API_KEY"),
             "api_secret": os.getenv("BYBIT_MAINNET_API_SECRET"),
        },
        "strategies": ["strategies/a_aggressive_scalper.py"]
    }
}