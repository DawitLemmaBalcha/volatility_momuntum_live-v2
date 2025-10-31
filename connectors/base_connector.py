# connectors/base_connector.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
from core.core_types import Position, Tick

class BaseConnector(ABC):
    """
    Abstract Base Class for all platform connectors.
    Defines the standard interface for the trading bot to interact with any exchange.
    """
    # ... (connect, disconnect, place_order, etc. remain the same) ...
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float = None, stop_loss: float = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def close_position(self, position: Position) -> Dict[str, Any]:
        pass

    @abstractmethod
    def modify_stop_loss(self, symbol: str, trade_id: str, new_stop_price: float) -> bool:
        pass


    @abstractmethod
    def start_data_stream(self, symbol: str, on_tick_callback: Callable[[Tick], None]): # <-- FIX: Add 'symbol' parameter
        """
        Starts the flow of market data for a specific symbol.
        Args:
            symbol (str): The symbol to subscribe to (e.g., 'BTC/USDT').
            on_tick_callback (Callable): The function that will be called for each new data tick.
        """
        pass

    @abstractmethod
    def get_wallet_balance(self, coin: str) -> float:
        """Fetches the available balance for a specific coin."""
        pass