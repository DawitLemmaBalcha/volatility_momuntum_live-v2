# connectors/dummy_connector.py

from abc import ABC
from typing import Dict, Any, Callable
from core_types import SimulationClock, Tick, Position

class DummyConnector(ABC):
    """A simulated connector for backtesting purposes."""
    SLIPPAGE_MULTIPLIER = 0.001
    
    # --- FIX 1 of 2: Modify __init__ to accept 'engine' ---
    def __init__(self, clock: SimulationClock, params: dict, engine=None):
        self.clock, self.params, self._trade_id_counter = clock, params, 0
        self.engine = engine # <-- Add this line
    
    def connect(self): 
        pass
    def disconnect(self): 
        pass

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float = None, stop_loss: float = None) -> Dict[str, Any]:
        self._trade_id_counter += 1
        base_entry_price = price if price is not None else self.clock.current_price
        slipped_entry_price = base_entry_price * (1 + self.SLIPPAGE_MULTIPLIER) if side == 'buy' else base_entry_price * (1 - self.SLIPPAGE_MULTIPLIER)
        return {"success": True, "trade_id": f"dummy_{self._trade_id_counter}", "entry_price": slipped_entry_price}

    def close_position(self, position: Position) -> Dict[str, Any]:
        base_close_price = self.clock.current_price
        slipped_close_price = base_close_price * (1 - self.SLIPPAGE_MULTIPLIER) if position.is_long else base_close_price * (1 + self.SLIPPAGE_MULTIPLIER)
        gross_pnl = (slipped_close_price - position.entry_price) * position.amount if position.is_long else (position.entry_price - slipped_close_price) * position.amount
        
        commission_rate = 0.00055 
        total_commission = ((position.entry_price * position.amount) + (slipped_close_price * position.amount)) * commission_rate
        net_pnl = gross_pnl - total_commission
        
        # --- FIX 2 of 2: Update the engine's master capital ---
        if self.engine:
            # This line adds the P&L to the engine's master variable
            self.engine.master_realized_capital += net_pnl
        
        return {"success": True, "close_price": slipped_close_price, "pnl": net_pnl, "commission": total_commission}

    def modify_stop_loss(self, symbol: str, trade_id: str, new_stop_price: float) -> bool: 
        return True

    def start_data_stream(self, on_tick_callback: Callable[[Tick], None]): 
        pass