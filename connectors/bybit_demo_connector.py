# connectors/bybit_demo_connector.py

import logging
import time
from pybit.unified_trading import HTTP, WebSocket
from .base_connector import BaseConnector
from typing import Dict, Any, Callable
from core_types import Tick, Position

class BybitDemoConnector(BaseConnector):
    """
    A dedicated, fully functional connector for paper trading on the Bybit Testnet.
    This connector is hardcoded with testnet=True for safety.
    """

    def __init__(self, api_key: str, api_secret: str, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.ws = None
        
        self.logger.warning("Initializing Bybit TESTNET Connector. This connector will NOT execute live trades.")
        
        # HTTP session is hardcoded to testnet=True
        self.session = HTTP(
            testnet=True,
            api_key=self.api_key,
            api_secret=self.api_secret,
        )

    def connect(self):
        self.logger.info("Connecting to Bybit Testnet...")
        pass

    def disconnect(self):
        self.logger.info("Disconnecting from Bybit Testnet...")
        if self.ws:
            self.ws.exit()

    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float = None, stop_loss: float = None) -> Dict[str, Any]:
        """
        Places a market order on Bybit Testnet.
        """
        side_bybit = "Buy" if side == "buy" else "Sell"
        symbol_bybit = symbol.replace('/', '')

        self.logger.info(f"PLACING TESTNET ORDER: {side_bybit} {qty} {symbol_bybit}")
        
        try:
            response = self.session.place_order(
                category="linear",
                symbol=symbol_bybit,
                side=side_bybit,
                orderType="Market",
                qty=str(qty),
            )
            
            if response and response.get('retCode') == 0:
                order_id = response['result']['orderId']
                time.sleep(0.5) # Allow time for the market order to fill
                
                trade_history = self.session.get_order_history(category="linear", orderId=order_id, limit=1)
                
                if trade_history and trade_history['result']['list']:
                    entry_price = float(trade_history['result']['list'][0]['avgPrice'])
                    self.logger.info(f"Successfully placed testnet order {order_id} at avg price {entry_price}")
                    return {"success": True, "trade_id": order_id, "entry_price": entry_price}
                else:
                    self.logger.error(f"Placed order {order_id} but could not fetch execution price.")
                    return {"success": False, "error": "Could not fetch execution price"}
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"Failed to place testnet order: {error_msg}")
                return {"success": False, "error": error_msg}

        except Exception as e:
            self.logger.error(f"An exception occurred while placing testnet order: {e}")
            return {"success": False, "error": str(e)}

    def close_position(self, position: Position) -> Dict[str, Any]:
        """
        Closes a position on Bybit Testnet by placing an opposing market order.
        """
        symbol_bybit = position.symbol.replace('/', '') # Assuming position object has a symbol
        side_to_close = "Sell" if position.is_long else "Buy"
        
        self.logger.info(f"CLOSING TESTNET POSITION: {side_to_close} {position.amount} {symbol_bybit}")

        try:
            response = self.session.place_order(
                category="linear",
                symbol=symbol_bybit,
                side=side_to_close,
                orderType="Market",
                qty=str(position.amount),
                reduce_only=True
            )
            
            if response and response.get('retCode') == 0:
                order_id = response['result']['orderId']
                time.sleep(0.5)
                
                trade_history = self.session.get_order_history(category="linear", orderId=order_id, limit=1)
                
                if trade_history and trade_history['result']['list']:
                    close_price = float(trade_history['result']['list'][0]['avgPrice'])
                    pnl = (close_price - position.entry_price) * position.amount if position.is_long else (position.entry_price - close_price) * position.amount
                    commission = 0 # In a real scenario, you might estimate this or fetch from trade history

                    self.logger.info(f"Successfully closed testnet position {position.id} at avg price {close_price}")
                    return {"success": True, "close_price": close_price, "pnl": pnl, "commission": commission}
                else:
                    return {"success": False, "error": "Could not fetch execution price for closing order"}
            else:
                error_msg = response.get('retMsg', 'Unknown error')
                self.logger.error(f"Failed to close testnet position: {error_msg}")
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"An exception occurred while closing testnet position: {e}")
            return {"success": False, "error": str(e)}

    def modify_stop_loss(self, symbol: str, trade_id: str, new_stop_price: float) -> bool:
        self.logger.info(f"MODIFYING TESTNET STOP for {trade_id} to {new_stop_price} (not implemented)")
        return True

    def start_data_stream(self, symbol: str, on_tick_callback: Callable[[Tick, str], None]):
        """
        Starts a WebSocket connection to stream live public trades for the given symbol.
        """
        self.logger.info(f"Initializing Bybit Testnet WebSocket for {symbol}...")
        self.ws = WebSocket(
            testnet=True,
            channel_type="linear",
        )
        
        symbol_bybit = symbol.replace('/', '')
        
        self.ws.trade_stream(
            symbol=symbol_bybit,
            callback=lambda msg: self._handle_websocket_message(msg, on_tick_callback, symbol)
        )
        self.logger.info(f"WebSocket subscribed to public trades for {symbol_bybit}")

    def _handle_websocket_message(self, msg: Dict, on_tick_callback: Callable, symbol: str):
        """
        Processes messages from the WebSocket and formats them into a standard Tick object.
        """
        try:
            if 'data' in msg:
                for trade in msg['data']:
                    tick = Tick(
                        timestamp=float(trade['T']) / 1000,
                        price=float(trade['p']),
                        volume=float(trade['v']),
                        candle_volume=0
                    )
                    # Pass both the tick and the original symbol to the callback
                    on_tick_callback(tick, symbol)
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")


# ... (inside the BybitDemoConnector class) ...

    def get_wallet_balance(self, coin: str) -> float:
        """
        Fetches the available balance for a specific coin from the Unified Trading Account.
        This version asks for all balances and then finds the specific coin for robustness.
        """
        try:
            # --- THE FIX: Ask for ALL coins in the UNIFIED account ---
            response = self.session.get_wallet_balance(
                accountType="UNIFIED"
            )
            
            if response and response.get('retCode') == 0:
                # The balance data is in response['result']['list']
                balances = response['result']['list']
                
                # Check if the list is not empty
                if not balances:
                    self.logger.warning("API returned an empty list of balances for the UNIFIED account.")
                    return 0.0

                # Find the specific coin in the list of all balances
                for balance_info in balances:
                    if balance_info['coin'] == coin:
                        # Use 'walletBalance' which represents the total equity of that coin
                        balance = float(balance_info['walletBalance'])
                        self.logger.info(f"Successfully fetched wallet balance: {balance} {coin}")
                        return balance
                
                # If the loop finishes without finding the coin
                self.logger.warning(f"Could not find balance for coin '{coin}' in the UNIFIED account balances list.")
                return 0.0
            else:
                self.logger.error(f"Failed to fetch wallet balance: {response.get('retMsg', 'Unknown error')}")
                return 0.0
        except Exception as e:
            self.logger.error(f"An exception occurred while fetching wallet balance: {e}")
            return 0.0