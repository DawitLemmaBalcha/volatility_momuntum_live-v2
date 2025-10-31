# trading_bot_shared.py

import logging
import time
import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from core.core_types import Position, Order, SimulationClock, TradeLog, Tick
from connectors.base_connector import BaseConnector

default_logger = logging.getLogger(__name__)

# Helper function for timestamp formatting
def format_time(timestamp: float) -> str:
    return pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')

class AdvancedAdaptiveGridTradingBot:
    def __init__(self, initial_capital: float, simulation_clock: SimulationClock, config_module, connector: BaseConnector, silent: bool = False):
        self.config = config_module
        
        self.initial_capital = initial_capital
        self.portfolio_initial_capital = initial_capital
        self.capital = initial_capital      # This will be GLOBAL realized capital
        self.peak_capital = initial_capital # This will be GLOBAL equity peak
        
        self.clock = simulation_clock
        self.connector = connector
        self.silent = silent

        self.symbol = self.config.SYMBOL
        self.kelly_fraction = self.config.KELLY_FRACTION
        self.max_position_size = self.config.MAX_POSITION_SIZE_PERCENT
        
        self.position_degrading_factor = getattr(self.config, 'POSITION_DEGRADING_FACTOR', 1.0) 
        self.trades_since_grid_rebuild = 0

        self.open_positions, self.grid_orders, self.trade_history = [], [], []
        self.max_drawdown, self.last_grid_setup_time, self.grid_size, self.trend_regime = 0.0, 0, 0.01, "uninitialized"

        self.total_grids_built = 0
        self.total_grids_traded = 0
        
        self.unrealized_pnl = 0.0 # This will be GLOBAL UPL
        
        self.capital_allocation = 1.0
        
        # --- *** NEW (Request 2): For Realized Drawdown *** ---
        # These will be injected by the master SimulationEngine
        self.realized_equity_curve = [initial_capital]
        # --- *** END NEW *** ---


    def update_strategy_on_30m(self, data_row: dict):
        # ... (This logic is correct for Model C, as peak/drawdown are handled by the 1-min sync) ...
        old_trend_regime = self.trend_regime
        self.trend_regime = self._get_trend_regime(data_row)
        rebuild_reason = None
        if self.trend_regime != old_trend_regime: rebuild_reason = f"Strategy Shift to {self.trend_regime}"
        elif not self.open_positions and self.clock.time() - self.last_grid_setup_time > getattr(self.config, 'GRID_MAX_LIFESPAN_SECONDS', 14400): rebuild_reason = "Grid Lifespan Expired"
        if rebuild_reason: self.setup_asymmetric_grid(rebuild_reason, data_row)
        self.update_kelly_fraction()
        
        # We still calculate max_drawdown here, using the *synced* values
        current_total_equity = self.capital + self.unrealized_pnl
        current_drawdown = (self.peak_capital - current_total_equity) / self.peak_capital * 100 if self.peak_capital > 0 else 0
        
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        if current_drawdown > self.config.MAX_DRAWDOWN_PERCENT: self.gradual_de_risk()

    def initialize_state(self, initial_data_row: dict):
        # ... (no changes in this function) ...
        self.trend_regime = self._get_trend_regime(initial_data_row)
        self.setup_asymmetric_grid("Initial Setup", initial_data_row)

    def _get_trend_regime(self, data_row: dict) -> str:
        # ... (no changes in this function) ...
        short_ma, long_ma, rsi, macd = data_row.get('short_ema', 0), data_row.get('long_ema', 0), data_row.get('rsi', 50), data_row.get('macd', 0)
        bb_upper, bb_lower, current_price = data_row.get('bb_upper', 0), data_row.get('bb_lower', 0), data_row.get('close', 0)
        trend = "neutral"
        if short_ma > long_ma * (1 + self.config.TREND_EMA_THRESHOLD_PERCENT / 100): trend = "up"
        elif short_ma < long_ma * (1 - self.config.TREND_EMA_THRESHOLD_PERCENT / 100): trend = "down"
        if rsi > self.config.RSI_OVERBOUGHT: trend = "strong_up"
        elif rsi < self.config.RSI_OVERSOLD: trend = "strong_down"
        if macd > 0 and "up" in trend: trend = "strong_up"
        elif macd < 0 and "down" in trend: trend = "strong_down"
        bb_width = (bb_upper - bb_lower) / current_price if current_price != 0 else 0
        regime = "trending" if bb_width > self.config.BB_REGIME_THRESHOLD else "ranging"
        return f"{trend}_{regime}"

    def setup_asymmetric_grid(self, reason: str, data_row: dict):
        # ... (no changes in this function) ...
        current_time = self.clock.time()
        current_time_str = format_time(current_time)
        if current_time - self.last_grid_setup_time < self.config.GRID_SETUP_COOLDOWN_SECONDS: return
        current_price = data_row['close']
        self.grid_orders.clear() 
        
        self.trades_since_grid_rebuild = 0
        
        old_grid_size = self.grid_size
        bb_width = (data_row['bb_upper'] - data_row['bb_lower']) / current_price if current_price > 0 else 0
        self.grid_size = max(self.config.GRID_MIN_SIZE_PERCENT, min(bb_width * getattr(self.config, 'GRID_BB_WIDTH_MULTIPLIER', 0.5), self.config.GRID_MAX_SIZE_PERCENT))
        total_grids = self.config.NUM_GRIDS
        if "ranging" in self.trend_regime: total_grids = 2
        ratio_normal, ratio_strong = getattr(self.config, 'TREND_GRID_RATIO_NORMAL', 0.6), getattr(self.config, 'TREND_GRID_RATIO_STRONG', 0.8)
        if "strong_up" in self.trend_regime: upper_grids, lower_grids = int(round(total_grids * ratio_strong)), total_grids - int(round(total_grids * ratio_strong))
        elif "up" in self.trend_regime: upper_grids, lower_grids = int(round(total_grids * ratio_normal)), total_grids - int(round(total_grids * ratio_normal))
        elif "strong_down" in self.trend_regime: lower_grids, upper_grids = int(round(total_grids * ratio_strong)), total_grids - int(round(total_grids * ratio_strong))
        elif "down" in self.trend_regime: lower_grids, upper_grids = int(round(total_grids * ratio_normal)), total_grids - int(round(total_grids * ratio_normal))
        else: upper_grids, lower_grids = total_grids // 2, total_grids - (total_grids // 2)

        new_grid_orders = [] 
        for i in range(1, upper_grids + 1):
            price = current_price * (1 + i * self.grid_size)
            new_order = Order(price, "buy")
            new_grid_orders.append(new_order)
            if not self.silent:
                default_logger.info(f"{current_time_str} - Created internal grid level: buy @ stop {new_order.price:,.2f}")
        for i in range(1, lower_grids + 1):
            price = current_price * (1 - i * self.grid_size)
            new_order = Order(price, "sell")
            new_grid_orders.append(new_order)
            if not self.silent:
                default_logger.info(f"{current_time_str} - Created internal grid level: sell @ stop {new_order.price:,.2f}")

        self.grid_orders = new_grid_orders 
        if self.grid_orders: 
            self.total_grids_built += len(self.grid_orders)

        if not self.silent:
            buy_grids = [o for o in self.grid_orders if o.order_type == 'buy']
            sell_grids = [o for o in self.grid_orders if o.order_type == 'sell']
            default_logger.info(f"\n{current_time_str} - [GRID REBUILT] Reason: {reason}\n  - Center Price:    {current_price:,.2f}\n  - Trend/Regime:    {self.trend_regime}\n  - Grid Size:       {self.grid_size:.3%} (was {old_grid_size:.3%})\n  - Setup:           {len(buy_grids)} Buy Grids | {len(sell_grids)} Sell Grids\n  - Buy Levels:      [ {', '.join([f'{o.price:,.2f}' for o in buy_grids])} ]\n  - Sell Levels:     [ {', '.join([f'{o.price:,.2f}' for o in sell_grids])} ]")
        self.last_grid_setup_time = current_time

    def check_exits_on_1m(self, current_price: float, atr: float):
        # ... (no changes in this function) ...
        current_time = self.clock.time()
        current_time_str = format_time(current_time) 
        for position in self.open_positions[:]:
            if current_time - position.entry_time > getattr(self.config, 'MAX_POSITION_DURATION_SECONDS', 86400):
                if not self.silent: default_logger.info(f"{current_time_str} - Triggering Time Stop for position {position.id}")
                self.close_position(position, current_price, "Time Stop"); continue
            if not (atr > 0 and pd.notna(atr)): continue
            if position.stop_loss is None: position.stop_loss = position.entry_price - (atr * self.config.ATR_INITIAL_STOP_MULTIPLIER) if position.is_long else position.entry_price + (atr * self.config.ATR_INITIAL_STOP_MULTIPLIER)
            activation_price = position.entry_price + (atr * self.config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER) if position.is_long else position.entry_price - (atr * self.config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER)

            if not position.is_trailing_active and ((position.is_long and current_price >= activation_price) or (not position.is_long and current_price <= activation_price)):
                position.is_trailing_active = True
                if not self.silent: default_logger.info(f"{current_time_str} - Trailing stop activated for position {position.id} at price {current_price:,.2f}.")

            if position.is_trailing_active:
                trailing_distance = atr * self.config.ATR_TRAILING_STOP_MULTIPLIER
                if position.is_long:
                    position.peak_price = max(position.peak_price, current_price)
                    new_stop = max(position.stop_loss, position.peak_price - trailing_distance)
                else:
                    position.peak_price = min(position.peak_price, current_price)
                    new_stop = min(position.stop_loss, position.peak_price + trailing_distance)
                position.stop_loss = new_stop

            if position.stop_loss is not None and ((position.is_long and current_price <= position.stop_loss) or (not position.is_long and current_price >= position.stop_loss)):
                exit_reason = "Trail Stop" if position.is_trailing_active else "ATR Stop"
                if not self.silent: default_logger.info(f"{current_time_str} - Triggering {exit_reason} for position {position.id} at price {current_price:,.2f} (Stop: {position.stop_loss:,.2f})")
                self.close_position(position, current_price, exit_reason)

    def check_entries_on_tick(self, tick: Tick, rsi_1m: float, macd_1m: float, volume_1m: float, volume_ma_1m: float, atr: float):
        # ... (no changes in this function) ...
        for order in self.grid_orders[:]:
            if (order.order_type == "buy" and tick.price >= order.price) or (order.order_type == "sell" and tick.price <= order.price):

                volume_ok = volume_1m > (volume_ma_1m * getattr(self.config, 'VOLUME_CONFIRMATION_FACTOR', 1.0))

                if order.order_type == "buy": rsi_ok, macd_ok = rsi_1m > self.config.RSI_BUY_CONFIRMATION, macd_1m > 0
                else: rsi_ok, macd_ok = rsi_1m < self.config.RSI_SELL_CONFIRMATION, macd_1m < 0

                if rsi_ok and macd_ok and volume_ok:
                    if not self.silent: default_logger.info(f"{format_time(tick.timestamp)} - Grid level triggered: {order.order_type} @ {order.price:,.2f} by tick price {tick.price:,.2f}")
                    
                    size, size_details_str = self.calculate_position_size(tick.price, return_details=True)
                    if not self.silent:
                        default_logger.info(f"{format_time(tick.timestamp)} - {size_details_str}")
                    
                    details = {
                        "price_ok": f"{tick.price:,.2f} {' >=' if order.order_type == 'buy' else ' <='} {order.price:,.2f}",
                        "rsi_ok": rsi_ok, "rsi_val": rsi_1m, "rsi_req": getattr(self.config, f'RSI_{"BUY" if order.order_type == "buy" else "SELL"}_CONFIRMATION'),
                        "macd_ok": macd_ok, "macd_val": macd_1m,
                        "volume_ok": volume_ok,
                        "volume_val": volume_1m,
                        "volume_req": volume_ma_1m * getattr(self.config, 'VOLUME_CONFIRMATION_FACTOR', 1.0)
                    }
                    
                    self.execute_trade(order, size, details, atr)
                    self.grid_orders.remove(order) 

    def execute_trade(self, order: Order, amount: float, confirmation_details: dict, atr: float):
        # ... (no changes in this function) ...
        current_time_str = format_time(self.clock.time()) 
        
        if len(self.open_positions) >= self.config.MAX_POSITIONS or amount <= 0: 
            if amount <= 0 and not self.silent:
                default_logger.info(f"{current_time_str} - Skipping trade. Calculated size is {amount:.5f}.")
            return
        
        direction = 'LONG' if order.order_type == 'buy' else 'SHORT'
        if not self.silent: default_logger.info(f"{current_time_str} - Attempting to place order via connector: {order.order_type} {amount:.5f} {self.symbol}")

        result = self.connector.place_order(symbol=self.symbol, side=order.order_type, order_type="Market", qty=amount, price=order.price)
        if not result or not result.get("success"):
            if not self.silent: default_logger.error(f"{current_time_str} - Connector failed to place order.")
            return

        self.total_grids_traded += 1
        self.trades_since_grid_rebuild += 1

        entry_price = result['entry_price']
        new_position = Position(entry_price, amount, direction == 'LONG')
        new_position.symbol = self.symbol 
        new_position.id, new_position.entry_time, new_position.entry_regime = result['trade_id'], self.clock.time(), self.trend_regime 
        stop_distance = atr * self.config.ATR_INITIAL_STOP_MULTIPLIER if (atr > 0 and pd.notna(atr)) else entry_price * 0.01 
        new_position.kelly_risk_usd, new_position.atr_risk_usd = entry_price * new_position.amount, stop_distance * new_position.amount
        self.open_positions.append(new_position)

        if not self.silent:
            initial_stop_price = (entry_price - stop_distance) if new_position.is_long else (entry_price + stop_distance)
            default_logger.info(f"\n{current_time_str} - [TRADE OPEN] | {direction} @ {entry_price:,.2f}\n  - Position ID:    {new_position.id}\n  - Entry Regime:   {new_position.entry_regime}\n  - Risk Profile:\n    - Position Value (Kelly Risk): ${new_position.kelly_risk_usd:,.2f}\n    - Stop-Loss Risk (ATR Risk):  -${new_position.atr_risk_usd:,.2f}\n    - Initial Stop Price:          ${initial_stop_price:,.2f}\n  - Justification:\n    - Price ({confirmation_details['price_ok']}) -> [PASS]\n    - RSI ({confirmation_details['rsi_val']:.2f} {'>' if direction == 'LONG' else '<'} {confirmation_details['rsi_req']}) -> [{'PASS' if confirmation_details['rsi_ok'] else 'FAIL'}]\n    - MACD ({confirmation_details['macd_val']:.2f} {'> 0' if direction == 'LONG' else '< 0'}) -> [{'PASS' if confirmation_details['macd_ok'] else 'FAIL'}]\n    - Volume ({confirmation_details['volume_val']:,.2f} > {confirmation_details['volume_req']:,.2f}) -> [{'PASS' if confirmation_details['volume_ok'] else 'FAIL'}]")

    def gradual_de_risk(self):
        # ... (no changes in this function) ...
        current_time_str = format_time(self.clock.time()) 
        
        current_total_equity = self.capital + self.unrealized_pnl
        risk_reduction = min(0.5, ((self.peak_capital - current_total_equity) / self.peak_capital) / self.config.MAX_DRAWDOWN_PERCENT) if self.peak_capital > 0 and self.config.MAX_DRAWDOWN_PERCENT > 0 else 0
        
        if risk_reduction > 0: 
            self.max_position_size *= (1 - risk_reduction)
            self.kelly_fraction *= (1 - risk_reduction)
            if not self.silent: default_logger.info(f"{current_time_str} - De-risking due to drawdown. New max size: {self.max_position_size:.2%}, New Kelly: {self.kelly_fraction:.2f}")

    def close_position(self, position: Position, current_price: float, reason: str):
        current_time_str = format_time(self.clock.time()) 
        capital_before = self.capital
        
        if not self.silent: default_logger.info(f"{current_time_str} - Attempting to close position {position.id} via connector. Reason: {reason}")

        close_result = self.connector.close_position(position)
        if not close_result or not close_result.get("success"):
            if not self.silent: default_logger.error(f"{current_time_str} - Connector failed to close position {position.id}")
            return

        profit = close_result['pnl']
        
        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        is_model_a_backtest = is_backtest and self.capital_allocation == 1.0
        
        if is_model_a_backtest:
            # This should not be called in Model C, but we leave it for safety
            self.capital += profit
        
        profit_percent = (profit / (position.entry_price * position.amount)) * 100 if position.entry_price * position.amount != 0 else 0
        self.trade_history.append(TradeLog(entry_price=position.entry_price, close_price=close_result['close_price'], entry_time=position.entry_time, close_time=self.clock.time(), pnl_percent=profit_percent, pnl_cash=profit, reason=reason, direction='LONG' if position.is_long else 'SHORT', size=position.amount, kelly_risk_usd=position.kelly_risk_usd, atr_risk_usd=position.atr_risk_usd, entry_regime=position.entry_regime))
        if position in self.open_positions: self.open_positions.remove(position)

        if not self.silent:
            duration_s = self.clock.time() - position.entry_time
            duration_str = time.strftime('%Hh %Mm %Ss', time.gmtime(duration_s)) if duration_s >=0 else "N/A"
            gross_pnl = profit + close_result.get('commission', 0)
            
            capital_update_log = ""
            if is_model_a_backtest:
                capital_update_log = f"${self.capital:,.2f}" 
            else:
                capital_update_log = "(Waiting for next sync...)" 
                
            default_logger.info(f"\n{current_time_str} - [TRADE CLOSE] | {'LONG' if position.is_long else 'SHORT'} | Reason: {reason}\n"
                                f"  - Position ID:      {position.id}\n"
                                f"  - Duration:         {duration_str}\n"
                                f"  - Entry Price:      {position.entry_price:,.2f}\n"
                                f"  - Close Price:      {close_result['close_price']:,.2f}\n"
                                f"  - Gross PnL:        ${gross_pnl:,.2f}\n"
                                f"  - Commission Paid:  -${close_result.get('commission', 0):,.2f}\n"
                                f"  - Net PnL:          ${profit:,.2f} ({profit_percent:.2f}%)\n"
                                f"  - Portfolio Impact: Capital ${capital_before:,.2f} -> {capital_update_log}")

    def calculate_position_size(self, price: float, return_details: bool = False) -> float | Tuple[float, str]:
        # ... (no changes in this function) ...
        if price <= 0:
            details_str = "Pos Size Calc: Price <= 0, returning 0.0"
            return (0.0, details_str) if return_details else 0.0

        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        effective_capital_base = 0.0
        details_log_header = ""

        is_model_a_backtest = is_backtest and self.capital_allocation == 1.0
        
        if is_model_a_backtest:
            # effective_capital_base = self.capital + self.unrealized_pnl # <<< USE THE INJECTED 'self.unrealized_pnl'
            # effective_capital = min(effective_capital_base, self.peak_capital)
            # details_log_header = (f"Pos Size Calc (Model A): RealizedCap={self.capital:,.2f}, UPL={self.unrealized_pnl:,.2f} -> " # <<< USE 'self.unrealized_pnl'
            #                       f"EffCapBase={effective_capital_base:,.2f}, PeakCap={self.peak_capital:,.2f} -> ")
            
            # --- EVEN BETTER, just make it use the 'else' block's logic ---
            # which correctly uses the injected values.
            
            effective_capital_base = self.capital + self.unrealized_pnl
            effective_capital = min(effective_capital_base, self.peak_capital)
            
            details_log_header = (f"Pos Size Calc (Model A/C): SyncedTotalRealized={self.capital:,.2f}, SyncedTotalUPL={self.unrealized_pnl:,.2f} -> "
                                  f"EffCapBase={effective_capital_base:,.2f}, SyncedTotalPeak={self.peak_capital:,.2f} -> ")


        else:
            effective_capital_base = self.capital + self.unrealized_pnl
            effective_capital = min(effective_capital_base, self.peak_capital)
            
            details_log_header = (f"Pos Size Calc (Model C/Live): SyncedTotalRealized={self.capital:,.2f}, SyncedTotalUPL={self.unrealized_pnl:,.2f} -> "
                                  f"EffCapBase={effective_capital_base:,.2f}, SyncedTotalPeak={self.peak_capital:,.2f} -> ")

        effective_capital = max(0.0, effective_capital) 
        
        allocated_capital_for_sizing = effective_capital * self.capital_allocation

        max_value_limit = allocated_capital_for_sizing * self.max_position_size
        kelly_value = allocated_capital_for_sizing * self.kelly_fraction
        final_pos_value = min(max_value_limit, kelly_value)

        base_amount = final_pos_value / price if price > 0 else 0.0
        
        trade_count_in_set = self.trades_since_grid_rebuild 
        
        degrading_multiplier = pow(self.position_degrading_factor, trade_count_in_set)
        
        final_amount = base_amount * degrading_multiplier
        
        details_str = (f"{details_log_header}"
                       f"AllocFactor={self.capital_allocation:.2f} -> AllocCapForSizing={allocated_capital_for_sizing:,.2f}\n"
                       f"             -> MaxVal={max_value_limit:,.2f}, KellyVal={kelly_value:,.2f} (MaxSz={self.max_position_size:.3f}, KellyFrac={self.kelly_fraction:.3f}) \n"
                       f"             -> FinalVal={final_pos_value:,.2f} -> BaseAmount={base_amount:.5f}\n"
                       f"             -> Degrading: Factor={self.position_degrading_factor}, TradeCountInSet={trade_count_in_set} -> Multiplier={degrading_multiplier:.4f}\n"
                       f"             -> Final Amount={final_amount:.5f}")

        return (final_amount, details_str) if return_details else final_amount

    def update_kelly_fraction(self):
        # ... (no changes in this function) ...
        if len(self.trade_history) < self.config.KELLY_LOOKBACK: return
        recent_pnls = [t.pnl_percent / 100 for t in self.trade_history[-self.config.KELLY_LOOKBACK:]]
        wins = [p for p in recent_pnls if p > 0]
        if not wins:
            self.kelly_fraction = self.config.KELLY_MIN_FRACTION; return 
        win_rate = len(wins) / len(recent_pnls)
        avg_win = np.mean(wins)
        losses = [p for p in recent_pnls if p <= 0]
        avg_loss = abs(np.mean(losses)) if losses else 0.00001 

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else avg_win / 0.00001 

        kelly_raw = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0.0
        old_kelly = self.kelly_fraction
        self.kelly_fraction = max(self.config.KELLY_MIN_FRACTION, min(kelly_raw, self.config.KELLY_FRACTION)) 

    def log_performance(self, print_log: bool = False, logger=None) -> dict:
        if logger is None: logger = default_logger
        
        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        is_model_c_run = is_backtest and (self.capital_allocation != 1.0)

        if is_model_c_run:

            pnl_cash = self.capital - self.portfolio_initial_capital
            total_return_pct = (pnl_cash / self.portfolio_initial_capital) * 100 if self.portfolio_initial_capital > 0 else 0
        else:

            pnl_cash = self.capital - self.initial_capital
            total_return_pct = (pnl_cash / self.initial_capital) * 100 if self.initial_capital > 0 else 0


        metrics = {"total_return_pct": total_return_pct, "pnl_cash": pnl_cash, "max_drawdown": self.max_drawdown, "sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0, "win_rate": 0, "profit_factor": 0, "total_trades": len(self.trade_history), "regime_performance": {},
                   "total_grids_built": self.total_grids_built,
                   "total_grids_traded": self.total_grids_traded,
                   "grid_hit_rate": (self.total_grids_traded / self.total_grids_built * 100) if self.total_grids_built > 0 else 0,
                   # --- *** NEW (Request 2): Add new metric *** ---
                   "realized_max_drawdown": 0.0 
                   # --- *** END NEW *** ---
                   }
                   
        if len(self.trade_history) > 1:
            metrics['win_rate'] = sum(1 for t in self.trade_history if t.pnl_cash > 0) / len(self.trade_history) * 100
            gross_profit = sum(t.pnl_cash for t in self.trade_history if t.pnl_cash > 0) or 1
            gross_loss = abs(sum(t.pnl_cash for t in self.trade_history if t.pnl_cash < 0)) or 1
            metrics['profit_factor'] = gross_profit / gross_loss
            if self.trade_history:
                 trade_df = pd.DataFrame([vars(t) for t in self.trade_history])
                 trade_df['close_datetime'] = pd.to_datetime(trade_df['close_time'], unit='s')
                 trade_df['daily_return'] = trade_df['pnl_percent'] / 100
                 daily_returns = trade_df.set_index('close_datetime')['daily_return'].resample('D').sum() 
                 if len(daily_returns) > 1:
                     avg_daily_return = daily_returns.mean()
                     std_daily_return = daily_returns.std()
                     metrics['sharpe_ratio'] = (avg_daily_return / std_daily_return) * np.sqrt(365) if std_daily_return > 0 else 0 

                     neg_daily_returns = daily_returns[daily_returns < 0]
                     downside_dev_daily = neg_daily_returns.std() if len(neg_daily_returns) > 1 else 0
                     metrics['sortino_ratio'] = (avg_daily_return / downside_dev_daily) * np.sqrt(365) if downside_dev_daily > 0 else (999.0 if avg_daily_return > 0 else 0)

            if self.max_drawdown > 0: metrics['calmar_ratio'] = total_return_pct / self.max_drawdown
            elif total_return_pct > 0: metrics['calmar_ratio'] = 999.0
            else: metrics['calmar_ratio'] = 0.0

            # --- *** NEW (Request 2): Calculate Realized Drawdown *** ---
            # self.realized_equity_curve is injected by the master engine
            if len(self.realized_equity_curve) > 1:
                realized_equity_series = pd.Series(self.realized_equity_curve)
                realized_peak_series = realized_equity_series.expanding().max()
                realized_drawdown_series = (realized_equity_series - realized_peak_series) / realized_peak_series
                metrics['realized_max_drawdown'] = abs(realized_drawdown_series.min()) * 100
            # --- *** END NEW *** ---

            regime_trades = defaultdict(list)
            for t in self.trade_history: regime_trades[t.entry_regime].append(t)
            for regime, trades in regime_trades.items():
                if not trades: continue 
                regime_pnl = sum(t.pnl_cash for t in trades)
                regime_wins = sum(1 for t in trades if t.pnl_cash > 0)
                regime_gross_profit = sum(t.pnl_cash for t in trades if t.pnl_cash > 0) or 1
                regime_gross_loss = abs(sum(t.pnl_cash for t in trades if t.pnl_cash < 0)) or 1
                metrics["regime_performance"][regime] = {"total_trades": len(trades), "win_rate": f"{(regime_wins / len(trades)) * 100 if trades else 0:.2f}%", "profit_factor": f"{regime_gross_profit / regime_gross_loss:.2f}", "net_pnl": f"${regime_pnl:,.2f}"}

        metrics["performance_log_str"] = self._get_performance_log_str(metrics) 
        if print_log: logger.info(metrics["performance_log_str"])
        return metrics

    def _get_performance_log_str(self, metrics: dict) -> str:
        winning_trades, losing_trades = [t.pnl_cash for t in self.trade_history if t.pnl_cash > 0], [t.pnl_cash for t in self.trade_history if t.pnl_cash < 0]
        wins, losses = len(winning_trades), len(losing_trades)
        avg_winner_usd, avg_loser_usd = np.mean(winning_trades) if wins > 0 else 0, np.mean(losing_trades) if losses > 0 else 0
        avg_pos_value = np.mean([t.kelly_risk_usd for t in self.trade_history]) if self.trade_history else 0
        open_longs, open_shorts = sum(1 for p in self.open_positions if p.is_long), sum(1 for p in self.open_positions if not p.is_long)

        is_backtest = self.connector.__class__.__name__ == 'DummyConnector'
        
        capital_source = "(Backtest Model A)"
        if not is_backtest:
            capital_source = "(Synced from exchange)"
        elif self.capital_allocation != 1.0:
            capital_source = "(Synced from Model C Engine)"

        log_msg = (f"\n{'='*20} PORTFOLIO STATUS | {format_time(self.clock.time())} {'='*20}\n" 
                   f"  Core Metrics:\n"
                   # Note: PnL and Capital are PORTFOLIO-WIDE in Model C
                   f"    - Total Return:    {metrics['total_return_pct']:,.2f}%\n"
                   f"    - PnL:             ${metrics['pnl_cash']:,.2f}\n"
                   f"    - Current Capital: ${self.capital:,.2f} {capital_source}\n"
                   f"  Risk & Performance:\n"
                   # Note: Peaks and Drawdowns are PORTFOLIO-WIDE in Model C
                   f"    - Peak Equity:     ${self.peak_capital:,.2f} {capital_source}\n"
                   # --- *** NEW (Request 2): Modify log string *** ---
                   f"    - Max Equity Drawdown:   -{metrics['max_drawdown']:.2f}% (incl. UPL)\n"
                   f"    - Max Realized Drawdown: -{metrics['realized_max_drawdown']:.2f}% (prop firm style)\n"
                   # --- *** END NEW *** ---
                   f"    - Sharpe Ratio:    {metrics['sharpe_ratio']:.2f} (Daily Approx.)\n"
                   f"    - Sortino Ratio:   {metrics['sortino_ratio']:.2f} (Daily Approx.)\n"
                   f"    - Calmar Ratio:    {metrics['calmar_ratio']:.2f}\n"
                   f"  Trade Stats (This Bot Only):\n" 
                   f"    - Total Trades:    {metrics['total_trades']}\n"
                   f"    - Win Rate:        {metrics.get('win_rate', 0):.2f}% ({wins} W / {losses} L)\n"
                   f"    - Profit Factor:   {metrics.get('profit_factor', 0):.2f}\n"
                   f"    - Avg Winner ($):  ${avg_winner_usd:,.2f}\n"
                   f"    - Avg Loser ($):   ${avg_loser_usd:,.2f}\n"
                   f"  Grid Stats (This Bot Only):\n" 
                   f"    - Total Grids Built: {metrics['total_grids_built']}\n"
                   f"    - Total Grids Traded:{metrics['total_grids_traded']}\n"
                   f"    - Hit Rate:          {metrics['grid_hit_rate']:.2f}%\n"
                   f"  Risk Stats (This Bot Only):\n" 
                   f"    - Avg Position Value: ${avg_pos_value:,.2f}\n"
                   f"  Current State:\n"
                   f"    - Open Positions:  {len(self.open_positions)} ({open_longs} L, {open_shorts} S)\n"
                   f"    - Trend/Regime:    {self.trend_regime}\n"
                   f"    - Kelly Fraction:  {self.kelly_fraction:.3f}\n"
                   f"    - Cap Allocation:  {self.capital_allocation:.2%}\n") 

        if metrics["regime_performance"]:
            log_msg += f"  {'='*20} Regime Performance Breakdown (This Bot Only) {'='*11}\n" 
            log_msg += f"  {'Regime':<25} | {'Trades':>8} | {'Win Rate':>10} | {'Profit Factor':>15} | {'Net PnL':>15}\n"
            log_msg += f"  {'-'*78}\n"
            for regime, stats in sorted(metrics["regime_performance"].items()):
                log_msg += f"  {regime:<25} | {stats['total_trades']:>8} | {stats['win_rate']:>10} | {stats['profit_factor']:>15} | {stats['net_pnl']:>15}\n"

        log_msg += f"{'='*78}" 
        return log_msg