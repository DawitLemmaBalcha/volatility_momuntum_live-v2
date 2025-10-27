# backtest_runner.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
# import random <-- 1. REMOVED
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List
from trading_bot import AdvancedAdaptiveGridTradingBot
from core_types import SimulationClock, Tick, Position

# DummyConnector remains a private helper class
class DummyConnector(ABC):
    SLIPPAGE_PERCENT = 0.001
    def __init__(self, clock: SimulationClock, params: dict):
        self.clock, self.params, self._trade_id_counter = clock, params, 0
    def connect(self): pass
    def disconnect(self): pass
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, price: float = None, stop_loss: float = None) -> Dict[str, Any]:
        self._trade_id_counter += 1
        base_entry_price = price if price is not None else self.clock.current_price
        slipped_entry_price = base_entry_price * (1 + self.SLIPPAGE_PERCENT) if side == 'buy' else base_entry_price * (1 - self.SLIPPAGE_PERCENT)
        return {"success": True, "trade_id": f"dummy_{self._trade_id_counter}", "entry_price": slipped_entry_price}
    def close_position(self, position: Position) -> Dict[str, Any]:
        base_close_price = self.clock.current_price
        slipped_close_price = base_close_price * (1 - self.SLIPPAGE_PERCENT) if position.is_long else base_close_price * (1 + self.SLIPPAGE_PERCENT)
        gross_pnl = (slipped_close_price - position.entry_price) * position.amount if position.is_long else (position.entry_price - slipped_close_price) * position.amount
        commission_rate = self.params.get('BYBIT_TAKER_FEE', 0.00055)
        total_commission = ((position.entry_price * position.amount) + (slipped_close_price * position.amount)) * commission_rate
        net_pnl = gross_pnl - total_commission
        return {"success": True, "close_price": slipped_close_price, "pnl": net_pnl, "commission": total_commission}
    def modify_stop_loss(self, symbol: str, trade_id: str, new_stop_price: float) -> bool: return True
    def start_data_stream(self, on_tick_callback: Callable[[Tick], None]): pass


class SimulationEngine:
    """
    Handles the time-synchronized, concurrent simulation of one or more trading bots.
    """
    def __init__(self, df_1m_full: pd.DataFrame, bots: List[AdvancedAdaptiveGridTradingBot], logger=None):
        self.df_1m_full = df_1m_full.reset_index(drop=True)
        self.bots = bots
        self.logger = logger or logging.getLogger(__name__)
        self.portfolio_equity_curve = []
        self.initial_portfolio_capital = sum(bot.initial_capital for bot in self.bots)

    def _prepare_data(self):
        # Data prep is done once for all bots, assuming they use the same indicator periods for now.
        # A more advanced version could handle different indicator sets per bot.
        bot_config = self.bots[0].config
        
        df_30m = self.df_1m_full.set_index('timestamp').resample('30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).reset_index().dropna()
        df_30m['short_ema'] = ta.ema(df_30m['close'], length=bot_config.SHORT_EMA_PERIOD)
        df_30m['long_ema'] = ta.ema(df_30m['close'], length=bot_config.LONG_EMA_PERIOD)
        df_30m['rsi'] = ta.rsi(df_30m['close'], length=bot_config.RSI_PERIOD)
        df_30m['atr'] = ta.atr(df_30m['high'], df_30m['low'], df_30m['close'], length=bot_config.ATR_PERIOD)
        macd_30m = ta.macd(df_30m['close'], fast=bot_config.MACD_FAST_PERIOD, slow=bot_config.MACD_SLOW_PERIOD, signal=bot_config.MACD_SIGNAL_PERIOD)
        df_30m['macd'] = macd_30m[f'MACD_{bot_config.MACD_FAST_PERIOD}_{bot_config.MACD_SLOW_PERIOD}_{bot_config.MACD_SIGNAL_PERIOD}']
        bbands_30m = ta.bbands(df_30m['close'], length=bot_config.BOLLINGER_PERIOD, std=bot_config.BOLLINGER_STD_DEV)
        df_30m['bb_lower'] = bbands_30m[f'BBL_{bot_config.BOLLINGER_PERIOD}_{bot_config.BOLLINGER_STD_DEV}']
        df_30m['bb_upper'] = bbands_30m[f'BBU_{bot_config.BOLLINGER_PERIOD}_{bot_config.BOLLINGER_STD_DEV}']
        
        df_1m = self.df_1m_full.copy()
        # --- FIX: Shift all 1m indicators to prevent lookahead ---
        # We use shift(1) so that the data for a given row (e.g., 10:01) is based
        # on the candle that *closed* at 10:00.
        df_1m['rsi_1m'] = ta.rsi(df_1m['close'], length=bot_config.CONFIRMATION_RSI_PERIOD).shift(1)
        df_1m['volume_ma_1m'] = df_1m['volume'].rolling(window=bot_config.CONFIRMATION_VOLUME_MA_PERIOD).mean().shift(1)
        macd_1m = ta.macd(df_1m['close'], fast=bot_config.CONFIRMATION_MACD_FAST_PERIOD, slow=bot_config.CONFIRMATION_MACD_SLOW_PERIOD, signal=bot_config.CONFIRMATION_MACD_SIGNAL_PERIOD)
        df_1m['macd_1m'] = macd_1m[f'MACD_{bot_config.CONFIRMATION_MACD_FAST_PERIOD}_{bot_config.CONFIRMATION_MACD_SLOW_PERIOD}_{bot_config.CONFIRMATION_MACD_SIGNAL_PERIOD}'].shift(1)
        
        indicator_cols = ['timestamp', 'short_ema', 'long_ema', 'rsi', 'atr', 'macd', 'bb_lower', 'bb_upper']
        # --- FIX: Shift 30m indicators to prevent lookahead ---
        # We shift them here so that when merge_asof finds the 10:00 30m bar,
        # it's using the indicators calculated at 09:30 (the last *closed* bar).
        for col in indicator_cols:
            if col != 'timestamp':
                df_30m[col] = df_30m[col].shift(1)
                
        df_merged = pd.merge_asof(df_1m.sort_values('timestamp'), df_30m[indicator_cols].sort_values('timestamp'), on='timestamp', direction='backward')
        
        return df_merged.dropna()

    def run(self):
        df_merged = self._prepare_data()
        
        if len(df_merged) < 100:
            self.logger.warning("Backtest failed: Insufficient data after indicator calculation.")
            return None, []

        sim_start_time = df_merged['timestamp'].iloc[0].timestamp()
        clock = SimulationClock(start_time=sim_start_time)
        dummy_connector = DummyConnector(clock, {})

        first_row = df_merged.iloc[0]
        for bot in self.bots:
            bot.clock = clock
            bot.connector = dummy_connector
            bot.initialize_state(first_row.to_dict())

        # --- HIGH-SPEED CONCURRENT SIMULATION LOOP ---
        # random.seed(42) # <-- 2. REMOVED: No longer needed as path is deterministic
        for i in range(1, len(df_merged)):
            current_row, prev_row = df_merged.iloc[i], df_merged.iloc[i-1]
            
            # Update 30-minute strategies
            if current_row['timestamp'].floor('30min') > prev_row['timestamp'].floor('30min'):
                for bot in self.bots:
                    # <-- 3. FIX: Use PREVIOUS row for 30m strategy decisions
                    bot.update_strategy_on_30m(prev_row.to_dict())
            
            # Simulate ticks for the current 1m candle
            o, h, l, c, v, base_ts = current_row['open'], current_row['high'], current_row['low'], current_row['close'], current_row['volume'], current_row['timestamp'].timestamp()
            
            # --- START: Deterministic Worst-Case Path Simulation ---
            #
            # We determine the "worst path" based on the bar's direction.
            #
            # If the bar is bullish (Close > Open):
            # The most destructive path for stops is Open -> Low -> High -> Close.
            # This path tries to stop you out at the bottom *before* moving up.
            #
            # If the bar is bearish (Close <= Open):
            # The most destructive path for stops is Open -> High -> Low -> Close.
            # This path tries to hit your buy entries *before* crashing down.
            
            if c > o:
                # Bullish Bar: Test stops first (O -> L -> H -> C)
                key_prices = [o, l, h, c]
            else:
                # Bearish Bar: Test entries first (O -> H -> L -> C)
                key_prices = [o, h, l, c]
            # --- END: Deterministic Worst-Case Path Simulation ---
            
            
            # <-- 5. FIX: Get all indicator values from the PREVIOUS, CLOSED candle
            # These values are constant for all 4 ticks within the current candle.
            rsi_1m_val = prev_row['rsi_1m']
            macd_1m_val = prev_row['macd_1m']
            volume_ma_1m_val = prev_row['volume_ma_1m']
            volume_1m_val = prev_row['volume']
            atr_val = prev_row['atr'] # 30m ATR is already from the past via merge_asof

            for price in key_prices:
                # ...
                tick = Tick(timestamp=base_ts, price=price, volume=0, candle_volume=v) # candle_volume is now ignored by the bot's logic
                clock.current_time, clock.current_price = tick.timestamp, price
                
                for bot in self.bots:
                    # --- MODIFY THIS CALL ---
                    bot.check_entries_on_tick(tick, rsi_1m_val, macd_1m_val, volume_1m_val, volume_ma_1m_val, atr_val)
                    bot.check_exits_on_1m(price, atr_val)

            # Record portfolio equity at the end of each 1m candle
            current_portfolio_equity = sum(bot.capital for bot in self.bots)
            self.portfolio_equity_curve.append(current_portfolio_equity)

        # --- Final Performance Calculation ---
        portfolio_results = self._calculate_portfolio_metrics()
        individual_results = [bot.log_performance(print_log=False) for bot in self.bots]
        
        return portfolio_results, individual_results

    def _calculate_portfolio_metrics(self):
        equity_series = pd.Series(self.portfolio_equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
        
        final_equity = self.portfolio_equity_curve[-1] if self.portfolio_equity_curve else self.initial_portfolio_capital
        total_pnl = final_equity - self.initial_portfolio_capital
        total_return = (total_pnl / self.initial_portfolio_capital) * 100 if self.initial_portfolio_capital > 0 else 0

        # Note: A proper portfolio Sharpe/Sortino would require calculating portfolio returns,
        # which is more complex. We'll use a simplified version for now.
        
        return {
            "initial_capital": self.initial_portfolio_capital,
            "final_capital": final_equity,
            "pnl_cash": total_pnl,
            "total_return_pct": total_return,
            "max_drawdown": max_drawdown,
        }

def run_single_backtest(df_1m_full: pd.DataFrame, config_module, verbose: bool = False, logger=None):
    """
    A wrapper to run a backtest for a single strategy using the SimulationEngine.
    """
    bot_config = config_module
    # Use a default fee if not present, but it should be in the config module
    fee = getattr(bot_config, 'BYBIT_TAKER_FEE', 0.00055)
    dummy_connector = DummyConnector(SimulationClock(0), { "BYBIT_TAKER_FEE": fee })

    bot = AdvancedAdaptiveGridTradingBot(
        initial_capital=bot_config.INITIAL_CAPITAL,
        simulation_clock=SimulationClock(0), # Engine will manage the clock
        config_module=bot_config,
        connector=dummy_connector,
        silent=not verbose
    )
    
    engine = SimulationEngine(df_1m_full, [bot], logger)
    portfolio_results, individual_results = engine.run()
    
    # For a single run, individual_results will have one item.
    final_metrics = individual_results[0] if individual_results else {}
    
    if verbose and logger and final_metrics:
        logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
    # Also return portfolio metrics, which contain the correct final capital
    elif not final_metrics and portfolio_results:
        # If bot failed (e.g., no trades), return portfolio stats
        return portfolio_results

    return final_metrics