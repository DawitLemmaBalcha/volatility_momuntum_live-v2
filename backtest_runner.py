# backtest_runner.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
# import random <-- 1. REMOVED (Original file)
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Optional
from trading_bot import AdvancedAdaptiveGridTradingBot
from core_types import SimulationClock, Tick, Position
import optuna # <-- NEW: Import Optuna for pruning

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
    # --- MODIFIED: Accepts prepared data and a flag ---
    def __init__(self, data: pd.DataFrame, bots: List[AdvancedAdaptiveGridTradingBot], logger=None, prepared: bool = False):
        self.data = data # This is either raw 1m data OR prepared merged data
        self.is_prepared = prepared
        # --- END MODIFIED ---
        
        self.bots = bots
        self.logger = logger or logging.getLogger(__name__)
        self.portfolio_equity_curve = []
        self.initial_portfolio_capital = sum(bot.initial_capital for bot in self.bots)

    def _prepare_data(self):
        # --- MODIFIED: Use self.data instead of self.df_1m_full ---
        if self.is_prepared:
            # Data is already prepared, just return it
            return self.data 
            
        df_1m_full = self.data.reset_index(drop=True)
        # --- END MODIFIED ---

        # Data prep is done once for all bots, assuming they use the same indicator periods for now.
        # A more advanced version could handle different indicator sets per bot.
        bot_config = self.bots[0].config
        
        df_30m = df_1m_full.set_index('timestamp').resample('30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).reset_index().dropna()
        df_30m['short_ema'] = ta.ema(df_30m['close'], length=bot_config.SHORT_EMA_PERIOD)
        df_30m['long_ema'] = ta.ema(df_30m['close'], length=bot_config.LONG_EMA_PERIOD)
        df_30m['rsi'] = ta.rsi(df_30m['close'], length=bot_config.RSI_PERIOD)
        df_30m['atr'] = ta.atr(df_30m['high'], df_30m['low'], df_30m['close'], length=bot_config.ATR_PERIOD)
        macd_30m = ta.macd(df_30m['close'], fast=bot_config.MACD_FAST_PERIOD, slow=bot_config.MACD_SLOW_PERIOD, signal=bot_config.MACD_SIGNAL_PERIOD)
        df_30m['macd'] = macd_30m[f'MACD_{bot_config.MACD_FAST_PERIOD}_{bot_config.MACD_SLOW_PERIOD}_{bot_config.MACD_SIGNAL_PERIOD}']

        # --- ROBUST FIX: Handle pandas_ta bbands column naming conventions ---
        bb_period = bot_config.BOLLINGER_PERIOD
        bb_std = bot_config.BOLLINGER_STD_DEV
        
        bbands_30m = ta.bbands(df_30m['close'], length=bb_period, std=bb_std)

        # pandas-ta has inconsistent naming. We must find the column.
        bbl_col_name = None
        bbu_col_name = None

        # 1. Try common patterns first
        patterns_to_try = [
            f'BBL_{bb_period}_{bb_std}',      # e.g., 'BBL_20_2.0'
            f'BBL_{bb_period}_{int(bb_std)}',  # e.g., 'BBL_20_2'
            f'BBL_{bb_period}'              # e.g., 'BBL_20' (if std is default)
        ]
        
        for pattern in patterns_to_try:
            if pattern in bbands_30m.columns:
                bbl_col_name = pattern
                bbu_col_name = pattern.replace('BBL', 'BBU') # Assumes BBU follows the same pattern
                break
        
        # 2. If no pattern matched, find the first column that starts with 'BBL_'
        if bbl_col_name is None:
            try:
                bbl_col_name = [col for col in bbands_30m.columns if col.startswith('BBL_')][0]
                # Find corresponding BBU
                bbu_suffix = bbl_col_name[4:] # Get the suffix (e.g., '_20_2.0')
                bbu_col_name = f'BBU{bbu_suffix}'
                if bbu_col_name not in bbands_30m.columns:
                    # Fallback for BBU
                    bbu_col_name = [col for col in bbands_30m.columns if col.startswith('BBU_')][0]
            except IndexError:
                # If we still can't find it, raise a helpful error
                raise KeyError(f"Could not find Bollinger Bands columns in {bbands_30m.columns}. Looked for patterns like 'BBL_20_2.0', 'BBL_20_2', 'BBL_20', etc.")

        df_30m['bb_lower'] = bbands_30m[bbl_col_name]
        df_30m['bb_upper'] = bbands_30m[bbu_col_name]
        # --- END ROBUST FIX ---

        df_1m = df_1m_full.copy()
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

    # --- MODIFIED: Added trial=None for pruning ---
    def run(self, trial: Optional[optuna.trial.Trial] = None):
        
        # --- MODIFIED: Call _prepare_data based on self.is_prepared flag ---
        try:
            df_merged = self._prepare_data()
        except KeyError as e:
            self.logger.error(f"Failed to prepare data due to missing column: {e}")
            return None, []
        # --- END MODIFIED ---
        
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
            
            # --- NEW: Set fee from bot's config for accuracy ---
            fee = getattr(bot.config, 'BYBIT_TAKER_FEE', 0.00055)
            bot.connector.params['BYBIT_TAKER_FEE'] = fee
            # --- END NEW ---


        # --- HIGH-SPEED CONCURRENT SIMULATION LOOP ---
        # random.seed(42) # <-- 2. REMOVED (Original file)
        for i in range(1, len(df_merged)):
            current_row, prev_row = df_merged.iloc[i], df_merged.iloc[i-1]
            
            # Update 30-minute strategies
            if current_row['timestamp'].floor('30min') > prev_row['timestamp'].floor('30min'):
                for bot in self.bots:
                    # <-- 3. FIX: Use PREVIOUS row for 30m strategy decisions (Original file)
                    bot.update_strategy_on_30m(prev_row.to_dict())
            
            # Simulate ticks for the current 1m candle
            o, h, l, c, v, base_ts = current_row['open'], current_row['high'], current_row['low'], current_row['close'], current_row['volume'], current_row['timestamp'].timestamp()
            
            # --- START: Deterministic Worst-Case Path Simulation ---
            # (Original file logic)
            if c > o:
                # Bullish Bar: Test stops first (O -> L -> H -> C)
                key_prices = [o, l, h, c]
            else:
                # Bearish Bar: Test entries first (O -> H -> L -> C)
                key_prices = [o, h, l, c]
            # --- END: Deterministic Worst-Case Path Simulation ---
            
            
            # <-- 5. FIX: Get all indicator values from the PREVIOUS, CLOSED candle
            # These values are constant for all 4 ticks within the current candle.
            
            # --- START OF FIX ---
            # Read from 'current_row' because 'shift(1)' in _prepare_data
            # already made these values point-in-time correct (i.e., from t-1).
            rsi_1m_val = current_row['rsi_1m']
            macd_1m_val = current_row['macd_1m']
            volume_ma_1m_val = current_row['volume_ma_1m']
            atr_val = current_row['atr'] # 30m ATR is already from the past via merge_asof

            # This one remains 'prev_row' as it's the volume of the *last completed bar*.
            volume_1m_val = prev_row['volume'] 
            # --- END OF FIX ---


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
            
            # --- NEW: ADD PRUNING CHECK ---
            # Check every 1000 candles (approx every 16 hours of trading)
            if trial and i % 1000 == 0:
                # Report the intermediate portfolio equity to Optuna
                trial.report(current_portfolio_equity, i)

                # Check if Optuna thinks this trial should be pruned
                if trial.should_prune():
                    # This exception will be caught by the objective function
                    raise optuna.TrialPruned()
            # --- END NEW ---


        # --- Final Performance Calculation ---
        portfolio_results = self._calculate_portfolio_metrics()
        
        # --- NEW: Report final value if pruning ---
        if trial:
            trial.report(portfolio_results['final_capital'], len(df_merged))
        # --- END NEW ---
        
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

# --- NEW: Wrapper function to prepare data once ---
def prepare_data_for_simulation(df_1m_full: pd.DataFrame, config_module) -> pd.DataFrame:
    """
    Runs only the data preparation step and returns the prepared DataFrame.
    This allows for a "prepare-once, run-many" optimization loop.
    """
    # Create a dummy bot just to get the config for indicators
    dummy_bot = AdvancedAdaptiveGridTradingBot(
        initial_capital=0,
        simulation_clock=SimulationClock(0),
        config_module=config_module,
        connector=None,
        silent=True
    )
    
    # Use a temporary SimulationEngine to just run the _prepare_data method
    temp_engine = SimulationEngine(df_1m_full, [dummy_bot], prepared=False)
    
    try:
        df_prepared = temp_engine._prepare_data()
        return df_prepared
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed during data preparation: {e}", exc_info=True)
        return pd.DataFrame() # Return empty dataframe on failure

# --- NEW: Wrapper function to run simulation on PREPARED data ---
def run_simulation_from_prepared_data(
    df_prepared: pd.DataFrame, 
    config_module, 
    verbose: bool = False, 
    logger=None, 
    trial: Optional[optuna.trial.Trial] = None
):
    """
    A wrapper to run a backtest for a single strategy using PREPARED data.
    This is the fast function called by Optuna.
    """
    bot_config = config_module
    fee = getattr(bot_config, 'BYBIT_TAKER_FEE', 0.00055)
    
    # Connector is re-created for each bot, but this is lightweight
    dummy_connector = DummyConnector(SimulationClock(0), { "BYBIT_TAKER_FEE": fee })

    bot = AdvancedAdaptiveGridTradingBot(
        initial_capital=bot_config.INITIAL_CAPITAL,
        simulation_clock=SimulationClock(0), # Engine will manage the clock
        config_module=bot_config,
        connector=dummy_connector,
        silent=not verbose
    )
    
    # --- MODIFIED: Pass prepared=True ---
    engine = SimulationEngine(df_prepared, [bot], logger, prepared=True)
    
    # --- MODIFIED: Pass the trial object to the engine's run method ---
    portfolio_results, individual_results = engine.run(trial=trial)
    # --- END MODIFIED ---
    
    final_metrics = individual_results[0] if individual_results else {}
    
    if verbose and logger and final_metrics:
        logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
    elif not final_metrics and portfolio_results:
        return portfolio_results

    return final_metrics


# --- MODIFIED: `run_single_backtest` now supports pruning ---
def run_single_backtest(
    df_1m_full: pd.DataFrame, 
    config_module, 
    verbose: bool = False, 
    logger=None, 
    trial: Optional[optuna.trial.Trial] = None # <-- ADDED trial
):
    """
    A wrapper to run a full (slower) backtest, preparing data first.
    """
    bot_config = config_module
    fee = getattr(bot_config, 'BYBIT_TAKER_FEE', 0.00055)
    dummy_connector = DummyConnector(SimulationClock(0), { "BYBIT_TAKER_FEE": fee })

    bot = AdvancedAdaptiveGridTradingBot(
        initial_capital=bot_config.INITIAL_CAPITAL,
        simulation_clock=SimulationClock(0), # Engine will manage the clock
        config_module=bot_config,
        connector=dummy_connector,
        silent=not verbose
    )
    
    # --- MODIFIED: Pass prepared=False ---
    engine = SimulationEngine(df_1m_full, [bot], logger, prepared=False)
    
    # --- MODIFIED: Pass trial to engine.run ---
    portfolio_results, individual_results = engine.run(trial=trial)
    
    # For a single run, individual_results will have one item.
    final_metrics = individual_results[0] if individual_results else {}
    
    if verbose and logger and final_metrics:
        logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
    # Also return portfolio metrics, which contain the correct final capital
    elif not final_metrics and portfolio_results:
        # If bot failed (e.g., no trades), return portfolio stats
        return portfolio_results

    return final_metrics