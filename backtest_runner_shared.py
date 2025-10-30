# backtest_runner_shared.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Optional
# --- *** NEW: Importing shared bot *** ---
from trading_bot_shared import AdvancedAdaptiveGridTradingBot
# --- *** END NEW *** ---
from core_types import SimulationClock, Tick, Position
import optuna 

# --- NEW: Importing the standardized DummyConnector ---
from connectors.dummy_connector import DummyConnector


class SimulationEngine:
    """
    Handles the time-synchronized, concurrent simulation of one or more trading bots
    using a SHARED CAPITAL POOL (Model C).
    """
    def __init__(self, data: pd.DataFrame, bots: List[AdvancedAdaptiveGridTradingBot], logger=None, prepared: bool = False):
        self.data = data 
        self.is_prepared = prepared
        self.bots = bots
        self.logger = logger or logging.getLogger(__name__)
        
        # --- *** NEW: Master Portfolio State *** ---
        # Get initial capital from the *first bot* and multiply by total bots
        # This assumes all bots are intended to start with the same capital
        if not self.bots:
            raise ValueError("SimulationEngine requires at least one bot.")
        
        self.initial_per_bot_capital = self.bots[0].initial_capital
        self.initial_portfolio_capital = self.initial_per_bot_capital * len(self.bots)
        self.master_peak_capital = self.initial_portfolio_capital
        
        # This tracks the *portfolio's* equity curve, not individual bots
        self.portfolio_equity_curve = [self.initial_portfolio_capital]
        # --- *** END NEW *** ---

    def _prepare_data(self):
        # (This function is identical to the one in backtest_runner.py)
        # (No changes needed)
        
        if self.is_prepared:
            return self.data 
            
        df_1m_full = self.data.reset_index(drop=True)

        bot_config = self.bots[0].config
        
        df_30m = df_1m_full.set_index('timestamp').resample('30min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).reset_index().dropna()
        df_30m['short_ema'] = ta.ema(df_30m['close'], length=bot_config.SHORT_EMA_PERIOD)
        df_30m['long_ema'] = ta.ema(df_30m['close'], length=bot_config.LONG_EMA_PERIOD)
        df_30m['rsi'] = ta.rsi(df_30m['close'], length=bot_config.RSI_PERIOD)
        df_30m['atr'] = ta.atr(df_30m['high'], df_30m['low'], df_30m['close'], length=bot_config.ATR_PERIOD)
        macd_30m = ta.macd(df_30m['close'], fast=bot_config.MACD_FAST_PERIOD, slow=bot_config.MACD_SLOW_PERIOD, signal=bot_config.MACD_SIGNAL_PERIOD)
        df_30m['macd'] = macd_30m[f'MACD_{bot_config.MACD_FAST_PERIOD}_{bot_config.MACD_SLOW_PERIOD}_{bot_config.MACD_SIGNAL_PERIOD}']

        bb_period = bot_config.BOLLINGER_PERIOD
        bb_std = bot_config.BOLLINGER_STD_DEV
        bbands_30m = ta.bbands(df_30m['close'], length=bb_period, std=bb_std)
        bbl_col_name = None
        bbu_col_name = None
        patterns_to_try = [
            f'BBL_{bb_period}_{bb_std}', f'BBL_{bb_period}_{int(bb_std)}', f'BBL_{bb_period}'
        ]
        for pattern in patterns_to_try:
            if pattern in bbands_30m.columns:
                bbl_col_name = pattern
                bbu_col_name = pattern.replace('BBL', 'BBU') 
                break
        if bbl_col_name is None:
            try:
                bbl_col_name = [col for col in bbands_30m.columns if col.startswith('BBL_')][0]
                bbu_suffix = bbl_col_name[4:] 
                bbu_col_name = f'BBU{bbu_suffix}'
                if bbu_col_name not in bbands_30m.columns:
                    bbu_col_name = [col for col in bbands_30m.columns if col.startswith('BBU_')][0]
            except IndexError:
                raise KeyError(f"Could not find Bollinger Bands columns in {bbands_30m.columns}.")

        df_30m['bb_lower'] = bbands_30m[bbl_col_name]
        df_30m['bb_upper'] = bbands_30m[bbu_col_name]

        df_1m = df_1m_full.copy()
        df_1m['rsi_1m'] = ta.rsi(df_1m['close'], length=bot_config.CONFIRMATION_RSI_PERIOD).shift(1)
        df_1m['volume_ma_1m'] = df_1m['volume'].rolling(window=bot_config.CONFIRMATION_VOLUME_MA_PERIOD).mean().shift(1)
        macd_1m = ta.macd(df_1m['close'], fast=bot_config.CONFIRMATION_MACD_FAST_PERIOD, slow=bot_config.CONFIRMATION_MACD_SLOW_PERIOD, signal=bot_config.CONFIRMATION_MACD_SIGNAL_PERIOD)
        df_1m['macd_1m'] = macd_1m[f'MACD_{bot_config.CONFIRMATION_MACD_FAST_PERIOD}_{bot_config.CONFIRMATION_MACD_SLOW_PERIOD}_{bot_config.CONFIRMATION_MACD_SIGNAL_PERIOD}'].shift(1)
        
        indicator_cols = ['timestamp', 'short_ema', 'long_ema', 'rsi', 'atr', 'macd', 'bb_lower', 'bb_upper']
        for col in indicator_cols:
            if col != 'timestamp':
                df_30m[col] = df_30m[col].shift(1)
                
        df_merged = pd.merge_asof(df_1m.sort_values('timestamp'), df_30m[indicator_cols].sort_values('timestamp'), on='timestamp', direction='backward')
        
        return df_merged.dropna()

    def run(self, trial: Optional[optuna.trial.Trial] = None):
        
        try:
            df_merged = self._prepare_data()
        except KeyError as e:
            self.logger.error(f"Failed to prepare data due to missing column: {e}")
            return None, []
        
        if len(df_merged) < 100:
            self.logger.warning("Backtest failed: Insufficient data after indicator calculation.")
            return None, []

        sim_start_time = df_merged['timestamp'].iloc[0].timestamp()
        clock = SimulationClock(start_time=sim_start_time)
        dummy_connector = DummyConnector(clock, {})

        first_row = df_merged.iloc[0]
        
        # --- *** NEW: Bot Initialization for Model C *** ---
        # The allocation is set *before* the simulation,
        # typically by the script that calls this (e.g., walk_forward..._shared.py)
        for bot in self.bots:
            # Set each bot's *initial* capital to its per-bot slice.
            # The connector will add/subtract from this.
            bot.initial_capital = self.initial_per_bot_capital
            bot.capital = self.initial_per_bot_capital
            bot.peak_capital = self.initial_per_bot_capital # Start with bot's own peak
            
            bot.clock = clock
            bot.connector = dummy_connector
            bot.initialize_state(first_row.to_dict())
            
            fee = getattr(bot.config, 'BYBIT_TAKER_FEE', 0.00055)
            bot.connector.params['BYBIT_TAKER_FEE'] = fee
        
        # --- *** END NEW *** ---


        # --- HIGH-SPEED CONCURRENT SIMULATION LOOP ---
        for i in range(1, len(df_merged)):
            current_row, prev_row = df_merged.iloc[i], df_merged.iloc[i-1]
            
            # Update 30-minute strategies
            if current_row['timestamp'].floor('30min') > prev_row['timestamp'].floor('30min'):
                for bot in self.bots:
                    bot.update_strategy_on_30m(prev_row.to_dict())
            
            # Simulate ticks for the current 1m candle
            o, h, l, c, v, base_ts = current_row['open'], current_row['high'], current_row['low'], current_row['close'], current_row['volume'], current_row['timestamp'].timestamp()
            
            if c > o: key_prices = [o, l, h, c]
            else: key_prices = [o, h, l, c]
            
            rsi_1m_val = current_row['rsi_1m']
            macd_1m_val = current_row['macd_1m']
            volume_ma_1m_val = current_row['volume_ma_1m']
            atr_val = current_row['atr']
            volume_1m_val = prev_row['volume'] 

            for price in key_prices:
                tick = Tick(timestamp=base_ts, price=price, volume=0, candle_volume=v) 
                clock.current_time, clock.current_price = tick.timestamp, price
                
                for bot in self.bots:
                    bot.check_entries_on_tick(tick, rsi_1m_val, macd_1m_val, volume_1m_val, volume_ma_1m_val, atr_val)
                    bot.check_exits_on_1m(price, atr_val)

            # --- *** NEW: MASTER PORTFOLIO SYNC (Model C) *** ---
            # At the end of every 1-minute candle, sync the portfolio state.
            
            # 1. Calculate GLOBAL portfolio state
            total_realized_capital = sum(bot.capital for bot in self.bots)
            
            total_unrealized_pnl = 0.0
            for bot in self.bots:
                for pos in bot.open_positions:
                    total_unrealized_pnl += pos.calculate_profit(clock.current_price)
            
            current_total_equity = total_realized_capital + total_unrealized_pnl
            
            # 2. Update GLOBAL peak capital
            self.master_peak_capital = max(self.master_peak_capital, current_total_equity)

            # 3. Inject GLOBAL state into ALL bots
            # This forces them to use the "live" logic path for sizing and risk
            for bot in self.bots:
                bot.capital = total_realized_capital        # Set to GLOBAL realized cap
                bot.unrealized_pnl = total_unrealized_pnl   # Set to GLOBAL UPL
                bot.peak_capital = self.master_peak_capital # Set to GLOBAL peak
            # --- *** END NEW *** ---

            # Record portfolio equity at the end of each 1m candle
            self.portfolio_equity_curve.append(current_total_equity)
            
            # --- Pruning Check ---
            if trial and i % 1000 == 0:
                trial.report(current_total_equity, i) # Report total equity
                if trial.should_prune():
                    raise optuna.TrialPruned()


        # --- Final Performance Calculation ---
        # This now correctly reflects the *total portfolio* performance
        portfolio_results = self._calculate_portfolio_metrics()
        
        if trial:
            trial.report(portfolio_results['final_capital'], len(df_merged))
        
        # Note: The *individual* logs will be slightly weird, because
        # 'self.capital' in the log_performance string will be the
        # *total portfolio capital*. This is expected.
        # The key metrics (Win Rate, PF) are per-bot.
        individual_results = [bot.log_performance(print_log=False) for bot in self.bots]
        
        return portfolio_results, individual_results

    def _calculate_portfolio_metrics(self):
        # This logic is now correct, as it uses the total portfolio equity
        equity_series = pd.Series(self.portfolio_equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100
        
        final_equity = self.portfolio_equity_curve[-1] if self.portfolio_equity_curve else self.initial_portfolio_capital
        total_pnl = final_equity - self.initial_portfolio_capital
        total_return = (total_pnl / self.initial_portfolio_capital) * 100 if self.initial_portfolio_capital > 0 else 0
        
        return {
            "initial_capital": self.initial_portfolio_capital,
            "final_capital": final_equity,
            "pnl_cash": total_pnl,
            "total_return_pct": total_return,
            "max_drawdown": max_drawdown,
        }

# --- Wrapper functions (prepare_data, run_from_prepared, run_single) ---
# These wrappers are modified to import from the new _shared files

def prepare_data_for_simulation(df_1m_full: pd.DataFrame, config_module) -> pd.DataFrame:
    """
    Runs only the data preparation step and returns the prepared DataFrame.
    """
    dummy_bot = AdvancedAdaptiveGridTradingBot( # from trading_bot_shared
        initial_capital=0,
        simulation_clock=SimulationClock(0),
        config_module=config_module,
        connector=None,
        silent=True
    )
    
    temp_engine = SimulationEngine(df_1m_full, [dummy_bot], prepared=False)
    
    try:
        df_prepared = temp_engine._prepare_data()
        return df_prepared
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed during data preparation: {e}", exc_info=True)
        return pd.DataFrame() 

def run_simulation_from_prepared_data(
    df_prepared: pd.DataFrame, 
    config_module, 
    verbose: bool = False, 
    logger=None, 
    trial: Optional[optuna.trial.Trial] = None
):
    """
    A wrapper to run a *Model A* (single bot) backtest using PREPARED data.
    This is used by the *optimizer* which only tests one bot at a time.
    """
    bot_config = config_module
    fee = getattr(bot_config, 'BYBIT_TAKER_FEE', 0.00055)
    
    dummy_connector = DummyConnector(SimulationClock(0), { "BYBIT_TAKER_FEE": fee })

    bot = AdvancedAdaptiveGridTradingBot( # from trading_bot_shared
        initial_capital=bot_config.INITIAL_CAPITAL,
        simulation_clock=SimulationClock(0), 
        config_module=bot_config,
        connector=dummy_connector,
        silent=not verbose
    )
    # --- *** NOTE: This bot will run with self.capital_allocation = 1.0 *** ---
    # --- This is correct for the optimization step (1 bot) ---
    
    engine = SimulationEngine(df_prepared, [bot], logger, prepared=True)
    
    portfolio_results, individual_results = engine.run(trial=trial)
    
    final_metrics = individual_results[0] if individual_results else {}
    
    if verbose and logger and final_metrics:
        logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
    elif not final_metrics and portfolio_results:
        return portfolio_results

    return final_metrics


def run_single_backtest(
    df_1m_full: pd.DataFrame, 
    config_module, 
    verbose: bool = False, 
    logger=None, 
    trial: Optional[optuna.trial.Trial] = None
):
    """
    A wrapper to run a full (slower) *Model A* (single bot) backtest.
    """
    bot_config = config_module
    fee = getattr(bot_config, 'BYBIT_TAKER_FEE', 0.00055)
    dummy_connector = DummyConnector(SimulationClock(0), { "BYBIT_TAKER_FEE": fee })

    bot = AdvancedAdaptiveGridTradingBot( # from trading_bot_shared
        initial_capital=bot_config.INITIAL_CAPITAL,
        simulation_clock=SimulationClock(0), 
        config_module=bot_config,
        connector=dummy_connector,
        silent=not verbose
    )
    
    engine = SimulationEngine(df_1m_full, [bot], logger, prepared=False)
    
    portfolio_results, individual_results = engine.run(trial=trial)
    
    final_metrics = individual_results[0] if individual_results else {}
    
    if verbose and logger and final_metrics:
        logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
    elif not final_metrics and portfolio_results:
        return portfolio_results

    return final_metrics