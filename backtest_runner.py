# backtest_runner.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List
from trading_bot import AdvancedAdaptiveGridTradingBot
from core_types import SimulationClock, Tick, Position

# NEW: Imports for Numba kernel
import numba as nb
from simulation_kernels import run_numba_simulation

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

def prepare_data_for_simulation(df_1m_full: pd.DataFrame, bot_config) -> pd.DataFrame:
    """
    Pre-calculates all indicators for a simulation run.
    This is extracted to be run only ONCE per optimization walk.
    """
    
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
    
    # Merge 30m indicators into 1m data (asof for point-in-time)
    df_merged = pd.merge_asof(df_1m.sort_values('timestamp'), df_30m.sort_values('timestamp'), on='timestamp', direction='backward')
    df_merged = df_merged.dropna(subset=['short_ema'])  # Drop early rows without 30m data

    return df_merged

class SimulationEngine:
    def __init__(self, df_prepared: pd.DataFrame, bots: List[AdvancedAdaptiveGridTradingBot], logger=None):
        self.df_merged = df_prepared
        self.bots = bots
        self.logger = logger or logging.getLogger(__name__)
        self.initial_portfolio_capital = sum(bot.initial_capital for bot in bots)
        self.portfolio_equity_curve = []

    def run(self):
        # NEW: Gutted loop; now extract arrays and params, call Numba kernel
        # Assume one bot for simplicity (as in your code)
        config = self.bots[0].config
        
        # Extract NumPy arrays (timestamps as float unix seconds)
        timestamps = self.df_merged['timestamp'].dt.timestamp().to_numpy(dtype=np.float64)
        opens = self.df_merged['open'].to_numpy(dtype=np.float64)
        highs = self.df_merged['high'].to_numpy(dtype=np.float64)
        lows = self.df_merged['low'].to_numpy(dtype=np.float64)
        closes = self.df_merged['close'].to_numpy(dtype=np.float64)
        volumes = self.df_merged['volume'].to_numpy(dtype=np.float64)
        short_emas = self.df_merged['short_ema'].to_numpy(dtype=np.float64)
        long_emas = self.df_merged['long_ema'].to_numpy(dtype=np.float64)
        rsis = self.df_merged['rsi'].to_numpy(dtype=np.float64)
        atrs = self.df_merged['atr'].to_numpy(dtype=np.float64)
        macds = self.df_merged['macd'].to_numpy(dtype=np.float64)
        bb_lowers = self.df_merged['bb_lower'].to_numpy(dtype=np.float64)
        bb_uppers = self.df_merged['bb_upper'].to_numpy(dtype=np.float64)
        rsi_1ms = self.df_merged['rsi_1m'].to_numpy(dtype=np.float64)
        volume_ma_1ms = self.df_merged['volume_ma_1m'].to_numpy(dtype=np.float64)
        macd_1ms = self.df_merged['macd_1m'].to_numpy(dtype=np.float64)
        
        # Extract params as floats/ints
        initial_capital = float(config.INITIAL_CAPITAL)
        max_positions = int(config.MAX_POSITIONS)
        bybit_taker_fee = float(config.BYBIT_TAKER_FEE)
        grid_setup_cooldown_seconds = float(config.GRID_SETUP_COOLDOWN_SECONDS)
        atr_initial_stop_multiplier = float(config.ATR_INITIAL_STOP_MULTIPLIER)
        atr_trailing_stop_activation_multiplier = float(config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER)
        trailing_ratio = float(config.TRAILING_RATIO)
        grid_bb_width_multiplier = float(config.GRID_BB_WIDTH_MULTIPLIER)
        confirmation_rsi_period = int(config.CONFIRMATION_RSI_PERIOD)
        confirmation_volume_ma_period = int(config.CONFIRMATION_VOLUME_MA_PERIOD)
        atr_trailing_stop_multiplier = float(config.ATR_TRAILING_STOP_MULTIPLIER)
        position_degrading_factor = float(config.POSITION_DEGRADING_FACTOR)
        num_grids = int(config.NUM_GRIDS)
        grid_max_lifespan_seconds = float(config.GRID_MAX_LIFESPAN_SECONDS)
        bollinger_period = int(config.BOLLINGER_PERIOD)
        bollinger_std_dev = float(config.BOLLINGER_STD_DEV)
        rsi_buy_confirmation = float(config.RSI_BUY_CONFIRMATION)
        rsi_sell_confirmation = float(config.RSI_SELL_CONFIRMATION)
        volume_confirmation_factor = float(config.VOLUME_CONFIRMATION_FACTOR)
        trend_grid_ratio_normal = float(config.TREND_GRID_RATIO_NORMAL)
        trend_grid_ratio_strong = float(config.TREND_GRID_RATIO_STRONG)
        confirmation_macd_fast_period = int(config.CONFIRMATION_MACD_FAST_PERIOD)
        confirmation_macd_slow_period = int(config.CONFIRMATION_MACD_SLOW_PERIOD)
        confirmation_macd_signal_period = int(config.CONFIRMATION_MACD_SIGNAL_PERIOD)
        max_position_duration_seconds = float(config.MAX_POSITION_DURATION_SECONDS)
        grid_volatility_scaling_min = float(config.GRID_VOLATILITY_SCALING_MIN)
        grid_volatility_scaling_max = float(config.GRID_VOLATILITY_SCALING_MAX)
        grid_min_size_percent = float(config.GRID_MIN_SIZE_PERCENT)
        grid_max_size_percent = float(config.GRID_MAX_SIZE_PERCENT)
        take_profit_percent = float(config.TAKE_PROFIT_PERCENT)
        max_drawdown_percent = float(config.MAX_DRAWDOWN_PERCENT)
        atr_period = int(config.ATR_PERIOD)
        max_position_size_percent = float(config.MAX_POSITION_SIZE_PERCENT)
        kelly_fraction = float(config.KELLY_FRACTION)
        kelly_lookback = int(config.KELLY_LOOKBACK)
        kelly_min_fraction = float(config.KELLY_MIN_FRACTION)
        rsi_period = int(config.RSI_PERIOD)
        short_ema_period = int(config.SHORT_EMA_PERIOD)
        long_ema_period = int(config.LONG_EMA_PERIOD)
        macd_fast_period = int(config.MACD_FAST_PERIOD)
        macd_slow_period = int(config.MACD_SLOW_PERIOD)
        macd_signal_period = int(config.MACD_SIGNAL_PERIOD)
        volume_ma_period = int(config.VOLUME_MA_PERIOD)
        trend_ema_threshold_percent = float(config.TREND_EMA_THRESHOLD_PERCENT)
        bb_regime_threshold = float(config.BB_REGIME_THRESHOLD)
        rsi_oversold = float(config.RSI_OVERSOLD)
        rsi_overbought = float(config.RSI_OVERBOUGHT)
        
        # Call Numba kernel
        final_capital, pnl_cash, total_return_pct, max_drawdown, equity_curve, trade_history_trimmed, total_grids_built, total_grids_traded = run_numba_simulation(
            timestamps, opens, highs, lows, closes, volumes, short_emas, long_emas, rsis, atrs, macds, bb_lowers, bb_uppers, rsi_1ms, volume_ma_1ms, macd_1ms,
            initial_capital, max_positions, bybit_taker_fee, grid_setup_cooldown_seconds, atr_initial_stop_multiplier, atr_trailing_stop_activation_multiplier, trailing_ratio, grid_bb_width_multiplier, confirmation_rsi_period, confirmation_volume_ma_period, atr_trailing_stop_multiplier, position_degrading_factor, num_grids, grid_max_lifespan_seconds, bollinger_period, bollinger_std_dev, rsi_buy_confirmation, rsi_sell_confirmation, volume_confirmation_factor, trend_grid_ratio_normal, trend_grid_ratio_strong, confirmation_macd_fast_period, confirmation_macd_slow_period, confirmation_macd_signal_period, max_position_duration_seconds, grid_volatility_scaling_min, grid_volatility_scaling_max, grid_min_size_percent, grid_max_size_percent, take_profit_percent, max_drawdown_percent, atr_period, max_position_size_percent, kelly_fraction, kelly_lookback, kelly_min_fraction, rsi_period, short_ema_period, long_ema_period, macd_fast_period, macd_slow_period, macd_signal_period, volume_ma_period, trend_ema_threshold_percent, bb_regime_threshold, rsi_oversold, rsi_overbought
        )
        
        # Format results (reconstruct metrics dict; add regime_performance, etc., from trade_history_trimmed)
        portfolio_results = {
            "initial_capital": self.initial_portfolio_capital,
            "final_capital": final_capital,
            "pnl_cash": pnl_cash,
            "total_return_pct": total_return_pct,
            "max_drawdown": max_drawdown,
        }
        
        # Individual results (mock bot.log_performance from arrays)
        individual_results = [{
            "total_trades": len(trade_history_trimmed),
            "win_rate": np.sum(trade_history_trimmed[:, TRADE_PNL_CASH] > 0) / len(trade_history_trimmed) * 100 if len(trade_history_trimmed) > 0 else 0,
            # ... Add more: profit_factor, sharpe (from equity_curve), regime_performance (group by entry_regime), grid_hit_rate = total_grids_traded / total_grids_built * 100 if total_grids_built > 0 else 0
            "grid_hit_rate": total_grids_traded / total_grids_built * 100 if total_grids_built > 0 else 0,
            # For Sharpe/Sortino/Calmar: Compute from equity_curve as in original
            # equity_series = pd.Series(equity_curve)  # Convert back to pandas for calcs
            # ... (implement as needed)
        }]
        
        self.portfolio_equity_curve = equity_curve.tolist()  # For compatibility
        
        return portfolio_results, individual_results

    def _calculate_portfolio_metrics(self):
        # Deprecated; now in kernel
        pass

def run_simulation_from_prepared_data(df_prepared: pd.DataFrame, config_module, verbose: bool = False, logger=None):
    """
    Runs a single bot simulation using pre-calculated indicator data.
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
    
    # --- MODIFIED: Pass the prepared data to the engine ---
    engine = SimulationEngine(df_prepared, [bot], logger)
    portfolio_results, individual_results = engine.run()
    
    final_metrics = individual_results[0] if individual_results else {}
    
    if verbose and logger and final_metrics:
        logger.info(final_metrics.get("performance_log_str", "Performance log not available."))
    elif not final_metrics and portfolio_results:
        return portfolio_results

    return final_metrics


# --- MODIFIED: run_single_backtest is now a simple wrapper ---
def run_single_backtest(df_1m_full: pd.DataFrame, config_module, verbose: bool = False, logger=None):
    """
    A wrapper to PREPARE data and then run a single backtest.
    Used by 'main.py' or for one-off tests.
    """
    # 1. Prepare data
    df_prepared = prepare_data_for_simulation(df_1m_full, config_module)
    
    # 2. Run simulation
    return run_simulation_from_prepared_data(df_prepared, config_module, verbose, logger)