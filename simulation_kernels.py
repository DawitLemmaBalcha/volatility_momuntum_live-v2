# simulation_kernels.py

import numba as nb
import numpy as np

# Regime codes (int for Numba)
NEUTRAL_RANGING = 0
NEUTRAL_TRENDING = 1
UP_RANGING = 2
UP_TRENDING = 3
DOWN_RANGING = 4
DOWN_TRENDING = 5
STRONG_UP_RANGING = 6
STRONG_UP_TRENDING = 7
STRONG_DOWN_RANGING = 8
STRONG_DOWN_TRENDING = 9

# Reason codes for closes (int)
REASON_TRAILING_STOP = 0
REASON_MAX_DURATION = 1
REASON_DRAWDOWN = 2

# Direction codes (int)
DIR_LONG = 1
DIR_SHORT = 0

# Position array columns
POS_ENTRY_PRICE = 0
POS_AMOUNT = 1
POS_IS_LONG = 2
POS_STOP_LOSS = 3
POS_PEAK_PRICE = 4
POS_ENTRY_TIME = 5
POS_KELLY_RISK_USD = 6
POS_ATR_RISK_USD = 7
POS_IS_TRAILING_ACTIVE = 8
POS_ENTRY_REGIME = 9

# Trade history columns
TRADE_ENTRY_PRICE = 0
TRADE_CLOSE_PRICE = 1
TRADE_ENTRY_TIME = 2
TRADE_CLOSE_TIME = 3
TRADE_PNL_PERCENT = 4
TRADE_PNL_CASH = 5
TRADE_REASON = 6
TRADE_DIRECTION = 7
TRADE_SIZE = 8
TRADE_KELLY_RISK_USD = 9
TRADE_ATR_RISK_USD = 10
TRADE_ENTRY_REGIME = 11

@nb.jit(nopython=True)
def get_trend_regime(short_ema, long_ema, rsi, macd, bb_upper, bb_lower, current_price, trend_ema_threshold_pct, bb_regime_threshold, rsi_overbought, rsi_oversold):
    trend_code = 0  # neutral
    if short_ema > long_ema * (1 + trend_ema_threshold_pct / 100):
        trend_code = 1  # up
    elif short_ema < long_ema * (1 - trend_ema_threshold_pct / 100):
        trend_code = 2  # down
    
    if rsi > rsi_overbought:
        trend_code = 3  # strong_up
    elif rsi < rsi_oversold:
        trend_code = 4  # strong_down
    
    if macd > 0 and (trend_code == 1 or trend_code == 3):
        trend_code = 3  # strong_up
    elif macd < 0 and (trend_code == 2 or trend_code == 4):
        trend_code = 4  # strong_down
    
    bb_width = (bb_upper - bb_lower) / current_price if current_price != 0 else 0
    is_trending = 1 if bb_width > bb_regime_threshold else 0
    
    if trend_code == 0:
        return NEUTRAL_TRENDING if is_trending else NEUTRAL_RANGING
    elif trend_code == 1:
        return UP_TRENDING if is_trending else UP_RANGING
    elif trend_code == 2:
        return DOWN_TRENDING if is_trending else DOWN_RANGING
    elif trend_code == 3:
        return STRONG_UP_TRENDING if is_trending else STRONG_UP_RANGING
    elif trend_code == 4:
        return STRONG_DOWN_TRENDING if is_trending else STRONG_DOWN_RANGING
    return NEUTRAL_RANGING  # Fallback

@nb.jit(nopython=True)
def setup_asymmetric_grid(current_time, last_grid_setup_time, grid_setup_cooldown_seconds, current_price, bb_upper, bb_lower, trend_regime, grid_min_size_percent, grid_max_size_percent, grid_bb_width_multiplier, num_grids, trend_grid_ratio_normal, trend_grid_ratio_strong, buy_grids, sell_grids):
    if current_time - last_grid_setup_time < grid_setup_cooldown_seconds:
        return last_grid_setup_time, 0.0, buy_grids, sell_grids  # No change
    
    bb_width = (bb_upper - bb_lower) / current_price if current_price > 0 else 0
    grid_size = max(grid_min_size_percent, min(bb_width * grid_bb_width_multiplier, grid_max_size_percent))
    
    total_grids = num_grids
    if trend_regime in [NEUTRAL_RANGING, UP_RANGING, DOWN_RANGING, STRONG_UP_RANGING, STRONG_DOWN_RANGING]:
        total_grids = 2
    
    ratio = trend_grid_ratio_strong if trend_regime >= STRONG_UP_RANGING else trend_grid_ratio_normal
    
    if trend_regime in [UP_RANGING, UP_TRENDING, STRONG_UP_RANGING, STRONG_UP_TRENDING]:
        upper_grids = int(round(total_grids * ratio))
        lower_grids = total_grids - upper_grids
    elif trend_regime in [DOWN_RANGING, DOWN_TRENDING, STRONG_DOWN_RANGING, STRONG_DOWN_TRENDING]:
        lower_grids = int(round(total_grids * ratio))
        upper_grids = total_grids - lower_grids
    else:
        upper_grids = total_grids // 2
        lower_grids = total_grids - upper_grids
    
    # Reset arrays
    buy_grids.fill(0.0)
    sell_grids.fill(0.0)
    
    for i in range(1, upper_grids + 1):
        if i-1 < len(buy_grids):
            buy_grids[i-1] = current_price * (1 + i * grid_size)
    
    for i in range(1, lower_grids + 1):
        if i-1 < len(sell_grids):
            sell_grids[i-1] = current_price * (1 - i * grid_size)
    
    return current_time, grid_size, buy_grids, sell_grids

@nb.jit(nopython=True)
def calculate_position_size(capital, max_position_size_percent, kelly_fraction, position_degrading_factor, trades_since_grid_rebuild, kelly_lookback, kelly_min_fraction, recent_pnls, recent_pnl_count):
    if recent_pnl_count == 0:
        kelly = kelly_fraction
    else:
        positive_pnls = recent_pnls[:recent_pnl_count] > 0
        wins = np.sum(positive_pnls)
        win_rate = wins / recent_pnl_count
        avg_win = np.mean(recent_pnls[:recent_pnl_count][positive_pnls]) if wins > 0 else 1.0
        avg_loss = np.abs(np.mean(recent_pnls[:recent_pnl_count][~positive_pnls])) if (recent_pnl_count - wins) > 0 else 1.0
        kelly = (win_rate / avg_loss) - ((1 - win_rate) / avg_win) if avg_loss != 0 and avg_win != 0 else kelly_fraction
        kelly = max(kelly_min_fraction, min(kelly_fraction, kelly))
    
    degrade = position_degrading_factor ** trades_since_grid_rebuild
    size_pct = min(max_position_size_percent, kelly * degrade)
    return capital * size_pct

@nb.jit(nopython=True)
def check_entries_on_tick(tick_price, rsi_1m, macd_1m, volume_1m, volume_ma_1m, atr, buy_grids, sell_grids, positions, pos_count, capital, current_time, trend_regime, rsi_buy_confirmation, rsi_sell_confirmation, volume_confirmation_factor, max_positions, bybit_taker_fee, slippage_percent, max_position_size_percent, kelly_fraction, position_degrading_factor, kelly_lookback, kelly_min_fraction, recent_pnls, recent_pnl_count, atr_initial_stop_multiplier):
    if pos_count >= max_positions:
        return pos_count, capital, 0  # trades_since unchanged
    
    is_buy_trigger = False
    is_sell_trigger = False
    
    for grid_price in buy_grids:
        if grid_price > 0 and tick_price >= grid_price:
            is_buy_trigger = True
            break
    
    for grid_price in sell_grids:
        if grid_price > 0 and tick_price <= grid_price:
            is_sell_trigger = True
            break
    
    if not (is_buy_trigger or is_sell_trigger):
        return pos_count, capital, 0
    
    # Confirmations
    volume_confirmed = volume_1m > volume_ma_1m * volume_confirmation_factor
    if is_buy_trigger:
        rsi_confirmed = rsi_1m > rsi_buy_confirmation
        macd_confirmed = macd_1m > 0
    else:
        rsi_confirmed = rsi_1m < rsi_sell_confirmation
        macd_confirmed = macd_1m < 0
    
    if not (volume_confirmed and rsi_confirmed and macd_confirmed):
        return pos_count, capital, 0
    
    # Find empty pos slot
    for i in range(len(positions)):
        if positions[i, POS_ENTRY_PRICE] == 0:
            entry_price = tick_price * (1 + slippage_percent if is_buy_trigger else 1 - slippage_percent)
            if entry_price == 0:
                break
            size_usd = calculate_position_size(capital, max_position_size_percent, kelly_fraction, position_degrading_factor, trades_since_grid_rebuild, kelly_lookback, kelly_min_fraction, recent_pnls, recent_pnl_count)
            amount = size_usd / entry_price
            positions[i, POS_ENTRY_PRICE] = entry_price
            positions[i, POS_AMOUNT] = amount
            positions[i, POS_IS_LONG] = 1 if is_buy_trigger else 0
            stop_offset = atr * atr_initial_stop_multiplier
            positions[i, POS_STOP_LOSS] = entry_price - stop_offset if is_buy_trigger else entry_price + stop_offset
            positions[i, POS_PEAK_PRICE] = entry_price
            positions[i, POS_ENTRY_TIME] = current_time
            positions[i, POS_ENTRY_REGIME] = trend_regime
            # Risk fields approx
            positions[i, POS_KELLY_RISK_USD] = size_usd * (stop_offset / entry_price)
            positions[i, POS_ATR_RISK_USD] = amount * stop_offset
            pos_count += 1
            break
    
    return pos_count, capital, 1  # Increment trades_since

@nb.jit(nopython=True)
def close_position(i, positions, trade_history, trade_count, capital, current_time, current_price, bybit_taker_fee, slippage_percent, reason):
    if trade_count >= len(trade_history):
        return capital, trade_count
    
    entry_price = positions[i, POS_ENTRY_PRICE]
    amount = positions[i, POS_AMOUNT]
    is_long = positions[i, POS_IS_LONG] == 1
    close_price = current_price * (1 - slippage_percent if is_long else 1 + slippage_percent)
    gross_pnl = (close_price - entry_price) * amount if is_long else (entry_price - close_price) * amount
    commission = (entry_price * amount + close_price * amount) * bybit_taker_fee
    net_pnl = gross_pnl - commission
    capital += net_pnl
    
    entry_amount = entry_price * amount
    pnl_percent = (net_pnl / entry_amount) * 100 if entry_amount > 0 else 0
    
    trade_history[trade_count, TRADE_ENTRY_PRICE] = entry_price
    trade_history[trade_count, TRADE_CLOSE_PRICE] = close_price
    trade_history[trade_count, TRADE_ENTRY_TIME] = positions[i, POS_ENTRY_TIME]
    trade_history[trade_count, TRADE_CLOSE_TIME] = current_time
    trade_history[trade_count, TRADE_PNL_PERCENT] = pnl_percent
    trade_history[trade_count, TRADE_PNL_CASH] = net_pnl
    trade_history[trade_count, TRADE_REASON] = reason
    trade_history[trade_count, TRADE_DIRECTION] = DIR_LONG if is_long else DIR_SHORT
    trade_history[trade_count, TRADE_SIZE] = amount
    trade_history[trade_count, TRADE_KELLY_RISK_USD] = positions[i, POS_KELLY_RISK_USD]
    trade_history[trade_count, TRADE_ATR_RISK_USD] = positions[i, POS_ATR_RISK_USD]
    trade_history[trade_count, TRADE_ENTRY_REGIME] = positions[i, POS_ENTRY_REGIME]
    trade_count += 1
    
    # Reset pos
    positions[i].fill(0.0)
    
    return capital, trade_count

@nb.jit(nopython=True)
def check_exits_on_1m(current_price, atr, positions, pos_count, capital, current_time, trade_history, trade_count, bybit_taker_fee, slippage_percent, max_position_duration_seconds, atr_trailing_stop_activation_multiplier, atr_trailing_stop_multiplier, trailing_ratio, max_drawdown_percent, peak_capital, recent_pnls, recent_pnl_idx, kelly_lookback):
    new_pos_count = pos_count
    new_capital = capital
    new_trade_count = trade_count
    new_recent_pnl_idx = recent_pnl_idx
    
    drawdown = (capital - peak_capital) / peak_capital * 100 if peak_capital > 0 else 0
    if drawdown <= -max_drawdown_percent:
        # Close all
        for j in range(len(positions)):
            if positions[j, POS_ENTRY_PRICE] > 0:
                new_capital, new_trade_count = close_position(j, positions, trade_history, new_trade_count, new_capital, current_time, current_price, bybit_taker_fee, slippage_percent, REASON_DRAWDOWN)
                net_pnl = trade_history[new_trade_count - 1, TRADE_PNL_CASH]
                recent_pnls[new_recent_pnl_idx % kelly_lookback] = net_pnl
                new_recent_pnl_idx += 1
                new_pos_count -= 1
    
    for i in range(len(positions) - 1, -1, -1):  # Reverse to avoid index shifts, but since fill(0), ok
        if positions[i, POS_ENTRY_PRICE] == 0:
            continue
        
        duration = current_time - positions[i, POS_ENTRY_TIME]
        if duration > max_position_duration_seconds:
            new_capital, new_trade_count = close_position(i, positions, trade_history, new_trade_count, new_capital, current_time, current_price, bybit_taker_fee, slippage_percent, REASON_MAX_DURATION)
            net_pnl = trade_history[new_trade_count - 1, TRADE_PNL_CASH]
            recent_pnls[new_recent_pnl_idx % kelly_lookback] = net_pnl
            new_recent_pnl_idx += 1
            new_pos_count -= 1
            continue
        
        # Update peak and trailing
        entry_price = positions[i, POS_ENTRY_PRICE]
        is_long = positions[i, POS_IS_LONG] == 1
        atr_normalized = atr / entry_price if entry_price != 0 else 0
        profit = (current_price - entry_price) / entry_price if is_long else (entry_price - current_price) / entry_price if entry_price != 0 else 0
        
        if is_long:
            if current_price > positions[i, POS_PEAK_PRICE]:
                positions[i, POS_PEAK_PRICE] = current_price
            if profit > atr_trailing_stop_activation_multiplier * atr_normalized:
                positions[i, POS_IS_TRAILING_ACTIVE] = 1
            if positions[i, POS_IS_TRAILING_ACTIVE] == 1:
                new_stop = positions[i, POS_PEAK_PRICE] - atr * atr_trailing_stop_multiplier
                positions[i, POS_STOP_LOSS] = max(positions[i, POS_STOP_LOSS], new_stop)
            if current_price <= positions[i, POS_STOP_LOSS]:
                new_capital, new_trade_count = close_position(i, positions, trade_history, new_trade_count, new_capital, current_time, current_price, bybit_taker_fee, slippage_percent, REASON_TRAILING_STOP)
                net_pnl = trade_history[new_trade_count - 1, TRADE_PNL_CASH]
                recent_pnls[new_recent_pnl_idx % kelly_lookback] = net_pnl
                new_recent_pnl_idx += 1
                new_pos_count -= 1
                continue
        else:
            if current_price < positions[i, POS_PEAK_PRICE]:
                positions[i, POS_PEAK_PRICE] = current_price
            if profit > atr_trailing_stop_activation_multiplier * atr_normalized:
                positions[i, POS_IS_TRAILING_ACTIVE] = 1
            if positions[i, POS_IS_TRAILING_ACTIVE] == 1:
                new_stop = positions[i, POS_PEAK_PRICE] + atr * atr_trailing_stop_multiplier
                positions[i, POS_STOP_LOSS] = min(positions[i, POS_STOP_LOSS], new_stop)
            if current_price >= positions[i, POS_STOP_LOSS]:
                new_capital, new_trade_count = close_position(i, positions, trade_history, new_trade_count, new_capital, current_time, current_price, bybit_taker_fee, slippage_percent, REASON_TRAILING_STOP)
                net_pnl = trade_history[new_trade_count - 1, TRADE_PNL_CASH]
                recent_pnls[new_recent_pnl_idx % kelly_lookback] = net_pnl
                new_recent_pnl_idx += 1
                new_pos_count -= 1
                continue
    
    return new_pos_count, new_capital, new_trade_count, new_recent_pnl_idx

@nb.jit(nopython=True)
def run_numba_simulation(timestamps, opens, highs, lows, closes, volumes, short_emas, long_emas, rsis, atrs, macds, bb_lowers, bb_uppers, rsi_1ms, volume_ma_1ms, macd_1ms, initial_capital, max_positions, bybit_taker_fee, grid_setup_cooldown_seconds, atr_initial_stop_multiplier, atr_trailing_stop_activation_multiplier, trailing_ratio, grid_bb_width_multiplier, confirmation_rsi_period, confirmation_volume_ma_period, atr_trailing_stop_multiplier, position_degrading_factor, num_grids, grid_max_lifespan_seconds, bollinger_period, bollinger_std_dev, rsi_buy_confirmation, rsi_sell_confirmation, volume_confirmation_factor, trend_grid_ratio_normal, trend_grid_ratio_strong, confirmation_macd_fast_period, confirmation_macd_slow_period, confirmation_macd_signal_period, max_position_duration_seconds, grid_volatility_scaling_min, grid_volatility_scaling_max, grid_min_size_percent, grid_max_size_percent, take_profit_percent, max_drawdown_percent, atr_period, max_position_size_percent, kelly_fraction, kelly_lookback, kelly_min_fraction, rsi_period, short_ema_period, long_ema_period, macd_fast_period, macd_slow_period, macd_signal_period, volume_ma_period, trend_ema_threshold_percent, bb_regime_threshold, rsi_oversold, rsi_overbought):
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    last_grid_setup_time = 0.0
    grid_size = 0.01
    trend_regime = 0
    trades_since_grid_rebuild = 0
    total_grids_built = 0
    total_grids_traded = 0
    
    positions = np.zeros((max_positions, 10), dtype=np.float64)
    pos_count = 0
    buy_grids = np.zeros(num_grids, dtype=np.float64)
    sell_grids = np.zeros(num_grids, dtype=np.float64)
    
    max_trades = 10000  # From config
    trade_history = np.zeros((max_trades, 12), dtype=np.float64)
    trade_count = 0
    
    recent_pnls = np.zeros(kelly_lookback, dtype=np.float64)
    recent_pnl_idx = 0
    
    n = len(timestamps)
    equity_curve = np.zeros(n, dtype=np.float64)
    equity_curve[0] = initial_capital
    
    for i in range(1, n):
        current_row_ts = timestamps[i]
        current_row_open = opens[i]
        current_row_high = highs[i]
        current_row_low = lows[i]
        current_row_close = closes[i]
        current_row_volume = volumes[i]
        current_row_short_ema = short_emas[i]
        current_row_long_ema = long_emas[i]
        current_row_rsi = rsis[i]
        current_row_atr = atrs[i]
        current_row_macd = macds[i]
        current_row_bb_lower = bb_lowers[i]
        current_row_bb_upper = bb_uppers[i]
        current_row_rsi_1m = rsi_1ms[i]
        current_row_volume_ma_1m = volume_ma_1ms[i]
        current_row_macd_1m = macd_1ms[i]
        
        prev_row_ts = timestamps[i-1]
        prev_row_close = closes[i-1]
        prev_row_volume = volumes[i-1]
        
        current_time = current_row_ts
        current_price = current_row_close
        
        # Update regime and grid on 30m boundary (assuming timestamp in seconds)
        if int(current_row_ts // 1800) > int(prev_row_ts // 1800):
            trend_regime = get_trend_regime(current_row_short_ema, current_row_long_ema, current_row_rsi, current_row_macd, current_row_bb_upper, current_row_bb_lower, prev_row_close, trend_ema_threshold_percent, bb_regime_threshold, rsi_overbought, rsi_oversold)
            last_grid_setup_time, grid_size, buy_grids, sell_grids = setup_asymmetric_grid(current_time, last_grid_setup_time, grid_setup_cooldown_seconds, prev_row_close, current_row_bb_upper, current_row_bb_lower, trend_regime, grid_min_size_percent, grid_max_size_percent, grid_bb_width_multiplier, num_grids, trend_grid_ratio_normal, trend_grid_ratio_strong, buy_grids, sell_grids)
            if last_grid_setup_time == current_time:
                total_grids_built += 1
                trades_since_grid_rebuild = 0
        
        # Key prices for deterministic path
        if current_row_close > current_row_open:
            key_prices = np.array([current_row_open, current_row_low, current_row_high, current_row_close])
        else:
            key_prices = np.array([current_row_open, current_row_high, current_row_low, current_row_close])
        
        volume_1m_val = prev_row_volume  # From prev closed
        atr_val = current_row_atr
        
        recent_pnl_count = min(kelly_lookback, recent_pnl_idx)
        
        for price in key_prices:
            # Check entries
            inc_trades = 0
            pos_count, capital, inc_trades = check_entries_on_tick(price, current_row_rsi_1m, current_row_macd_1m, volume_1m_val, current_row_volume_ma_1m, atr_val, buy_grids, sell_grids, positions, pos_count, capital, current_time, trend_regime, rsi_buy_confirmation, rsi_sell_confirmation, volume_confirmation_factor, max_positions, bybit_taker_fee, 0.001, max_position_size_percent, kelly_fraction, position_degrading_factor, kelly_lookback, kelly_min_fraction, recent_pnls, recent_pnl_count, atr_initial_stop_multiplier)
            trades_since_grid_rebuild += inc_trades
            if inc_trades > 0:
                total_grids_traded += 1
            
            # Check exits
            pos_count, capital, trade_count, recent_pnl_idx = check_exits_on_1m(price, atr_val, positions, pos_count, capital, current_time, trade_history, trade_count, bybit_taker_fee, 0.001, max_position_duration_seconds, atr_trailing_stop_activation_multiplier, atr_trailing_stop_multiplier, trailing_ratio, max_drawdown_percent, peak_capital, recent_pnls, recent_pnl_idx, kelly_lookback)
        
        # Update peak and drawdown
        if capital > peak_capital:
            peak_capital = capital
        current_drawdown = (capital - peak_capital) / peak_capital * 100 if peak_capital > 0 else 0
        if current_drawdown < -max_drawdown:
            max_drawdown = -current_drawdown
        
        equity_curve[i] = capital
    
    # Trim trade_history
    trade_history_trimmed = trade_history[:trade_count]
    
    total_return_pct = ((capital - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
    pnl_cash = capital - initial_capital
    
    return capital, pnl_cash, total_return_pct, max_drawdown, equity_curve, trade_history_trimmed, total_grids_built, total_grids_traded