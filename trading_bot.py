# trading_bot.py

import logging
import time
import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from core_types import Position, Order, SimulationClock, TradeLog, Tick
from connectors.base_connector import BaseConnector

default_logger = logging.getLogger(__name__)

class AdvancedAdaptiveGridTradingBot:
    def __init__(self, initial_capital: float, simulation_clock: SimulationClock, config_module, connector: BaseConnector, silent: bool = False):
        self.config = config_module # <-- The key change is here
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.clock = simulation_clock
        self.connector = connector
        self.silent = silent
        
        self.symbol = self.config.SYMBOL
        self.kelly_fraction = self.config.KELLY_FRACTION
        self.max_position_size = self.config.MAX_POSITION_SIZE_PERCENT
        
        self.open_positions, self.grid_orders, self.trade_history = [], [], []
        self.max_drawdown, self.last_grid_setup_time, self.grid_size, self.trend_regime = 0.0, 0, 0.01, "uninitialized"

    def initialize_state(self, initial_data_row: dict):
        self.trend_regime = self._get_trend_regime(initial_data_row)
        self.setup_asymmetric_grid("Initial Setup", initial_data_row)

    def _get_trend_regime(self, data_row: dict) -> str:
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
        current_time = self.clock.time()
        if current_time - self.last_grid_setup_time < self.config.GRID_SETUP_COOLDOWN_SECONDS: return
        current_price = data_row['close']
        self.grid_orders.clear()
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
        self.grid_orders = [Order(current_price * (1 + i * self.grid_size), self.calculate_position_size(current_price * (1 + i * self.grid_size)), "buy") for i in range(1, upper_grids + 1)] + [Order(current_price * (1 - i * self.grid_size), self.calculate_position_size(current_price * (1 - i * self.grid_size)), "sell") for i in range(1, lower_grids + 1)]
        if not self.silent:
            buy_grids = [o for o in self.grid_orders if o.order_type == 'buy']
            sell_grids = [o for o in self.grid_orders if o.order_type == 'sell']
            default_logger.info(f"\n[GRID REBUILT] Reason: {reason}\n  - Center Price:    {current_price:,.2f}\n  - Trend/Regime:    {self.trend_regime}\n  - Grid Size:       {self.grid_size:.3%} (was {old_grid_size:.3%})\n  - Setup:           {len(buy_grids)} Buy Grids | {len(sell_grids)} Sell Grids\n  - Buy Levels:      [ {', '.join([f'{o.price:,.2f}' for o in buy_grids])} ]\n  - Sell Levels:     [ {', '.join([f'{o.price:,.2f}' for o in sell_grids])} ]")
        self.last_grid_setup_time = current_time

    def update_strategy_on_30m(self, data_row: dict):
        old_trend_regime = self.trend_regime
        self.trend_regime = self._get_trend_regime(data_row)
        rebuild_reason = None
        if self.trend_regime != old_trend_regime: rebuild_reason = f"Strategy Shift to {self.trend_regime}"
        elif not self.open_positions and self.clock.time() - self.last_grid_setup_time > getattr(self.config, 'GRID_MAX_LIFESPAN_SECONDS', 14400): rebuild_reason = "Grid Lifespan Expired"
        if rebuild_reason: self.setup_asymmetric_grid(rebuild_reason, data_row)
        self.update_kelly_fraction()
        if self.capital > self.peak_capital: self.peak_capital = self.capital
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital * 100 if self.peak_capital > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        if current_drawdown > self.config.MAX_DRAWDOWN_PERCENT: self.gradual_de_risk()

    def check_exits_on_1m(self, current_price: float, atr: float):
        for position in self.open_positions[:]:
            if self.clock.time() - position.entry_time > getattr(self.config, 'MAX_POSITION_DURATION_SECONDS', 86400): self.close_position(position, current_price, "Time Stop"); continue
            if not (atr > 0 and pd.notna(atr)): continue
            if position.stop_loss is None: position.stop_loss = position.entry_price - (atr * self.config.ATR_INITIAL_STOP_MULTIPLIER) if position.is_long else position.entry_price + (atr * self.config.ATR_INITIAL_STOP_MULTIPLIER)
            activation_price = position.entry_price + (atr * self.config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER) if position.is_long else position.entry_price - (atr * self.config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER)
            if not position.is_trailing_active and ((position.is_long and current_price >= activation_price) or (not position.is_long and current_price <= activation_price)):
                position.is_trailing_active = True
                if not self.silent: default_logger.info(f"Trailing stop activated for position {position.id}.")
            if position.is_trailing_active:
                trailing_distance = atr * self.config.ATR_TRAILING_STOP_MULTIPLIER
                if position.is_long:
                    position.peak_price = max(position.peak_price, current_price)
                    position.stop_loss = max(position.stop_loss, position.peak_price - trailing_distance)
                else:
                    position.peak_price = min(position.peak_price, current_price)
                    position.stop_loss = min(position.stop_loss, position.peak_price + trailing_distance)
            if position.stop_loss is not None and ((position.is_long and current_price <= position.stop_loss) or (not position.is_long and current_price >= position.stop_loss)):
                self.close_position(position, current_price, "Trail Stop" if position.is_trailing_active else "ATR Stop")

    def check_entries_on_tick(self, tick: Tick, rsi_1m: float, macd_1m: float, volume_ma_1m: float, atr: float):
        for order in self.grid_orders[:]:
            if (order.order_type == "buy" and tick.price >= order.price) or (order.order_type == "sell" and tick.price <= order.price):
                volume_ok = tick.candle_volume > (volume_ma_1m * getattr(self.config, 'VOLUME_CONFIRMATION_FACTOR', 1.0))
                if order.order_type == "buy": rsi_ok, macd_ok = rsi_1m > self.config.RSI_BUY_CONFIRMATION, macd_1m > 0
                else: rsi_ok, macd_ok = rsi_1m < self.config.RSI_SELL_CONFIRMATION, macd_1m < 0
                if rsi_ok and macd_ok and volume_ok:
                    details = {"price_ok": f"{tick.price:,.2f} {' >=' if order.order_type == 'buy' else ' <='} {order.price:,.2f}", "rsi_ok": rsi_ok, "rsi_val": rsi_1m, "rsi_req": getattr(self.config, f'RSI_{"BUY" if order.order_type == "buy" else "SELL"}_CONFIRMATION'), "macd_ok": macd_ok, "macd_val": macd_1m, "volume_ok": volume_ok, "volume_val": tick.candle_volume, "volume_req": volume_ma_1m * getattr(self.config, 'VOLUME_CONFIRMATION_FACTOR', 1.0)}
                    self.execute_trade(order, details, atr)
                    self.grid_orders.remove(order)

    def execute_trade(self, order: Order, confirmation_details: dict, atr: float):
        if len(self.open_positions) >= self.config.MAX_POSITIONS or order.amount <= 0: return
        direction = 'LONG' if order.order_type == 'buy' else 'SHORT'
        result = self.connector.place_order(symbol=self.symbol, side=order.order_type, order_type="Market", qty=order.amount, price=order.price)
        if not result or not result.get("success"):
            if not self.silent: default_logger.error("Connector failed to place order.")
            return
        entry_price = result['entry_price']
        new_position = Position(entry_price, order.amount, direction == 'LONG')
        new_position.id, new_position.entry_time, new_position.entry_regime = result['trade_id'], self.clock.time(), self.trend_regime
        stop_distance = atr * self.config.ATR_INITIAL_STOP_MULTIPLIER
        new_position.kelly_risk_usd, new_position.atr_risk_usd = entry_price * new_position.amount, stop_distance * new_position.amount
        self.open_positions.append(new_position)
        if not self.silent:
            initial_stop_price = (entry_price - stop_distance) if new_position.is_long else (entry_price + stop_distance)
            default_logger.info(f"\n[TRADE OPEN] {pd.to_datetime(self.clock.time(), unit='s')} | {direction} @ {entry_price:,.2f}\n  - Entry Regime:   {new_position.entry_regime}\n  - Risk Profile:\n    - Position Value (Kelly Risk): ${new_position.kelly_risk_usd:,.2f}\n    - Stop-Loss Risk (ATR Risk):  -${new_position.atr_risk_usd:,.2f}\n    - Initial Stop Price:          ${initial_stop_price:,.2f}\n  - Justification:\n    - Price ({confirmation_details['price_ok']}) -> [PASS]\n    - RSI ({confirmation_details['rsi_val']:.2f} {'>' if direction == 'LONG' else '<'} {confirmation_details['rsi_req']}) -> [{'PASS' if confirmation_details['rsi_ok'] else 'FAIL'}]\n    - MACD ({confirmation_details['macd_val']:.2f} {'> 0' if direction == 'LONG' else '< 0'}) -> [{'PASS' if confirmation_details['macd_ok'] else 'FAIL'}]\n    - Volume ({confirmation_details['volume_val']:,.2f} > {confirmation_details['volume_req']:,.2f}) -> [{'PASS' if confirmation_details['volume_ok'] else 'FAIL'}]")

    def gradual_de_risk(self):
        risk_reduction = min(0.5, ((self.peak_capital - self.capital) / self.peak_capital) / self.config.MAX_DRAWDOWN_PERCENT)
        self.max_position_size *= (1 - risk_reduction)
        self.kelly_fraction *= (1 - risk_reduction)
        if not self.silent: default_logger.info(f"De-risking due to drawdown. New max size: {self.max_position_size:.2%}, New Kelly: {self.kelly_fraction:.2f}")

    def close_position(self, position: Position, current_price: float, reason: str):
        capital_before = self.capital
        close_result = self.connector.close_position(position)
        if not close_result or not close_result.get("success"):
            if not self.silent: default_logger.error(f"Connector failed to close position {position.id}")
            return
        profit = close_result['pnl']
        self.capital += profit
        profit_percent = (profit / (position.entry_price * position.amount)) * 100 if position.entry_price * position.amount != 0 else 0
        self.trade_history.append(TradeLog(entry_price=position.entry_price, close_price=close_result['close_price'], entry_time=position.entry_time, close_time=self.clock.time(), pnl_percent=profit_percent, pnl_cash=profit, reason=reason, direction='LONG' if position.is_long else 'SHORT', size=position.amount, kelly_risk_usd=position.kelly_risk_usd, atr_risk_usd=position.atr_risk_usd, entry_regime=position.entry_regime))
        if position in self.open_positions: self.open_positions.remove(position)
        if not self.silent:
            duration_str, gross_pnl = time.strftime('%Hh %Mm %Ss', time.gmtime(self.clock.time() - position.entry_time)), profit + close_result.get('commission', 0)
            default_logger.info(f"\n[TRADE CLOSE] {pd.to_datetime(self.clock.time(), unit='s')} | {'LONG' if position.is_long else 'SHORT'} | Reason: {reason}\n  - Duration:         {duration_str}\n  - Entry Price:      {position.entry_price:,.2f}\n  - Close Price:      {close_result['close_price']:,.2f}\n  - Gross PnL:        ${gross_pnl:,.2f}\n  - Commission Paid:  -${close_result.get('commission', 0):,.2f}\n  - Net PnL:          ${profit:,.2f} ({profit_percent:.2f}%)\n  - Portfolio Impact: Capital ${capital_before:,.2f} -> ${self.capital:,.2f}")

    def calculate_position_size(self, price: float) -> float:
        if price <= 0: return 0.0
        unrealized_pnl = sum(pos.calculate_profit(self.clock.current_price) for pos in self.open_positions) if self.clock.current_price > 0 else 0
        effective_capital = min(self.capital + unrealized_pnl, self.peak_capital)
        return (effective_capital * self.max_position_size * self.kelly_fraction) / price

    def update_kelly_fraction(self):
        if len(self.trade_history) < self.config.KELLY_LOOKBACK: return
        recent_pnls = [t.pnl_percent / 100 for t in self.trade_history[-self.config.KELLY_LOOKBACK:]]
        wins = [p for p in recent_pnls if p > 0]
        if not wins: self.kelly_fraction = self.config.KELLY_MIN_FRACTION; return
        win_rate, avg_win = len(wins) / len(recent_pnls), np.mean(wins)
        avg_loss = abs(np.mean([p for p in recent_pnls if p <= 0])) if any(p <= 0 for p in recent_pnls) else 1.0
        if avg_loss > 0: self.kelly_fraction = max(self.config.KELLY_MIN_FRACTION, min(win_rate - (1 - win_rate) / (avg_win / avg_loss), self.config.KELLY_FRACTION))

    def log_performance(self, print_log: bool = False, logger=None) -> dict:
        if logger is None:
            logger = default_logger
        pnl_cash = self.capital - self.initial_capital
        total_return_pct = (pnl_cash / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        metrics = {"total_return_pct": total_return_pct, "pnl_cash": pnl_cash, "max_drawdown": self.max_drawdown, "sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0, "win_rate": 0, "profit_factor": 0, "total_trades": len(self.trade_history), "regime_performance": {}}
        if len(self.trade_history) > 1:
            metrics['win_rate'] = sum(1 for t in self.trade_history if t.pnl_cash > 0) / len(self.trade_history) * 100
            gross_profit = sum(t.pnl_cash for t in self.trade_history if t.pnl_cash > 0) or 1
            gross_loss = abs(sum(t.pnl_cash for t in self.trade_history if t.pnl_cash < 0)) or 1
            metrics['profit_factor'] = gross_profit / gross_loss
            returns = [t.pnl_percent / 100 for t in self.trade_history]
            avg_return, std_dev = np.mean(returns), np.std(returns)
            metrics['sharpe_ratio'] = (avg_return / std_dev) * np.sqrt(252) if std_dev > 0 else 0
            neg_returns = [r for r in returns if r < 0]
            downside_dev = np.std(neg_returns) if len(neg_returns) > 1 else 0
            metrics['sortino_ratio'] = (avg_return / downside_dev) * np.sqrt(252) if downside_dev > 0 else (999.0 if avg_return > 0 else 0)
            if self.max_drawdown > 0: metrics['calmar_ratio'] = total_return_pct / self.max_drawdown
            elif total_return_pct > 0: metrics['calmar_ratio'] = 999.0
            else: metrics['calmar_ratio'] = 0.0
            regime_trades = defaultdict(list)
            for t in self.trade_history: regime_trades[t.entry_regime].append(t)
            for regime, trades in regime_trades.items():
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
        log_msg = (f"\n{'='*20} PORTFOLIO STATUS | {pd.to_datetime(self.clock.time(), unit='s')} {'='*20}\n" f"  Core Metrics:\n" f"    - Total Return:    {metrics['total_return_pct']:,.2f}%\n" f"    - PnL:             ${metrics['pnl_cash']:,.2f}\n" f"    - Current Capital: ${self.capital:,.2f}\n" f"  Risk & Performance:\n" f"    - Peak Capital:    ${self.peak_capital:,.2f}\n" f"    - Max Drawdown:    -{metrics['max_drawdown']:.2f}%\n" f"    - Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}\n" f"    - Sortino Ratio:   {metrics['sortino_ratio']:.2f}\n" f"    - Calmar Ratio:    {metrics['calmar_ratio']:.2f}\n" f"  Trade Stats:\n" f"    - Total Trades:    {metrics['total_trades']}\n" f"    - Win Rate:        {metrics.get('win_rate', 0):.2f}% ({wins} W / {losses} L)\n" f"    - Profit Factor:   {metrics.get('profit_factor', 0):.2f}\n" f"    - Avg Winner ($):  ${avg_winner_usd:,.2f}\n" f"    - Avg Loser ($):   ${avg_loser_usd:,.2f}\n" f"  Risk Stats (Average Per Trade):\n" f"    - Avg Position Value: ${avg_pos_value:,.2f}\n" f"  Current State:\n" f"    - Open Positions:  {len(self.open_positions)} ({open_longs} L, {open_shorts} S)\n" f"    - Trend/Regime:    {self.trend_regime}\n" f"    - Kelly Fraction:  {self.kelly_fraction:.2f}\n")
        if metrics["regime_performance"]:
            log_msg += f"  {'='*20} Regime Performance Breakdown {'='*21}\n" f"  {'Regime':<25} | {'Trades':>8} | {'Win Rate':>10} | {'Profit Factor':>15} | {'Net PnL':>15}\n" f"  {'-'*78}\n"
            for regime, stats in sorted(metrics["regime_performance"].items()):
                log_msg += f"  {regime:<25} | {stats['total_trades']:>8} | {stats['win_rate']:>10} | {stats['profit_factor']:>15} | {stats['net_pnl']:>15}\n"
        log_msg += f"{'='*78}"
        return log_msg