# ensemble_main.py
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import importlib.util
import time
import logging
import threading
import pandas as pd
import pandas_ta as ta
from collections import deque
from dotenv import load_dotenv
dotenv_path = os.path.join(project_root, 'api.env')
load_dotenv(dotenv_path=dotenv_path)
from typing import List, Dict, Callable

from deployment.ensemble_config import VENUES
from core.trading_bot import AdvancedAdaptiveGridTradingBot
from core.core_types import SimulationClock, Tick

# --- The Definitive Fix ---
dotenv_path = os.path.join(os.path.dirname(__file__), 'api.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURABLE CONSTANTS ---
PORTFOLIO_SYNC_INTERVAL_SECONDS = 10  # How often to sync capital
INDICATOR_PRIMING_CANDLES = 200      # How many candles to fetch on startup
MAX_DATAFRAME_LENGTH = 500          # Max candles to keep in memory

# ==============================================================================
#  NEW: LIVE DATA & INDICATOR MANAGER
# ==============================================================================

class LiveDataManager:
    """
    Manages all live data streams (ticks, 1m, 30m) for a single venue.
    Calculates indicators in real-time and routes data to the correct bots.
    This class solves the "dummy indicator" problem.
    """
    def __init__(self, bots: List[AdvancedAdaptiveGridTradingBot], connector):
        self.bots = bots
        self.connector = connector
        self.bots_by_symbol: Dict[str, List[AdvancedAdaptiveGridTradingBot]] = {}
        
        # Stores the latest calculated indicator values
        self.indicator_state_by_symbol: Dict[str, Dict] = {}
        
        # Stores the rolling candle data
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        # Map bots to symbols
        for bot in self.bots:
            if bot.symbol not in self.bots_by_symbol:
                self.bots_by_symbol[bot.symbol] = []
            self.bots_by_symbol[bot.symbol].append(bot)
            self.indicator_state_by_symbol[bot.symbol] = {} # Init state
            
    def _parse_candle(self, candle_data) -> dict:
        """Helper to parse a single Bybit kline."""
        # Bybit kline format: [timestamp, open, high, low, close, volume, turnover]
        return {
            "timestamp": pd.to_datetime(int(candle_data[0]), unit='ms'),
            "open": float(candle_data[1]),
            "high": float(candle_data[2]),
            "low": float(candle_data[3]),
            "close": float(candle_data[4]),
            "volume": float(candle_data[5]),
        }

    def prime_all_data(self):
        """
        Fetches initial historical data for all symbols to "prime" the
        indicators before starting live streams.
        """
        logger.info(f"Priming indicators for {len(self.bots_by_symbol.keys())} symbol(s)...")
        for symbol in self.bots_by_symbol.keys():
            symbol_bybit = symbol.replace('/', '')
            
            try:
                # --- ASSUMPTION: Connector has a 'get_kline' method ---
                # Fetch initial 1m data
                kline_1m = self.connector.session.get_kline(
                    category="linear", 
                    symbol=symbol_bybit, 
                    interval="1", 
                    limit=INDICATOR_PRIMING_CANDLES
                )['result']['list']
                
                # Fetch initial 30m data
                kline_30m = self.connector.session.get_kline(
                    category="linear", 
                    symbol=symbol_bybit, 
                    interval="30", 
                    limit=INDICATOR_PRIMING_CANDLES
                )['result']['list']

                # Create and store dataframes
                self.dataframes[f"{symbol}_1m"] = pd.DataFrame([self._parse_candle(c) for c in kline_1m]).sort_values(by="timestamp")
                self.dataframes[f"{symbol}_30m"] = pd.DataFrame([self._parse_candle(c) for c in kline_30m]).sort_values(by="timestamp")
                
                # Run initial calculations
                self._calculate_1m_indicators(symbol)
                logger.info(f"Successfully primed data for {symbol}")

            except Exception as e:
                logger.error(f"Failed to prime data for {symbol}: {e}. This symbol will not trade.")
                
    def _calculate_1m_indicators(self, symbol: str):
        """Calculates and stores the latest 1-minute confirmation indicators."""
        df = self.dataframes.get(f"{symbol}_1m")
        if df is None or df.empty:
            return

        # Get the config from the first bot for this symbol (assumes all bots on symbol share params)
        config = self.bots_by_symbol[symbol][0].config
        
        # Calculate indicators
        df['rsi_1m'] = ta.rsi(df['close'], length=config.CONFIRMATION_RSI_PERIOD)
        df['volume_ma_1m'] = df['volume'].rolling(window=config.CONFIRMATION_VOLUME_MA_PERIOD).mean()
        macd_1m = ta.macd(df['close'], fast=config.CONFIRMATION_MACD_FAST_PERIOD, slow=config.CONFIRMATION_MACD_SLOW_PERIOD, signal=config.CONFIRMATION_MACD_SIGNAL_PERIOD)
        df['macd_1m'] = macd_1m[f'MACD_{config.CONFIRMATION_MACD_FAST_PERIOD}_{config.CONFIRMATION_MACD_SLOW_PERIOD}_{config.CONFIRMATION_MACD_SIGNAL_PERIOD}']

        # Store the *latest complete* (t-1) indicator values
        if len(df) > 1:
            latest_indicators = df.iloc[-2] # Use -2 to get the last *closed* candle's data
            self.indicator_state_by_symbol[symbol] = {
                'rsi_1m': latest_indicators['rsi_1m'],
                'macd_1m': latest_indicators['macd_1m'],
                'volume_ma_1m': latest_indicators['volume_ma_1m'],
                'volume_1m': latest_indicators['volume'],
            }

    def _calculate_30m_indicators(self, symbol: str) -> dict:
        """Calculates 30m strategy indicators and returns the latest row."""
        df = self.dataframes.get(f"{symbol}_30m")
        if df is None or df.empty or len(df) < 2:
            return {}
            
        config = self.bots_by_symbol[symbol][0].config
        
        # Calculate all strategy indicators
        df['short_ema'] = ta.ema(df['close'], length=config.SHORT_EMA_PERIOD)
        df['long_ema'] = ta.ema(df['close'], length=config.LONG_EMA_PERIOD)
        df['rsi'] = ta.rsi(df['close'], length=config.RSI_PERIOD)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=config.ATR_PERIOD)
        macd = ta.macd(df['close'], fast=config.MACD_FAST_PERIOD, slow=config.MACD_SLOW_PERIOD, signal=config.MACD_SIGNAL_PERIOD)
        df['macd'] = macd[f'MACD_{config.MACD_FAST_PERIOD}_{config.MACD_SLOW_PERIOD}_{config.MACD_SIGNAL_PERIOD}']
        bbands = ta.bbands(df['close'], length=config.BOLLINGER_PERIOD, std=config.BOLLINGER_STD_DEV)
        # Find bbands columns dynamically (robust fix from backtest_runner)
        bbl_col = [col for col in bbands.columns if col.startswith('BBL_')][0]
        bbu_col = [col for col in bbands.columns if col.startswith('BBU_')][0]
        df['bb_lower'] = bbands[bbl_col]
        df['bb_upper'] = bbands[bbu_col]
        
        # Return the latest *closed* candle's data as a dict
        return df.iloc[-2].to_dict()


    def start_all_streams(self):
        """
        Subscribes to all required streams for all symbols.
        --- ASSUMPTION: Connector has a 'start_kline_stream' method. ---
        """
        logger.info(f"Starting data streams for all symbols...")
        for symbol in self.bots_by_symbol.keys():
            try:
                # 1. Tick Stream (for entries)
                self.connector.start_data_stream(
                    symbol=symbol,
                    on_tick_callback=self._on_tick
                )
                
                # 2. 1-Minute Candle Stream (for confirmation indicators)
                self.connector.start_kline_stream(
                    symbol=symbol,
                    interval="1",
                    on_kline_callback=self._on_1m_candle
                )
                
                # 3. 30-Minute Candle Stream (for strategy logic)
                self.connector.start_kline_stream(
                    symbol=symbol,
                    interval="30",
                    on_kline_callback=self._on_30m_candle
                )
                logger.info(f"Successfully subscribed to Tick, 1m, and 30m streams for {symbol}")
            except Exception as e:
                logger.error(f"Failed to start streams for {symbol}: {e}")

    def _on_1m_candle(self, candle_data, symbol: str):
        """Callback for when a new 1-minute candle arrives."""
        try:
            candle_dict = self._parse_candle(candle_data)
            df = self.dataframes[f"{symbol}_1m"]
            
            # Append new candle and keep dataframe size managed
            new_df = pd.concat([df, pd.DataFrame([candle_dict])], ignore_index=True)
            if len(new_df) > MAX_DATAFRAME_LENGTH:
                new_df = new_df.iloc[-MAX_DATAFRAME_LENGTH:]
            
            self.dataframes[f"{symbol}_1m"] = new_df
            
            # Recalculate and store the latest indicators
            self._calculate_1m_indicators(symbol)
        except Exception as e:
            logger.error(f"Error processing 1m candle for {symbol}: {e}")

    def _on_30m_candle(self, candle_data, symbol: str):
        """Callback for when a new 30-minute candle arrives."""
        try:
            candle_dict = self._parse_candle(candle_data)
            df = self.dataframes[f"{symbol}_30m"]

            new_df = pd.concat([df, pd.DataFrame([candle_dict])], ignore_index=True)
            if len(new_df) > MAX_DATAFRAME_LENGTH:
                new_df = new_df.iloc[-MAX_DATAFRAME_LENGTH:]
            self.dataframes[f"{symbol}_30m"] = new_df
            
            # Get the full data row for the *previous* closed 30m candle
            strategy_data_row = self._calculate_30m_indicators(symbol)
            if not strategy_data_row:
                return

            # Update all bots for this symbol
            for bot in self.bots_by_symbol[symbol]:
                bot.update_strategy_on_30m(strategy_data_row)
                
        except Exception as e:
            logger.error(f"Error processing 30m candle for {symbol}: {e}")

    def _on_tick(self, tick: Tick, symbol: str):
        """
        Callback for when a new tick arrives.
        This is the new 'route_tick_to_bots'
        """
        try:
            # Get the latest *stored* 1m indicators
            indicators = self.indicator_state_by_symbol.get(symbol, {})
            if not indicators:
                # Data is not primed yet, skip
                return

            # Get the latest *stored* 30m ATR (from the 30m data row)
            # This is a bit of a shortcut; ideally, the 30m function would store this.
            atr_30m = self.dataframes[f"{symbol}_30m"]['atr'].iloc[-2] if f"{symbol}_30m" in self.dataframes and 'atr' in self.dataframes[f"{symbol}_30m"].columns else 0.0

            # Route to all bots for this symbol
            for bot in self.bots_by_symbol[symbol]:
                bot.clock.current_time = tick.timestamp
                bot.clock.current_price = tick.price
                
                # --- THIS IS THE FIX ---
                # Pass the *real, stored* indicator values
                bot.check_entries_on_tick(
                    tick,
                    rsi_1m=indicators.get('rsi_1m', 50),
                    macd_1m=indicators.get('macd_1m', 0),
                    volume_1m=indicators.get('volume_1m', 0),
                    volume_ma_1m=indicators.get('volume_ma_1m', 0),
                    atr=atr_30m
                )
                bot.check_exits_on_1m(tick.price, atr=atr_30m)
        except Exception as e:
            logger.error(f"Error processing tick for {symbol}: {e}", exc_info=True)


# ==============================================================================
#  MODIFIED: VENUE CLASS
# ==============================================================================

class Venue:
    """
    Manages a single trading venue (e.g., one exchange account).
    This class now runs the portfolio sync loop in a separate thread.
    This solves the "shared capital" problem.
    """
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.connector = None
        self.bots: List[AdvancedAdaptiveGridTradingBot] = []
        self.is_ready = False
        
        self.master_peak_capital = 0.0
        self.data_manager: LiveDataManager = None
        self.portfolio_sync_thread: threading.Thread = None
        
        self.is_ready = self._setup()

    def _setup(self):
        connector_class = self.config["connector_class"]
        if not connector_class:
            logger.warning(f"Connector for venue '{self.name}' not available. It will be skipped.")
            return False

        connection_params = self.config["connection_params"]
        
        if 'api_key' in connection_params and not connection_params['api_key']:
             logger.error(f"API key for venue '{self.name}' is missing or was not loaded correctly. Check your .env file and ensemble_config.py.")
             return False

        self.connector = connector_class(**connection_params)

        # --- ASSUMPTION: Connector has 'get_wallet_balance' ---
        live_balance = self.connector.get_wallet_balance(coin="USD") # Bybit uses USD or USDT
        
        if live_balance <= 0:
            logger.error(f"Venue '{self.name}' has zero or invalid balance ({live_balance}). Cannot start bots.")
            return False
        
        self.master_peak_capital = live_balance # Initialize peak capital

        num_bots_in_venue = len(self.config["strategies"])
        if num_bots_in_venue == 0:
            logger.warning(f"Venue '{self.name}' has no strategies assigned.")
            return False
            
        # This initial capital is just a *starting point* for the bot's internal math.
        # The portfolio sync will override it on its first run.
        capital_per_bot = live_balance / num_bots_in_venue
        logger.info(f"Venue '{self.name}' live balance is {live_balance:,.2f} USD. Allocating {capital_per_bot:,.2f} USD per bot (as initial value).")

        for strategy_path in self.config["strategies"]:
            spec = importlib.util.spec_from_file_location(strategy_path, strategy_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)

            clock = SimulationClock(start_time=time.time())
            bot = AdvancedAdaptiveGridTradingBot(
                initial_capital=capital_per_bot,
                simulation_clock=clock,
                config_module=strategy_module,
                connector=self.connector,
                silent=False
            )
            bot.bot_id = f"{self.name}_{strategy_path.split('/')[-1].replace('.py', '')}"
            self.bots.append(bot)
            logger.info(f"Initialized bot: {bot.bot_id} on Venue: {self.name} for symbol {bot.symbol}")
        
        # Setup the data manager for this venue's bots
        self.data_manager = LiveDataManager(self.bots, self.connector)
        
        return True

    def connect_and_run_streams(self):
        """Connects, primes data, and starts all data streams."""
        if self.connector:
            logger.info(f"Connecting Venue: {self.name}...")
            self.connector.connect()
            
            # Prime data *before* starting streams
            self.data_manager.prime_all_data()
            
            # Start all streams (Tick, 1m, 30m)
            self.data_manager.start_all_streams()

    def start_portfolio_sync(self):
        """Starts the portfolio synchronization loop in a separate thread."""
        logger.info(f"Starting portfolio sync thread for venue '{self.name}'...")
        self.portfolio_sync_thread = threading.Thread(target=self._portfolio_sync_loop, daemon=True)
        self.portfolio_sync_thread.start()

    def _portfolio_sync_loop(self):
        """
        [THREADED] This loop runs continuously to sync the true portfolio
        state with all bot instances.
        """
        while True:
            try:
                # 1. Get Realized Capital
                # --- ASSUMPTION: Connector has 'get_wallet_balance' ---
                realized_capital = self.connector.get_wallet_balance("USD")
                
                # 2. Get Unrealized PNL
                unrealized_pnl = 0.0
                
                # --- ASSUMPTION: Connector has 'get_open_positions' & 'get_mark_price' ---
                # TODO: This block MUST be implemented for your specific connector
                #
                # try:
                #     open_positions = self.connector.get_open_positions() # e.g., returns list of position dicts
                #     for pos in open_positions:
                #         symbol = pos['symbol']
                #         entry_price = float(pos['entry_price'])
                #         qty = float(pos['qty'])
                #         side = pos['side'] # 'Buy' or 'Sell'
                #         mark_price = self.connector.get_mark_price(symbol)
                #
                #         if side == 'Buy':
                #             unrealized_pnl += (mark_price - entry_price) * qty
                #         else:
                #             unrealized_pnl += (entry_price - mark_price) * qty
                #
                # except Exception as e:
                #     logger.warning(f"Could not calculate UPL: {e}. UPL will be 0.")
                #
                if unrealized_pnl == 0.0:
                    logger.warning(f"Unrealized PNL calculation is not implemented or failed. Using UPL=0.0. "
                                     f"Portfolio drawdown and sizing logic will be based on REALIZED capital only.")

                # 3. Update Master Capital States
                current_total_equity = realized_capital + unrealized_pnl
                self.master_peak_capital = max(self.master_peak_capital, current_total_equity)

                # 4. Inject State into ALL Bots
                # This makes all bots aware of the true portfolio state
                for bot in self.bots:
                    bot.capital = realized_capital
                    bot.unrealized_pnl = unrealized_pnl
                    bot.peak_capital = self.master_peak_capital
                
                logger.debug(f"[{self.name}] Portfolio Synced: "
                            f"Realized={realized_capital:,.2f}, "
                            f"UPL={unrealized_pnl:,.2f}, "
                            f"Peak={self.master_peak_capital:,.2f}")

            except Exception as e:
                logger.error(f"CRITICAL: Portfolio sync loop failed for venue '{self.name}': {e}")
            
            time.sleep(PORTFOLIO_SYNC_INTERVAL_SECONDS)


    def disconnect(self):
        if self.connector:
            logger.info(f"Disconnecting Venue: {self.name}...")
            self.connector.disconnect()

# ==============================================================================
#  MODIFIED: ENSEMBLE TRADER (Orchestrator)
# ==============================================================================

class EnsembleTrader:
    def __init__(self, venues_to_run: dict):
        # Create all venue objects (which also runs their _setup)
        self.venues = [Venue(name, config) for name, config in venues_to_run.items()]
        # Filter out any venues that failed to set up
        self.venues = [v for v in self.venues if v.is_ready]

    def run(self):
        if not self.venues:
            logger.error("No runnable venues were successfully initialized. Exiting.")
            return
            
        for venue in self.venues:
            # Connect, prime data, and start Tick/1m/30m streams
            venue.connect_and_run_streams()
            
            # Start the separate portfolio sync thread
            venue.start_portfolio_sync()

        logger.info("\n" + "="*50)
        logger.info(" ENSEMBLE IS LIVE AND RUNNING")
        logger.info(" All data streams and portfolio sync threads are active.")
        logger.info(" Press Ctrl+C to disconnect and exit.")
        logger.info("="*50 + "\n")

        try:
            while True:
                # Main thread just keeps the process alive
                # All work is done in the connector's data threads
                # and our portfolio sync threads.
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nShutting down ensemble...")

        for venue in self.venues:
            venue.disconnect()

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: Please specify which venue to run.")
        print("Usage: python ensemble_main.py <venue_name>")
        print("\nAvailable venues in ensemble_config.py:")
        for name, config in VENUES.items():
            if config.get("mode") in ["live", "paper_trading"]:
                print(f"  - {name}")
        sys.exit(1)

    venue_to_run_name = sys.argv[1]
    
    venues_to_run = {name: config for name, config in VENUES.items() if name == venue_to_run_name}
    
    if not venues_to_run:
        print(f"ERROR: Venue '{venue_to_run_name}' not found in ensemble_config.py.")
        sys.exit(1)

    ensemble = EnsembleTrader(venues_to_run)
    ensemble.run()