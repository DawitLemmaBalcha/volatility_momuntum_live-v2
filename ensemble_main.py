# ensemble_main.py
import importlib.util
import sys
import time
import logging
import os # <-- 1. Import the 'os' module
from dotenv import load_dotenv

# --- The Definitive Fix ---
# 2. Construct an explicit path to the .env file in the same directory as the script
dotenv_path = os.path.join(os.path.dirname(__file__), 'api.env')
# 3. Load the .env file from that specific path
load_dotenv(dotenv_path=dotenv_path)

from ensemble_config import VENUES
from trading_bot import AdvancedAdaptiveGridTradingBot
from core_types import SimulationClock, Tick

# --- Setup Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Venue:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.connector = None
        self.bots = []
        self.is_ready = self._setup()

    def _setup(self):
        connector_class = self.config["connector_class"]
        if not connector_class:
            logging.warning(f"Connector for venue '{self.name}' not available. It will be skipped.")
            return False

        connection_params = self.config["connection_params"]
        
        if 'api_key' in connection_params and not connection_params['api_key']:
             logging.error(f"API key for venue '{self.name}' is missing or was not loaded correctly. Check your .env file and ensemble_config.py.")
             return False

        self.connector = connector_class(**connection_params)

        live_balance = self.connector.get_wallet_balance(coin="USD")
        
        if live_balance <= 0:
            logging.error(f"Venue '{self.name}' has zero or invalid balance. Cannot start bots.")
            return False

        num_bots_in_venue = len(self.config["strategies"])
        if num_bots_in_venue == 0:
            logging.warning(f"Venue '{self.name}' has no strategies assigned.")
            return False
            
        capital_per_bot = live_balance / num_bots_in_venue
        logging.info(f"Venue '{self.name}' live balance is {live_balance:,.2f} USD. Allocating {capital_per_bot:,.2f} USD to each of the {num_bots_in_venue} bots.")

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
            logging.info(f"Initialized bot: {bot.bot_id} on Venue: {self.name} for symbol {bot.symbol}")
        
        return True

    def connect(self):
        if self.connector:
            logging.info(f"Connecting Venue: {self.name}...")
            self.connector.connect()

    def disconnect(self):
        if self.connector:
            logging.info(f"Disconnecting Venue: {self.name}...")
            self.connector.disconnect()

class EnsembleTrader:
    def __init__(self, venues_to_run: dict):
        self.venues = [Venue(name, config) for name, config in venues_to_run.items()]
        self.venues = [v for v in self.venues if v.is_ready]
        self.bots_by_symbol = self._map_bots_to_symbols()

    def _map_bots_to_symbols(self):
        mapping = {}
        for venue in self.venues:
            for bot in venue.bots:
                symbol = bot.symbol
                if symbol not in mapping:
                    mapping[symbol] = []
                mapping[symbol].append(bot)
        return mapping

    def _route_tick_to_bots(self, tick: Tick, symbol: str):
        bots_for_symbol = self.bots_by_symbol.get(symbol, [])
        if not bots_for_symbol:
            return

        for bot in bots_for_symbol:
            bot.clock.current_time = tick.timestamp
            bot.clock.current_price = tick.price
            
            bot.check_entries_on_tick(tick, rsi_1m=50, macd_1m=0, volume_ma_1m=0, atr=0)
            bot.check_exits_on_1m(tick.price, atr=0)

    def run(self):
        if not self.venues:
            logging.error("No runnable venues were successfully initialized. Exiting.")
            return
            
        for venue in self.venues:
            venue.connect()

        for venue in self.venues:
            unique_symbols_in_venue = {bot.symbol for bot in venue.bots}
            for symbol in unique_symbols_in_venue:
                 venue.connector.start_data_stream(
                    symbol=symbol,
                    on_tick_callback=self._route_tick_to_bots
                )

        logging.info("\nEnsemble is running. Data streams are active.")
        logging.info("Press Ctrl+C to disconnect and exit.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nShutting down ensemble...")

        for venue in self.venues:
            venue.disconnect()

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