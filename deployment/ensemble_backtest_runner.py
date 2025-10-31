# ensemble_backtest_runner.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import logging
from datetime import datetime
import importlib.util

# Import the main VENUES config
from deployment.ensemble_config import VENUES
from simulation.data_fetcher import fetch_all_dataframes
from simulation.backtest_runner import SimulationEngine
from core.trading_bot import AdvancedAdaptiveGridTradingBot
from core.core_types import SimulationClock

# ... (setup_logging and log_prop_firm_summary functions remain the same) ...
def setup_logging():
    log_filename = f"ensemble_backtest_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logging.info(f"Logging configured. Output will be saved to: {log_filename}")

def log_prop_firm_summary(portfolio_metrics: dict):
    if not portfolio_metrics:
        logging.warning("No portfolio metrics to summarize.")
        return
    logging.info("\n" + "="*25 + " CONSOLIDATED PORTFOLIO SUMMARY " + "="*26)
    logging.info("   (Simulating all strategies on a single account)")
    logging.info(f"  Initial Portfolio Capital: ${portfolio_metrics['initial_capital']:,.2f}")
    logging.info(f"  Final Portfolio Capital:   ${portfolio_metrics['final_capital']:,.2f}")
    logging.info(f"  Total Net PnL:             ${portfolio_metrics['pnl_cash']:,.2f}")
    logging.info(f"  Total Portfolio Return:    {portfolio_metrics['total_return_pct']:.2f}%")
    logging.info(f"  **True Max Drawdown**:       {portfolio_metrics['max_drawdown']:.2f}%")
    logging.info("="*80)


def run_ensemble_backtest():
    """Orchestrates a concurrent backtest for all strategies in 'backtest' venues."""
    logging.info("--- Starting Concurrent Ensemble Backtest ---")
    
    # --- The key change is here: Filter for backtest venues ---
    backtest_venues = {name: config for name, config in VENUES.items() if config.get("mode") == "backtest"}
    if not backtest_venues:
        logging.error("No venues with mode: 'backtest' found in ensemble_config.py. Exiting.")
        return

    all_bots = []
    for venue_name, venue_config in backtest_venues.items():
        for strategy_path in venue_config["strategies"]:
            spec = importlib.util.spec_from_file_location(strategy_path, strategy_path)
            strategy_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(strategy_module)
            
            bot = AdvancedAdaptiveGridTradingBot(
                initial_capital=strategy_module.INITIAL_CAPITAL,
                simulation_clock=SimulationClock(0),
                config_module=strategy_module,
                connector=None,
                silent=True
            )
            bot.bot_id = f"{venue_name}_{strategy_path.split('/')[-1].replace('.py', '')}"
            all_bots.append(bot)

    if not all_bots:
        logging.error("No bots were created from the backtest venues. Exiting.")
        return

    # ... (rest of the file remains the same) ...
    first_bot_config = all_bots[0].config
    import config
    config.SYMBOL = first_bot_config.SYMBOL
    config.START_DATE = first_bot_config.START_DATE
    config.END_DATE = first_bot_config.END_DATE
    
    logging.info(f"1. Fetching data for {config.SYMBOL} from {config.START_DATE} to {config.END_DATE}...")
    _, df_1m, _ = fetch_all_dataframes()

    if df_1m.empty:
        logging.error(f"FATAL: 1m dataframe is empty for the specified date range.")
        return
    
    logging.info(f"2. Initializing SimulationEngine with {len(all_bots)} bots...")
    engine = SimulationEngine(df_1m, all_bots, logging)
    portfolio_results, individual_results = engine.run()

    logging.info("\n--- Backtest Finished. Generating Reports ---")
    
    log_prop_firm_summary(portfolio_results)
    
    logging.info("\n" + "="*28 + " INDIVIDUAL STRATEGY STATS " + "="*27)
    for i, bot_metrics in enumerate(individual_results):
        bot_id = all_bots[i].bot_id
        logging.info(f"\n--- Performance for Bot: {bot_id} ---")
        logging.info(bot_metrics.get("performance_log_str", "Performance log not available."))

if __name__ == "__main__":
    setup_logging()
    run_ensemble_backtest()