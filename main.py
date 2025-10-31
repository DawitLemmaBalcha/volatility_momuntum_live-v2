import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import logging
from datetime import datetime
import config
from simulation.data_fetcher import fetch_all_dataframes
from simulation.backtest_runner import run_single_backtest

def setup_logging():
    """Configures logging for a standard backtest run."""
    log_filename = f"backtest_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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

def run_backtest(print_log=True):
    """
    Main function to set up and run a single, detailed backtest using the efficient runner.
    """
    logging.info("--- Inside run_backtest ---")
    logging.info("1. Fetching dataframes...")
    _, df_1m, _ = fetch_all_dataframes() # Ignore df_30m and df_4h; derive 30m from 1m
    logging.info("2. Data fetching complete.")

    if df_1m.empty:
        logging.error("EXIT POINT: 1m dataframe is empty. Check data_fetcher.py or your date range.")
        return {}

    logging.info("3. Running efficient backtest via runner...")
    # --- NEW: Delegate to runner for consistent, optimized simulation (empty params = defaults) ---
    performance = run_single_backtest(df_1m, config, verbose=print_log, logger=logging)
    
    if not performance:
        logging.error("EXIT POINT: Backtest failed (insufficient data).")
        return {}
    
    logging.info("9. Simulation finished.")
    # The detailed performance is now logged within the bot's log_performance method
    # when verbose=True is passed to the runner.
    logging.info("10. Performance logged.")
    return performance

if __name__ == "__main__":
    setup_logging()
    logging.info("Script execution started.")
    run_backtest()
    logging.info("Script execution finished.")