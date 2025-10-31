# main_shared.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import logging
from datetime import datetime
import config  # It still reads the same base config.py for dates/symbols

# --- *** ADDED THIS MISSING IMPORT *** ---
from simulation.data_fetcher import fetch_all_dataframes
# --- *** END CHANGE *** ---

# --- We import from the new _shared runner, which uses the _shared bot ---
from simulation.backtest_runner_shared import run_single_backtest


def setup_logging():
    """Configures logging for a shared backtest run."""
    log_filename = f"logs/backtest_SHARED_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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
    Main function to set up and run a single backtest using the 
    SHARED (Model C) runner.
    """
    logging.info("--- Inside run_backtest (SHARED) ---")
    logging.info("1. Fetching dataframes...")
    _, df_1m, _ = fetch_all_dataframes() 
    logging.info("2. Data fetching complete.")

    if df_1m.empty:
        logging.error("EXIT POINT: 1m dataframe is empty.")
        return {}

    logging.info("3. Running efficient backtest via SHARED runner (Model C)...")
    # This now calls the run_single_backtest from backtest_runner_shared.py
    performance = run_single_backtest(df_1m, config, verbose=print_log, logger=logging)
    
    if not performance:
        logging.error("EXIT POINT: Backtest failed (insufficient data).")
        return {}
    
    logging.info("9. Simulation finished.")
    logging.info("10. Performance logged.")
    return performance

if __name__ == "__main__":
    setup_logging()
    logging.info("Script execution started (SHARED - Model C).")
    run_backtest()
    logging.info("Script execution finished (SHARED - Model C).")