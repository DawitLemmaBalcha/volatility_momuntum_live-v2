# optimize.py

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import optuna
import logging
import pandas as pd
from datetime import datetime
import json
import config
from data_fetcher import fetch_all_dataframes
# --- MODIFIED: Import new simulation functions ---
from backtest_runner import prepare_data_for_simulation, run_simulation_from_prepared_data

# --- A simple, pickle-able class to hold config values ---
class ConfigContainer:
    pass

def setup_optimization_logging():
    """Configures an isolated logger for the optimization process."""
    opt_logger = logging.getLogger('optimization_logger')
    opt_logger.setLevel(logging.INFO)
    opt_logger.propagate = False
    if not opt_logger.handlers:
        log_filename = f"optimization_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        opt_logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        console_handler.setLevel(logging.INFO)
        opt_logger.addHandler(console_handler)
    return opt_logger

# --- MODIFIED: Accepts pre-prepared DataFrame ---
def objective(trial, df_prepared: pd.DataFrame, opt_logger):
    """The objective function for Optuna, with robust logging and parameter constraints."""
    try:
        trial_config = ConfigContainer()
        for attr in dir(config):
            if attr.isupper():
                setattr(trial_config, attr, getattr(config, attr))

        initial_stop_multiplier = trial.suggest_float("ATR_INITIAL_STOP_MULTIPLIER", 0.5, 1.5)
        activation_multiplier = trial.suggest_float("ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER", low=initial_stop_multiplier, high=initial_stop_multiplier * 3.0)
        trailing_ratio = trial.suggest_float("TRAILING_RATIO", 0.1, 0.9)
        
        trial_config.GRID_BB_WIDTH_MULTIPLIER = trial.suggest_float("GRID_BB_WIDTH_MULTIPLIER", 0.03, 0.2)
        trial_config.ATR_INITIAL_STOP_MULTIPLIER = initial_stop_multiplier
        trial_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER = activation_multiplier
        trial_config.ATR_TRAILING_STOP_MULTIPLIER = activation_multiplier * trailing_ratio
        trial_config.CONFIRMATION_RSI_PERIOD = trial.suggest_int("CONFIRMATION_RSI_PERIOD", 15, 30)
        trial_config.CONFIRMATION_VOLUME_MA_PERIOD = trial.suggest_int("CONFIRMATION_VOLUME_MA_PERIOD", 15, 30)

        # --- MODIFIED: Call the simulation runner directly ---
        # This skips the data preparation step, saving significant time
        performance_metrics = run_simulation_from_prepared_data(
            df_prepared, 
            trial_config, 
            verbose=False, 
            logger=opt_logger
        )

        if not performance_metrics:
            return -1.0 

        trial.set_user_attr("performance_log", performance_metrics.pop("performance_log_str", ""))
        trial.set_user_attr("performance", performance_metrics)

        sortino = performance_metrics.get("sortino_ratio", -1.0)

        params_to_log = trial.params.copy()
        params_to_log['ATR_TRAILING_STOP_MULTIPLIER'] = trial_config.ATR_TRAILING_STOP_MULTIPLIER
        
        params_string = json.dumps(params_to_log, indent=4)
        perf_string = json.dumps(performance_metrics, indent=4)
        
        opt_logger.info(
            f"\n--- Trial {trial.number} Complete ---\n"
            f"Objective: Sortino={sortino:.4f}\n"
            f"\nParameters Used:\n{params_string}\n"
            f"\nFull Performance Metrics:\n{perf_string}\n"
            f"---------------------------------"
        )
        
        return sortino
    except Exception as e:
        opt_logger.error(f"TRIAL #{trial.number} FAILED: {e}", exc_info=True)
        return -1.0

if __name__ == "__main__":
    optimization_logger = setup_optimization_logging()
    optimization_logger.info("--- New Single Optimization Run ---")
    optimization_logger.info(f"Optimizing for date range: {config.START_DATE} to {config.END_DATE}")
    
    logging.getLogger('data_fetcher').setLevel(logging.WARNING)
    
    optimization_logger.info("Loading historical data...")
    _, df_1m, _ = fetch_all_dataframes()
    
    # --- NEW: Prepare data ONCE before optimization ---
    optimization_logger.info("Preparing data for all trials...")
    df_prepared = prepare_data_for_simulation(df_1m, config)
    optimization_logger.info("Data prepared. Starting parallel optimization...")
    # --- END NEW ---

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    
    # --- MODIFIED: Pass the prepared DataFrame to the objective ---
    study.optimize(
        lambda trial: objective(trial, df_prepared, optimization_logger),
        n_trials=60,
        n_jobs=-1,
        show_progress_bar=True
    )

    optimization_logger.info("\n" + "="*50)
    optimization_logger.info("OPTIMIZATION FINISHED")
    optimization_logger.info("="*50 + "\n")

    # --- NEW: Get and rank the top 10 trials ---
    try:
        # Filter for successfully completed trials and sort them by Sortino ratio
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > -1.0]
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        
        top_trials = sorted_trials[:10] # Get the top 10

        if not top_trials:
            raise ValueError("No successful trials were completed.")

        optimization_logger.info(f"--- Top {len(top_trials)} Trials Ranked by Sortino Ratio ---")

        for i, trial in enumerate(top_trials):
            rank = i + 1
            sortino = trial.value
            header = f"\n--- Rank #{rank} (Sortino: {sortino:.4f}) ---"
            optimization_logger.info(header)

            is_log_output = trial.user_attrs.get("performance_log", "Performance log not captured.")
            optimization_logger.info("--- In-Sample Performance ---")
            optimization_logger.info(is_log_output)

            optimization_logger.info("  - Parameters:")
            params_to_log = trial.params.copy()
            if "ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER" in params_to_log and "TRAILING_RATIO" in params_to_log:
                params_to_log['ATR_TRAILING_STOP_MULTIPLIER'] = params_to_log["ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER"] * params_to_log["TRAILING_RATIO"]

            for key, value in params_to_log.items():
                optimization_logger.info(f"    - {key}: {value}")

    except ValueError as e:
        optimization_logger.warning(f"{e} No best trials can be shown.")