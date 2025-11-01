# walk_forward_optimizer.py
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import logging
from datetime import datetime
import pandas as pd
import optuna
import json
import copy

# Import necessary components from your existing project files
import config
from simulation.data_fetcher import fetch_all_dataframes
from analysis.optimize import objective, ConfigContainer # Import the objective and the container
# --- MODIFIED: Import new simulation functions ---
from simulation.backtest_runner import prepare_data_for_simulation, run_simulation_from_prepared_data

# --- NEW: Global list to store results from all walks ---
all_trials_data = []

def run_single_walk(walk_number: int, is_start: str, is_end: str, oos_end: str, n_trials: int):
    """
    Executes a full walk-forward cycle with comprehensive logging for both
    In-Sample (IS) optimization and Out-of-Sample (OOS) testing.
    """
    # ... (logging setup remains the same) ...
    log_filename = f"logs/walk_forward_log_WALK_{walk_number}.log"
    opt_logger = logging.getLogger(f'walk_{walk_number}_logger')
    opt_logger.setLevel(logging.INFO)
    opt_logger.propagate = False
    
    if opt_logger.hasHandlers():
        opt_logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    opt_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(f'WALK {walk_number}: %(message)s'))
    console_handler.setLevel(logging.INFO)
    opt_logger.addHandler(console_handler)

    opt_logger.info(f"--- Starting Walk #{walk_number} ---")
    opt_logger.info(f"In-Sample Period: {is_start} to {is_end}")
    opt_logger.info(f"Out-of-Sample Period: {is_end} to {oos_end}")


    # 2. --- Fetch and slice data ---
    config.START_DATE, config.END_DATE = is_start, oos_end
    opt_logger.info("Loading historical data for the full walk period...")
    try:
        _, df_full, _ = fetch_all_dataframes()
        df_is = df_full[(df_full['timestamp'] >= pd.to_datetime(is_start)) & (df_full['timestamp'] < pd.to_datetime(is_end))].copy()
        df_oos = df_full[(df_full['timestamp'] >= pd.to_datetime(is_end)) & (df_full['timestamp'] < pd.to_datetime(oos_end))].copy()
        
        if df_is.empty or len(df_is) < 200:
             opt_logger.error(f"Skipping walk {walk_number}. In-sample data is empty or too small ({len(df_is)} records).")
             return
        if df_oos.empty or len(df_oos) < 200:
             opt_logger.error(f"Skipping walk {walk_number}. Out-of-sample data is empty or too small ({len(df_oos)} records).")
             return

        # --- NEW: Prepare IS data ONCE ---
        opt_logger.info(f"Data loaded. Preparing {len(df_is)} IS records...")
        df_is_prepared = prepare_data_for_simulation(df_is, config)
        opt_logger.info(f"IS data prepared. {len(df_is_prepared)} records usable.")
        
        if df_is_prepared.empty:
            opt_logger.error(f"Skipping walk {walk_number}. In-sample data was unusable after preparation.")
            return

    except Exception as e:
        opt_logger.error(f"Failed to fetch or slice data: {e}", exc_info=True)
        return

    # 3. --- Run In-Sample (IS) Optimization ---
    opt_logger.info(f"--- PHASE 1: IN-SAMPLE OPTIMIZATION for {n_trials} trials ---")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize") 
    
    # --- MODIFIED: Pass prepared IS data to objective ---
    study.optimize(
        lambda trial: objective(trial, df_is_prepared, opt_logger),
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )
    opt_logger.info("\n" + "="*80)
    opt_logger.info(f"========== OPTIMIZATION FOR WALK #{walk_number} FINISHED ==========")
    opt_logger.info("="*80 + "\n")


    # 4. --- Log IS Results, Run OOS Tests, and Collect Data ---
    opt_logger.info("\n" + "="*50)
    opt_logger.info(f"--- PHASE 2: OOS TESTING & IN-SAMPLE RESULTS ---")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
    
    top_n_trials = 10
    best_trials = sorted_trials[:top_n_trials]

    if not best_trials:
        opt_logger.warning("No successful trials completed. Cannot run OOS tests.")
    else:
        opt_logger.info(f"Found {len(best_trials)} best trials to analyze (out of {len(completed_trials)} completed).")
        for i, trial in enumerate(best_trials):
            rank = i + 1
            sortino_score = trial.value

            opt_logger.info("\n" + "#"*80)
            opt_logger.info(f"### Analysing RANK #{rank} (Sortino: {sortino_score:.4f}) for WALK #{walk_number} ###")
            opt_logger.info("#"*80)
            
            opt_logger.info(f"\n--- IN-SAMPLE PERFORMANCE [RANK #{rank}] ---")
            
            optimal_params = trial.params.copy()
            opt_logger.info(f"Parameters Used (In-Sample):\n{json.dumps(optimal_params, indent=4)}")

            is_log_output = trial.user_attrs.get("performance_log", "Performance log not captured.")
            opt_logger.info(is_log_output)

            opt_logger.info(f"\n--- OUT-OF-SAMPLE PERFORMANCE [RANK #{rank}] ---")
            opt_logger.info(f"Parameters being tested (Out-of-Sample):\n{json.dumps(optimal_params, indent=4)}")
            
            # --- MODIFIED: Create a config container for the OOS backtest ---
            oos_config = ConfigContainer()
            for attr in dir(config):
                if attr.isupper():
                    setattr(oos_config, attr, getattr(config, attr))

            # Apply the optimal parameters from the trial to the config
            for key, value in optimal_params.items():
                setattr(oos_config, key, value)
            
            # Recalculate the dependent parameter
            oos_config.ATR_TRAILING_STOP_MULTIPLIER = oos_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER * oos_config.TRAILING_RATIO

            # --- NEW: Prepare OOS data ONCE for this trial ---
            df_oos_prepared = prepare_data_for_simulation(df_oos, oos_config)
            
            if df_oos_prepared.empty:
                opt_logger.error("OOS data was unusable after preparation. Skipping OOS test.")
                oos_performance = {}
            else:
                # --- MODIFIED: Run the OOS backtest using the new function ---
                oos_performance = run_simulation_from_prepared_data(
                    df_oos_prepared, 
                    oos_config, 
                    logger=opt_logger, 
                    verbose=True
                )
            
            opt_logger.info(f"--- OOS TEST for RANK #{rank} COMPLETE ---")

            is_performance = trial.user_attrs.get("performance", {})
            if not oos_performance:
                oos_performance = {}

            trial_data = {
                'walk': walk_number,
                'is_rank': rank,
                **{f'param_{k}': v for k, v in optimal_params.items()},
                'is_return_pct': is_performance.get('total_return_pct'),
                'is_sortino': is_performance.get('sortino_ratio'),
                'is_max_drawdown': is_performance.get('max_drawdown'),
                'oos_return_pct': oos_performance.get('total_return_pct'),
                'oos_sortino': oos_performance.get('sortino_ratio'),
                'oos_max_drawdown': oos_performance.get('max_drawdown'),
                'oos_win_rate': oos_performance.get('win_rate'),
                'oos_profit_factor': oos_performance.get('profit_factor'),
                'oos_total_trades': oos_performance.get('total_trades')
            }
            all_trials_data.append(trial_data)
    
    opt_logger.info("\n" + "="*50 + f"\nWALK #{walk_number} COMPLETE\n" + "="*50 + "\n")

# ... (rest of the file remains the same) ...
if __name__ == "__main__":
    for env_var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[env_var] = '1'
    
    walk_forward_schedule = [
        {'is_start': '2024-09-01 00:00:00', 'is_end': '2024-12-01 00:00:00', 'oos_end': '2025-01-01 00:00:00'},
        {'is_start': '2024-10-01 00:00:00', 'is_end': '2025-01-01 00:00:00', 'oos_end': '2025-02-01 00:00:00'},
        {'is_start': '2024-11-01 00:00:00', 'is_end': '2025-02-01 00:00:00', 'oos_end': '2025-03-01 00:00:00'},
        {'is_start': '2024-12-01 00:00:00', 'is_end': '2025-03-01 00:00:00', 'oos_end': '2025-04-01 00:00:00'},
        {'is_start': '2025-01-01 00:00:00', 'is_end': '2025-04-01 00:00:00', 'oos_end': '2025-05-01 00:00:00'},
        {'is_start': '2025-02-01 00:00:00', 'is_end': '2025-05-01 00:00:00', 'oos_end': '2025-06-01 00:00:00'},
        {'is_start': '2025-03-01 00:00:00', 'is_end': '2025-06-01 00:00:00', 'oos_end': '2025-07-01 00:00:00'},
        {'is_start': '2025-04-01 00:00:00', 'is_end': '2025-07-01 00:00:00', 'oos_end': '2025-08-01 00:00:00'},
        {'is_start': '2025-05-01 00:00:00', 'is_end': '2025-08-01 00:00:00', 'oos_end': '2025-09-01 00:00:00'},
    ]
    
    trials_per_walk = 50

    print("--- Starting Full Walk-Forward Analysis ---")
    for i, walk_dates in enumerate(walk_forward_schedule):
        walk_num = i + 1
        print(f"\n>>> PROCESSING WALK {walk_num} of {len(walk_forward_schedule)}...")
        run_single_walk(
            walk_number=walk_num,
            is_start=walk_dates['is_start'],
            is_end=walk_dates['is_end'],
            oos_end=walk_dates['oos_end'],
            n_trials=trials_per_walk
        )
    print("\n--- All Walk-Forward Analysis runs are complete. ---")

    if all_trials_data:
        results_df = pd.DataFrame(all_trials_data)
        results_filename = f"results/walk_forward_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        results_df.to_csv(results_filename, index=False)
        print(f"\nSuccessfully saved all trial data to: {results_filename}")
    else:
        print("\nNo trial data was generated to save.")