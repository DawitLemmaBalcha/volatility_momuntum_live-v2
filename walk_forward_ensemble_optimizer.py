# walk_forward_ensemble_optimizer.py

import os
import logging
from datetime import datetime
import pandas as pd
import optuna
import json

# Import necessary components from your existing project files
import config
from data_fetcher import fetch_all_dataframes
from optimize import ConfigContainer  # We need the container for building bots
from trading_bot import AdvancedAdaptiveGridTradingBot
from backtest_runner import SimulationEngine # The core engine for OOS testing

# --- Global list to store results from all walks ---
all_walks_data = []

# --- Helper function to create a specialized objective ---
def create_objective_function(parameter_range: dict):
    """
    Creates a tailored objective function for Optuna based on a defined parameter range.
    """
    def objective(trial, df_1m_full: pd.DataFrame, opt_logger):
        try:
            trial_config = ConfigContainer()
            for attr in dir(config):
                if attr.isupper():
                    setattr(trial_config, attr, getattr(config, attr))

            # Use the specific parameter range for this objective
            initial_stop_multiplier = trial.suggest_float("ATR_INITIAL_STOP_MULTIPLIER", 0.5, 1.5)
            activation_multiplier = trial.suggest_float("ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER", low=initial_stop_multiplier, high=initial_stop_multiplier * 2.0)
            trailing_ratio = trial.suggest_float("TRAILING_RATIO", 0.1, 0.9)
            
            trial_config.GRID_BB_WIDTH_MULTIPLIER = trial.suggest_float("GRID_BB_WIDTH_MULTIPLIER", 0.05, 0.2)
            trial_config.ATR_INITIAL_STOP_MULTIPLIER = initial_stop_multiplier
            trial_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER = activation_multiplier
            trial_config.ATR_TRAILING_STOP_MULTIPLIER = activation_multiplier * trailing_ratio

            # *** The key difference is here ***
            trial_config.CONFIRMATION_RSI_PERIOD = trial.suggest_int("CONFIRMATION_RSI_PERIOD", parameter_range["rsi_min"], parameter_range["rsi_max"])
            trial_config.CONFIRMATION_VOLUME_MA_PERIOD = trial.suggest_int("CONFIRMATION_VOLUME_MA_PERIOD", parameter_range["vol_min"], parameter_range["vol_max"])

            # This requires a small change in backtest_runner to accept a simple config object
            from backtest_runner import run_single_backtest
            performance_metrics = run_single_backtest(df_1m_full, trial_config, verbose=False)

            if not performance_metrics: return -1.0
            
            trial.set_user_attr("performance", performance_metrics)
            return performance_metrics.get("sortino_ratio", -1.0)
        except Exception as e:
            opt_logger.error(f"TRIAL FAILED: {e}", exc_info=False) # Keep logs cleaner
            return -1.0
    return objective

def run_ensemble_walk(walk_number: int, is_start: str, is_end: str, oos_end: str, n_trials: int, opt_logger):
    """
    Executes a full walk-forward cycle with dual optimization and ensemble OOS testing.
    """
    opt_logger.info(f"--- Starting Walk #{walk_number} | IS: {is_start} to {is_end} | OOS: {is_end} to {oos_end} ---")

    # 1. --- Fetch and slice data ---
    config.START_DATE, config.END_DATE = is_start, oos_end
    try:
        _, df_full, _ = fetch_all_dataframes()
        df_is = df_full[(df_full['timestamp'] >= pd.to_datetime(is_start)) & (df_full['timestamp'] < pd.to_datetime(is_end))].copy()
        df_oos = df_full[(df_full['timestamp'] >= pd.to_datetime(is_end)) & (df_full['timestamp'] < pd.to_datetime(oos_end))].copy()
        if df_is.empty or df_oos.empty:
            opt_logger.warning("Skipping walk due to empty in-sample or out-of-sample data.")
            return
    except Exception as e:
        opt_logger.error(f"Failed to fetch or slice data: {e}", exc_info=True)
        return

    # 2. --- Run Dual Optimizations ---
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    # Aggressive Optimization
    opt_logger.info(f"--- Running AGGRESSIVE optimization ({n_trials} trials)... ---")
    aggressive_params = {"rsi_min": 25, "rsi_max": 35, "vol_min": 25, "vol_max": 35}
    study_aggressive = optuna.create_study(direction="maximize")
    study_aggressive.optimize(lambda trial: create_objective_function(aggressive_params)(trial, df_is, opt_logger), n_trials=n_trials, n_jobs=-1)

    # Conservative Optimization
    opt_logger.info(f"--- Running CONSERVATIVE optimization ({n_trials} trials)... ---")
    conservative_params = {"rsi_min": 35, "rsi_max": 45, "vol_min": 35, "vol_max": 45}
    study_conservative = optuna.create_study(direction="maximize")
    study_conservative.optimize(lambda trial: create_objective_function(conservative_params)(trial, df_is, opt_logger), n_trials=n_trials, n_jobs=-1)

    # 3. --- Select Top Performers (Ranks 2-5) ---
    def select_top_bots(study, study_name):
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value > -1.0]
        if len(completed) < 5:
            opt_logger.warning(f"Not enough successful trials in {study_name} study to select top 4 (found {len(completed)}).")
            return []
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
        return sorted_trials[1:5] # Ranks 2, 3, 4, 5

    top_aggressive = select_top_bots(study_aggressive, "Aggressive")
    top_conservative = select_top_bots(study_conservative, "Conservative")
    
    if not top_aggressive and not top_conservative:
        opt_logger.error("No successful trials in either study. Cannot run OOS test.")
        return

    opt_logger.info(f"Selected {len(top_aggressive)} aggressive and {len(top_conservative)} conservative bots for OOS ensemble.")

    # 4. --- Run Out-of-Sample Ensemble Backtest ---
    ensemble_bots = []
    
    # Create bot configurations from selected trials
    for trial in top_aggressive + top_conservative:
        bot_config = ConfigContainer()
        for attr in dir(config):
            if attr.isupper():
                setattr(bot_config, attr, getattr(config, attr))
        
        for key, value in trial.params.items():
            setattr(bot_config, key, value)
        
        bot_config.ATR_TRAILING_STOP_MULTIPLIER = bot_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER * bot_config.TRAILING_RATIO
        
        bot = AdvancedAdaptiveGridTradingBot(
            initial_capital=bot_config.INITIAL_CAPITAL, 
            simulation_clock=None, # Engine will provide this
            config_module=bot_config,
            connector=None, # Engine will provide this
            silent=True
        )
        ensemble_bots.append(bot)

    # Run the concurrent simulation
    opt_logger.info(f"--- Running OOS ENSEMBLE backtest with {len(ensemble_bots)} bots... ---")
    engine = SimulationEngine(df_oos, ensemble_bots, opt_logger)
    portfolio_results, _ = engine.run()
    
    if not portfolio_results:
        opt_logger.error("Ensemble backtest failed to produce results.")
        return
        
    # 5. --- Log and Save Results ---
    opt_logger.info("\n" + "="*25 + f" OOS ENSEMBLE PERFORMANCE | WALK #{walk_number} " + "="*25)
    opt_logger.info(f"  Initial Portfolio Capital: ${portfolio_results['initial_capital']:,.2f}")
    opt_logger.info(f"  Final Portfolio Capital:   ${portfolio_results['final_capital']:,.2f}")
    opt_logger.info(f"  Total Net PnL:             ${portfolio_results['pnl_cash']:,.2f}")
    opt_logger.info(f"  Total Portfolio Return:    {portfolio_results['total_return_pct']:.2f}%")
    opt_logger.info(f"  True Max Drawdown:         {portfolio_results['max_drawdown']:.2f}%")
    opt_logger.info("="*87 + "\n")

    portfolio_results['walk'] = walk_number
    all_walks_data.append(portfolio_results)


if __name__ == "__main__":
    for env_var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[env_var] = '1'
    
    log_filename = f"walk_forward_ensemble_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    main_logger = logging.getLogger('wf_ensemble_logger')
    main_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    main_logger.addHandler(file_handler)
    main_logger.addHandler(console_handler)

    walk_forward_schedule = [
        #{'is_start': '2024-09-01 00:00:00', 'is_end': '2024-12-01 00:00:00', 'oos_end': '2025-01-01 00:00:00'},
        {'is_start': '2024-10-01 00:00:00', 'is_end': '2025-01-01 00:00:00', 'oos_end': '2025-02-01 00:00:00'},
        {'is_start': '2024-11-01 00:00:00', 'is_end': '2025-02-01 00:00:00', 'oos_end': '2025-03-01 00:00:00'},
        {'is_start': '2024-12-01 00:00:00', 'is_end': '2025-03-01 00:00:00', 'oos_end': '2025-04-01 00:00:00'},
        {'is_start': '2025-01-01 00:00:00', 'is_end': '2025-04-01 00:00:00', 'oos_end': '2025-05-01 00:00:00'},
        {'is_start': '2025-02-01 00:00:00', 'is_end': '2025-05-01 00:00:00', 'oos_end': '2025-06-01 00:00:00'},
        {'is_start': '2025-03-01 00:00:00', 'is_end': '2025-06-01 00:00:00', 'oos_end': '2025-07-01 00:00:00'},
        {'is_start': '2025-04-01 00:00:00', 'is_end': '2025-07-01 00:00:00', 'oos_end': '2025-08-01 00:00:00'},
        {'is_start': '2025-05-01 00:00:00', 'is_end': '2025-08-01 00:00:00', 'oos_end': '2025-09-01 00:00:00'},
    ]
    
    trials_per_optimization = 50 # 50 for aggressive, 50 for conservative

    main_logger.info("--- Starting Full Walk-Forward Ensemble Analysis ---")
    for i, walk_dates in enumerate(walk_forward_schedule):
        walk_num = i + 1
        run_ensemble_walk(
            walk_number=walk_num,
            is_start=walk_dates['is_start'],
            is_end=walk_dates['is_end'],
            oos_end=walk_dates['oos_end'],
            n_trials=trials_per_optimization,
            opt_logger=main_logger
        )
    main_logger.info("\n--- All Walk-Forward Ensemble Analysis runs are complete. ---")

    if all_walks_data:
        results_df = pd.DataFrame(all_walks_data)
        results_filename = f"walk_forward_ensemble_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        results_df.to_csv(results_filename, index=False)
        main_logger.info(f"\nSuccessfully saved all walk-forward ensemble data to: {results_filename}")