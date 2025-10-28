# walk_forward_ensemble_optimizer.py

import os
import logging
from datetime import datetime
import pandas as pd
import optuna
import json

# --- NEW: Add imports for clustering ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- END NEW ---

# Import necessary components from your existing project files
import config
from data_fetcher import fetch_all_dataframes
from optimize import ConfigContainer  # We need the container for building bots
from trading_bot import AdvancedAdaptiveGridTradingBot
from backtest_runner import SimulationEngine, run_single_backtest

# --- Global list to store results from all walks ---
all_walks_data = []

# --- MODIFIED: Helper function to create a *single* wide objective ---
def create_wide_objective_function():
    """
    Creates a tailored objective function for Optuna with wide parameter ranges
    to allow the optimizer to explore a large landscape.
    """
    def objective(trial, df_1m_full: pd.DataFrame, opt_logger):
        try:
            trial_config = ConfigContainer()
            for attr in dir(config):
                if attr.isupper():
                    setattr(trial_config, attr, getattr(config, attr))

            # --- MODIFIED: Use wide parameter ranges ---
            initial_stop_multiplier = trial.suggest_float("ATR_INITIAL_STOP_MULTIPLIER", 0.5, 2.0)
            activation_multiplier = trial.suggest_float("ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER", low=initial_stop_multiplier, high=initial_stop_multiplier * 4.0)
            trailing_ratio = trial.suggest_float("TRAILING_RATIO", 0.1, 0.9)
            
            trial_config.GRID_BB_WIDTH_MULTIPLIER = trial.suggest_float("GRID_BB_WIDTH_MULTIPLIER", 0.03, 0.3)
            trial_config.ATR_INITIAL_STOP_MULTIPLIER = initial_stop_multiplier
            trial_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER = activation_multiplier
            trial_config.ATR_TRAILING_STOP_MULTIPLIER = activation_multiplier * trailing_ratio

            trial_config.CONFIRMATION_RSI_PERIOD = trial.suggest_int("CONFIRMATION_RSI_PERIOD", 15, 50)
            trial_config.CONFIRMATION_VOLUME_MA_PERIOD = trial.suggest_int("CONFIRMATION_VOLUME_MA_PERIOD", 15, 50)
            # --- END MODIFIED ---

            performance_metrics = run_single_backtest(df_1m_full, trial_config, verbose=False)

            if not performance_metrics: return -1.0
            
            # Save all performance metrics, we need 'total_trades' for filtering
            trial.set_user_attr("performance", performance_metrics)
            return performance_metrics.get("sortino_ratio", -1.0)
        
        except Exception as e:
            opt_logger.error(f"TRIAL FAILED: {e}", exc_info=False) # Keep logs cleaner
            return -1.0
    return objective
# --- END MODIFIED ---


# --- MODIFIED: Main walk function to use K-Means ---
def run_ensemble_walk(walk_number: int, is_start: str, is_end: str, oos_end: str, n_trials: int, n_ensemble_bots: int, opt_logger):
    """
    Executes a full walk-forward cycle with a single wide optimization
    and K-Means clustering for diverse ensemble selection.
    """
    opt_logger.info(f"--- Starting Walk #{walk_number} | IS: {is_start} to {is_end} | OOS: {is_end} to {oos_end} ---")

    # 1. --- Fetch and slice data --- (No changes here)
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

    # 2. --- Run ONE Single, Wide Optimization ---
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    opt_logger.info(f"--- Running WIDE optimization ({n_trials} trials)... ---")
    objective_func = create_wide_objective_function()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_func(trial, df_is, opt_logger), n_trials=n_trials, n_jobs=-1, show_progress_bar=True)

    # 3. --- NEW: Filter, Cluster, and Select Top Performers ---
    
    # 3a. Filter: Get all high-quality, completed trials
    completed_trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            perf = t.user_attrs.get("performance", {})
            sortino = perf.get("sortino_ratio", -1.0)
            total_trades = perf.get("total_trades", 0)
            
            # --- Quality Filter ---
            if sortino > 0.5 and total_trades > 20:
                completed_trials.append(t)

    opt_logger.info(f"Optimization complete. Found {len(completed_trials)} high-quality trials (Sortino > 0.5, Trades > 20).")

    if len(completed_trials) < n_ensemble_bots:
        opt_logger.warning(f"Found only {len(completed_trials)} trials, which is less than the desired ensemble size of {n_ensemble_bots}. Using all found trials.")
        ensemble_bots_trials = completed_trials
    else:
        # 3b. Cluster: Prepare data and run K-Means
        data_for_clustering = []
        for trial in completed_trials:
            row = trial.params.copy()
            row['sortino'] = trial.value
            row['trial_object'] = trial  # Keep a reference to the trial
            data_for_clustering.append(row)
        
        df = pd.DataFrame(data_for_clustering)
        param_cols = [col for col in df.columns if col not in ['sortino', 'trial_object']]
        
        # Normalize parameters for stable clustering
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[param_cols])

        kmeans = KMeans(n_clusters=n_ensemble_bots, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df_scaled)
        
        opt_logger.info(f"Clustering complete. Grouped trials into {n_ensemble_bots} clusters.")

        # 3c. Select: Get the best trial (by Sortino) from each cluster
        best_trials_from_clusters = df.loc[df.groupby('cluster')['sortino'].idxmax()]
        ensemble_bots_trials = best_trials_from_clusters['trial_object'].tolist()
    
    # --- END NEW ---

    if not ensemble_bots_trials:
        opt_logger.error("No successful trials were selected. Cannot run OOS test.")
        return

    opt_logger.info(f"Selected {len(ensemble_bots_trials)} diverse bots for OOS ensemble.")

    # 4. --- Run Out-of-Sample Ensemble Backtest --- (No changes here)
    ensemble_bots = []
    
    # Create bot configurations from selected trials
    for trial in ensemble_bots_trials:
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
        
    # 5. --- Log and Save Results --- (No changes here)
    opt_logger.info("\n" + "="*25 + f" OOS ENSEMBLE PERFORMANCE | WALK #{walk_number} " + "="*25)
    opt_logger.info(f"  Initial Portfolio Capital: ${portfolio_results['initial_capital']:,.2f}")
    opt_logger.info(f"  Final Portfolio Capital:   ${portfolio_results['final_capital']:,.2f}")
    opt_logger.info(f"  Total Net PnL:             ${portfolio_results['pnl_cash']:,.2f}")
    opt_logger.info(f"  Total Portfolio Return:    {portfolio_results['total_return_pct']:.2f}%")
    opt_logger.info(f"  True Max Drawdown:         {portfolio_results['max_drawdown']:.2f}%")
    opt_logger.info("="*87 + "\n")

    portfolio_results['walk'] = walk_number
    all_walks_data.append(portfolio_results)
# --- END MODIFIED ---


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
    
    # --- MODIFIED: Set ONE high trial count and a cluster count ---
    # We run ONE study per walk, so n_trials should be high.
    trials_per_optimization_walk = 50 # <-- Run a large number of trials
    number_of_bots_to_cluster = 4    # <-- Select 8 diverse bots
    # --- END MODIFIED ---

    main_logger.info("--- Starting Full Walk-Forward Ensemble Analysis (with K-Means Clustering) ---")
    for i, walk_dates in enumerate(walk_forward_schedule):
        walk_num = i + 1
        run_ensemble_walk(
            walk_number=walk_num,
            is_start=walk_dates['is_start'],
            is_end=walk_dates['is_end'],
            oos_end=walk_dates['oos_end'],
            n_trials=trials_per_optimization_walk,         # <-- Use new variable
            n_ensemble_bots=number_of_bots_to_cluster,  # <-- Use new variable
            opt_logger=main_logger
        )
    main_logger.info("\n--- All Walk-Forward Ensemble Analysis runs are complete. ---")

    if all_walks_data:
        results_df = pd.DataFrame(all_walks_data)
        results_filename = f"walk_forward_ensemble_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        results_df.to_csv(results_filename, index=False)
        main_logger.info(f"\nSuccessfully saved all walk-forward ensemble data to: {results_filename}")