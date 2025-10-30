# walk_forward_ensemble_optimizer_shared.py

import os
import logging
from datetime import datetime
import pandas as pd
import optuna
import json

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config
from data_fetcher import fetch_all_dataframes
from optimize import ConfigContainer  

# --- *** NEW: Importing from SHARED files *** ---
from trading_bot_shared import AdvancedAdaptiveGridTradingBot
from backtest_runner_shared import (
    SimulationEngine, 
    prepare_data_for_simulation, 
    run_simulation_from_prepared_data # Note: This is still Model A, which is correct for *optimization*
)
# --- *** END NEW *** ---

all_walks_data = []

def create_wide_objective_function():
    """
    Creates the objective function for Optuna.
    This function tests ONE bot at a time (Model A), which is what we want
    for finding good *individual* strategies before combining them.
    """
    def objective(trial, df_prepared: pd.DataFrame, opt_logger):
        try:
            trial_config = ConfigContainer()
            for attr in dir(config):
                if attr.isupper():
                    setattr(trial_config, attr, getattr(config, attr))

            initial_stop_multiplier = trial.suggest_float("ATR_INITIAL_STOP_MULTIPLIER", 0.5, 2.0)
            activation_multiplier = trial.suggest_float("ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER", low=initial_stop_multiplier, high=initial_stop_multiplier * 4.0)
            trailing_ratio = trial.suggest_float("TRAILING_RATIO", 0.1, 0.9)
            
            trial_config.GRID_BB_WIDTH_MULTIPLIER = trial.suggest_float("GRID_BB_WIDTH_MULTIPLIER", 0.03, 0.3)
            trial_config.ATR_INITIAL_STOP_MULTIPLIER = initial_stop_multiplier
            trial_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER = activation_multiplier
            trial_config.ATR_TRAILING_STOP_MULTIPLIER = activation_multiplier * trailing_ratio

            trial_config.CONFIRMATION_RSI_PERIOD = trial.suggest_int("CONFIRMATION_RSI_PERIOD", 15, 50)
            trial_config.CONFIRMATION_VOLUME_MA_PERIOD = trial.suggest_int("CONFIRMATION_VOLUME_MA_PERIOD", 15, 50)

            # --- This call uses run_simulation_from_prepared_data ---
            # --- It correctly runs a (Model A) test on ONE bot ---
            performance_metrics = run_simulation_from_prepared_data(
                df_prepared, 
                trial_config, 
                verbose=False, 
                logger=opt_logger,
                trial=trial 
            )

            if not performance_metrics: return -1.0
            
            trial.set_user_attr("performance", performance_metrics)
            return performance_metrics.get("sortino_ratio", -1.0)
        
        except optuna.TrialPruned:
            opt_logger.info(f"--- Trial {trial.number} Pruned ---")
            return -1.0
        except Exception as e:
            opt_logger.error(f"TRIAL FAILED: {e}", exc_info=False)
            return -1.0
    return objective


def run_ensemble_walk(walk_number: int, is_start: str, is_end: str, oos_end: str, n_trials: int, n_clusters: int, top_n_per_cluster: int, opt_logger):
    """
    Executes a full walk-forward cycle using (Model C) Shared Risk, Divided Sizing.
    """
    opt_logger.info(f"--- Starting Walk #{walk_number} (SHARED) | IS: {is_start} to {is_end} | OOS: {is_end} to {oos_end} ---")

    # 1. --- Fetch and slice data ---
    config.START_DATE, config.END_DATE = is_start, oos_end
    try:
        _, df_full, _ = fetch_all_dataframes()
        df_is = df_full[(df_full['timestamp'] >= pd.to_datetime(is_start)) & (df_full['timestamp'] < pd.to_datetime(is_end))].copy()
        df_oos = df_full[(df_full['timestamp'] >= pd.to_datetime(is_end)) & (df_full['timestamp'] < pd.to_datetime(oos_end))].copy()
        if df_is.empty or df_oos.empty:
            opt_logger.warning("Skipping walk due to empty in-sample or out-of-sample data.")
            return
            
        opt_logger.info(f"Preparing {len(df_is)} IS records...")
        df_is_prepared = prepare_data_for_simulation(df_is, config) # From backtest_runner_shared
        
        opt_logger.info(f"Preparing {len(df_oos)} OOS records...")
        df_oos_prepared = prepare_data_for_simulation(df_oos, config) # From backtest_runner_shared
        
        if df_is_prepared.empty or df_oos_prepared.empty:
            opt_logger.warning("Skipping walk due to empty prepared data (IS or OOS).")
            return
            
    except Exception as e:
        opt_logger.error(f"Failed to fetch or slice data: {e}", exc_info=True)
        return

    # 2. --- Run ONE Single, Wide Optimization (Model A) ---
    # (This is correct - we are finding the best *individual* bots first)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    opt_logger.info(f"--- Running WIDE optimization ({n_trials} trials)... ---")
    objective_func = create_wide_objective_function()
    study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda trial: objective_func(trial, df_is_prepared, opt_logger), 
        n_trials=n_trials, 
        n_jobs=-1, 
        show_progress_bar=True
    )


    # 3. --- Filter, Cluster, and Select Top Performers ---
    completed_trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            perf = t.user_attrs.get("performance", {})
            sortino = perf.get("sortino_ratio", -1.0)
            total_trades = perf.get("total_trades", 0)
            
            if sortino > 1 and total_trades > 20:
                completed_trials.append(t)

    opt_logger.info(f"Optimization complete. Found {len(completed_trials)} high-quality trials (Sortino > 1, Trades > 20).")

    if len(completed_trials) < n_clusters:
        opt_logger.warning(f"Found only {len(completed_trials)} trials, which is less than the desired cluster count of {n_clusters}. Using all found trials.")
        ensemble_bots_trials = completed_trials
    else:
        data_for_clustering = []
        for trial in completed_trials:
            row = trial.params.copy()
            row['sortino'] = trial.value
            row['trial_object'] = trial 
            data_for_clustering.append(row)
        
        df = pd.DataFrame(data_for_clustering)
        param_cols = [col for col in df.columns if col not in ['sortino', 'trial_object']]
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[param_cols])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df_scaled)
        
        opt_logger.info(f"Clustering complete. Grouped trials into {n_clusters} clusters.")

        best_trials_from_clusters = df.groupby('cluster').apply(
            lambda x: x.nlargest(top_n_per_cluster, 'sortino')
        ).reset_index(drop=True)
        ensemble_bots_trials = best_trials_from_clusters['trial_object'].tolist()

    if not ensemble_bots_trials:
        opt_logger.error("No successful trials were selected. Cannot run OOS test.")
        return

    # --- Enhanced Logging Block ---
    opt_logger.info(f"--- Selected {len(ensemble_bots_trials)} Trials for OOS Ensemble ---")
    for _, trial_row in best_trials_from_clusters.sort_values(by=['cluster', 'sortino'], ascending=[True, False]).iterrows():
        trial = trial_row['trial_object']
        cluster = trial_row['cluster']
        sortino = trial_row['sortino']
        
        opt_logger.info(f"  [Cluster {cluster}] - Trial {trial.number} | In-Sample Sortino: {sortino:.4f}")
        params_to_log = trial.params.copy()
        params_to_log['ATR_TRAILING_STOP_MULTIPLIER'] = params_to_log.get("ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER", 1.0) * params_to_log.get("TRAILING_RATIO", 0.5)
        opt_logger.info(f"    Params: {json.dumps(params_to_log, indent=2)}")
        is_perf = trial.user_attrs.get("performance", {})
        is_perf_str = (
            f"    IS Perf: Return={is_perf.get('total_return_pct', 0):.2f}%, "
            f"Trades={is_perf.get('total_trades', 0)}, "
            f"WinRate={is_perf.get('win_rate', 0):.2f}%, "
            f"PF={is_perf.get('profit_factor', 0):.2f}"
        )
        opt_logger.info(is_perf_str)
    # --- End Logging Block ---

    # 4. --- *** NEW: Run Out-of-Sample Ensemble Backtest (Model C) *** ---
    opt_logger.info(f"--- Preparing {len(ensemble_bots_trials)} bots for (Model C) OOS ensemble... ---")
    ensemble_bots = []
    
    # --- *** NEW: Calculate allocation per bot *** ---
    num_bots = len(ensemble_bots_trials)
    allocation_per_bot = 1.0 / num_bots
    opt_logger.info(f"Total capital will be shared. Each bot has a sizing allocation of {allocation_per_bot:.2%}")
    
    for trial in ensemble_bots_trials:
        bot_config = ConfigContainer()
        for attr in dir(config):
            if attr.isupper():
                setattr(bot_config, attr, getattr(config, attr))
        
        for key, value in trial.params.items():
            setattr(bot_config, key, value)
        
        bot_config.ATR_TRAILING_STOP_MULTIPLIER = bot_config.ATR_TRAILING_STOP_ACTIVATION_MULTIPLIER * bot_config.TRAILING_RATIO
        
        bot = AdvancedAdaptiveGridTradingBot( # from trading_bot_shared
            initial_capital=bot_config.INITIAL_CAPITAL, # Engine will override this
            simulation_clock=None, 
            config_module=bot_config,
            connector=None, 
            silent=True
        )
        
        # --- *** NEW: Set the bot's capital allocation *** ---
        bot.capital_allocation = allocation_per_bot
        
        ensemble_bots.append(bot)

    # --- Run the SHARED (Model C) simulation ---
    opt_logger.info(f"--- Running OOS (Model C) ENSEMBLE backtest... ---")
    engine = SimulationEngine( # from backtest_runner_shared
        df_oos_prepared,
        ensemble_bots, 
        opt_logger, 
        prepared=True
    )
    portfolio_results, _ = engine.run()
    # --- END NEW ---
    
    if not portfolio_results:
        opt_logger.error("Ensemble backtest failed to produce results.")
        return
        
    # 5. --- Log and Save Results ---
    opt_logger.info("\n" + "="*25 + f" OOS ENSEMBLE PERFORMANCE (SHARED) | WALK #{walk_number} " + "="*25)
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
    
    log_filename = f"walk_forward_ensemble_SHARED_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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
        #{'is_start': '2024-10-01 00:00:00', 'is_end': '2025-01-01 00:00:00', 'oos_end': '2025-02-01 00:00:00'},
        #{'is_start': '2024-11-01 00:00:00', 'is_end': '2025-02-01 00:00:00', 'oos_end': '2025-03-01 00:00:00'},
        #{'is_start': '2024-12-01 00:00:00', 'is_end': '2025-03-01 00:00:00', 'oos_end': '2025-04-01 00:00:00'},
        #{'is_start': '2025-01-01 00:00:00', 'is_end': '2025-04-01 00:00:00', 'oos_end': '2025-05-01 00:00:00'},
        #{'is_start': '2025-02-01 00:00:00', 'is_end': '2025-05-01 00:00:00', 'oos_end': '2025-06-01 00:00:00'},
        {'is_start': '2025-03-01 00:00:00', 'is_end': '2025-06-01 00:00:00', 'oos_end': '2025-07-01 00:00:00'},
        {'is_start': '2025-04-01 00:00:00', 'is_end': '2025-07-01 00:00:00', 'oos_end': '2025-08-01 00:00:00'},
        {'is_start': '2025-05-01 00:00:00', 'is_end': '2025-08-01 00:00:00', 'oos_end': '2025-09-01 00:00:00'},
    ]
    
    trials_per_optimization_walk = 200 
    number_of_clusters = 4           
    top_n_per_cluster = 3              

    main_logger.info("--- Starting Full Walk-Forward Ensemble Analysis (SHARED CAPITAL - MODEL C) ---")
    for i, walk_dates in enumerate(walk_forward_schedule):
        walk_num = i + 1
        run_ensemble_walk(
            walk_number=walk_num,
            is_start=walk_dates['is_start'],
            is_end=walk_dates['is_end'],
            oos_end=walk_dates['oos_end'],
            n_trials=trials_per_optimization_walk,
            n_clusters=number_of_clusters,
            top_n_per_cluster=top_n_per_cluster,
            opt_logger=main_logger
        )
    main_logger.info("\n--- All Walk-Forward Ensemble Analysis (SHARED) runs are complete. ---")

    if all_walks_data:
        results_df = pd.DataFrame(all_walks_data)
        results_filename = f"walk_forward_ensemble_SHARED_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        results_df.to_csv(results_filename, index=False)
        main_logger.info(f"\nSuccessfully saved all walk-forward ensemble (SHARED) data to: {results_filename}")