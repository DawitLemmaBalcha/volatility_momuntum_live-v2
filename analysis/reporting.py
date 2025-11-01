# reporting.py
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

def analyze_results(results_df, output_dir):
    """
    Analyzes the walk-forward optimization results and generates a report
    with statistics and visualizations.
    """
    print(f"--- Generating report in: {output_dir} ---")

    # 1. --- Statistical Summary ---
    stats_summary = results_df.describe()
    stats_summary.to_csv(os.path.join(output_dir, 'statistical_summary.csv'))
    print("  - Saved statistical_summary.csv")

    # Filter for only successful OOS trials for better visualization
    successful_trials = results_df.dropna(subset=['oos_return_pct'])

    # 2. --- Correlation Heatmap ---
    plt.figure(figsize=(16, 12))
    # Select only parameter and key OOS performance columns for clarity
    corr_df = successful_trials.filter(regex='param_|oos_')
    sns.heatmap(corr_df.corr(), annot=True, cmap='viridis', fmt='.2f')
    plt.title('Correlation Matrix of Parameters and OOS Performance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print("  - Saved correlation_heatmap.png")
    
    # 3. --- Parameter Distributions ---
    param_cols = [col for col in results_df.columns if 'param_' in col]
    fig, axes = plt.subplots(len(param_cols), 1, figsize=(10, 5 * len(param_cols)))
    fig.suptitle('Distribution of Parameter Values in Top Trials', fontsize=16)
    for i, param in enumerate(param_cols):
        sns.histplot(successful_trials[param], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {param}')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, 'parameter_distributions.png'))
    plt.close()
    print("  - Saved parameter_distributions.png")

    # 4. --- Parameter vs. Performance Scatter Plots (DYNAMICALLY GENERATED) ---
    # --- THIS IS THE FIX ---
    # Instead of a hardcoded list, we find all float-based parameter columns to plot.
    params_to_plot = [
        col for col in successful_trials.columns 
        if col.startswith('param_') and successful_trials[col].dtype == 'float64'
    ]
    
    if params_to_plot:
        fig, axes = plt.subplots(len(params_to_plot), 1, figsize=(10, 7 * len(params_to_plot)))
        # Handle case where there is only one plot
        if len(params_to_plot) == 1:
            axes = [axes]
        fig.suptitle('Key Parameters vs. OOS Return %', fontsize=16)
        for i, param in enumerate(params_to_plot):
            sns.scatterplot(data=successful_trials, x=param, y='oos_return_pct', hue='walk', palette='viridis', ax=axes[i])
            axes[i].grid(True)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(os.path.join(output_dir, 'param_vs_oos_return.png'))
        plt.close()
        print("  - Saved param_vs_oos_return.png")
    else:
        print("  - Skipped scatter plots: No float-based parameter columns found.")


    # 5. --- Walk-by-Walk OOS Performance ---
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=successful_trials, x='walk', y='oos_return_pct')
    plt.title('Out-of-Sample Return % Distribution per Walk')
    plt.xlabel('Walk Number')
    plt.ylabel('OOS Return %')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'walk_by_walk_performance.png'))
    plt.close()
    print("  - Saved walk_by_walk_performance.png")

    print("--- Report generation complete. ---")


if __name__ == '__main__':
    # Find the most recent results CSV file
    list_of_files = glob.glob('results/walk_forward_results_*.csv')
    if not list_of_files:
        print("Error: No 'walk_forward_results_*.csv' files found. Please run the optimizer first.")
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"--- Analyzing latest results file: {latest_file} ---")
        
        # Create a timestamped directory for the report
        report_dir = f"report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Load the data and generate the report
        main_df = pd.read_csv(latest_file)
        analyze_results(main_df, report_dir)
