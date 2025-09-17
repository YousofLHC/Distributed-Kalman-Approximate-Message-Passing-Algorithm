import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import pingouin as pg  # For advanced stats if needed
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
BASE_PATH = 'Analysis/MatrixRobustness'
os.makedirs(BASE_PATH, exist_ok=True)
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')
STATS_PATH = os.path.join(BASE_PATH, 'stats')
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(STATS_PATH, exist_ok=True)

# Excel files paths
EXCEL_FILES = [
    r'ThesisExperiments\test_matrix_type_robustness\results\amp_kamp_all_modes_20250909_185243.xlsx',
    r'ThesisExperiments\test_matrix_type_robustness\results\amp_kamp_all_modes_20250908_115543.xlsx'
]

def load_data():
    """
    Goal: Load and combine data from multiple Excel files for analysis.
    Expected: A single DataFrame containing all experimental results, cleaned and ready for analysis.
    Handles: Combining sheets if needed, dropping unnecessary columns, handling missing values.
    """
    dfs = []
    for file in EXCEL_FILES:
        df = pd.read_excel(file)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    # Clean data: Assume methods are AMP and KAMP; if DKAMP data is added later, it can be included
    combined_df = combined_df.dropna(subset=['nmse', 'fit_time'])  # Drop rows with missing key metrics
    combined_df['method'] = combined_df['method'].str.upper()  # Standardize method names
    # Note: If DKAMP data is in separate files, load and append here
    return combined_df

def remove_outliers(df, metrics=['nmse', 'snr', 'fit_time'], group_by=['method', 'sparsity_percent', 'matrix_type']):
    """
    Goal: Detect and remove outliers using IQR method for specified metrics, grouped by categories.
    Expected: A cleaned DataFrame with outliers removed per group and metric.
    Handles: Statistical outlier detection to ensure robust analysis and plotting.
    """
    cleaned_df = pd.DataFrame()
    for name, group in df.groupby(group_by):
        for metric in metrics:
            Q1 = group[metric].quantile(0.25)
            Q3 = group[metric].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            group = group[(group[metric] >= lower_bound) & (group[metric] <= upper_bound)]
        cleaned_df = pd.concat([cleaned_df, group])
    logging.info(f"Removed outliers; original shape: {df.shape}, cleaned shape: {cleaned_df.shape}")
    return cleaned_df

def descriptive_statistics(df):
    """
    Goal: Compute descriptive statistics for robustness, effectiveness, and efficiency across methods, sparsity, and matrix types.
    Expected: Summary tables (mean, std, min, max) for metrics like NMSE, SNR, fit_time, grouped by method, sparsity_percent, matrix_type.
    Handles: Saving stats to CSV files in stats subdirectory.
    """
    metrics = ['nmse', 'mse', 'rmse', 'snr', 'peak_snr', 'fit_time']
    summary = df.groupby(['method', 'sparsity_percent', 'matrix_type'])[metrics].agg(['mean', 'std', 'min', 'max'])
    summary.to_csv(os.path.join(STATS_PATH, 'descriptive_stats.csv'))
    print("Descriptive statistics saved to stats/descriptive_stats.csv")
    return summary

def statistical_tests(df):
    """
    Goal: Perform statistical tests to compare methods (AMP, KAMP, DKAMP) for significance in differences.
    Expected: p-values from t-tests (pairwise between methods), ANOVA (across groups), and post-hoc tests.
    Handles: Robustness (variance comparison via Levene's test), effectiveness (NMSE/SNR differences), efficiency (fit_time).
    Saves results to CSV.
    """
    results = {}
    
    # Pairwise t-tests for NMSE (effectiveness)
    methods = df['method'].unique()
    for metric in ['nmse', 'fit_time']:
        for m1, m2 in zip(methods, methods[1:]):
            group1 = df[df['method'] == m1][metric]
            group2 = df[df['method'] == m2][metric]
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's t-test for unequal variance
            results[f'{metric}_ttest_{m1}_vs_{m2}'] = {'t_stat': t_stat, 'p_val': p_val}
    
    # ANOVA for NMSE across sparsity and methods (robustness to sparsity)
    try:
        anova_df = pg.rm_anova(dv='nmse', within=['sparsity_percent', 'method'], subject='trial', data=df)
        anova_df.to_csv(os.path.join(STATS_PATH, 'anova_nmse.csv'))
    except Exception as e:
        logging.warning(f"ANOVA failed: {e}")
    
    # Levene's test for variance (robustness: lower variance means more robust)
    for metric in ['nmse', 'fit_time']:
        for key, group in df.groupby(['sparsity_percent', 'matrix_type']):
            groups = [sub[metric] for name, sub in group.groupby('method')]
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                levene_stat, levene_p = stats.levene(*groups)
                results[f'levene_{key}_{metric}'] = {'stat': levene_stat, 'p_val': levene_p}
    
    stats_df = pd.DataFrame(results).T
    stats_df.to_csv(os.path.join(STATS_PATH, 'statistical_tests.csv'))
    print("Statistical tests saved to stats/statistical_tests.csv")
    return stats_df

def plot_metric_comparison(df, data_name, sparsity_percent, metric, plots_dir, timestamp):
    """Plot comparison of a metric across matrix types and methods."""
    if df.empty or 'matrix_type' not in df.columns or 'method' not in df.columns:
        logging.warning(f"Skipping metric comparison plot for {data_name}, sparsity {sparsity_percent}%, metric {metric}: DataFrame is empty or missing required columns")
        print(f"Warning: Skipping metric comparison plot for {data_name}, sparsity {sparsity_percent}%, metric {metric}: DataFrame is empty or missing required columns")
        return
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
    matrix_types = df['matrix_type'].unique()
    methods = df['method'].unique()
    if len(methods) != 2:
        logging.warning(f"Expected 2 methods for subplots, found {len(methods)}. Skipping bar comparison for {metric} at sparsity {sparsity_percent}%.")
        return
    for i, method in enumerate(methods):
        metric_values = []
        for matrix_type in matrix_types:
            subset = df[(df['method'] == method) & (df['matrix_type'] == matrix_type)]
            metric_values.append(subset[metric].mean() if not subset.empty else np.nan)
        axs[i].bar(matrix_types, metric_values, alpha=0.8)
        axs[i].set_xlabel('Matrix Type')
        axs[i].set_ylabel(metric)
        axs[i].set_title(f'{metric.upper()} for {method}, Sparsity {sparsity_percent}%')
        axs[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{metric}_comparison_{data_name}_sparsity_{sparsity_percent}_{timestamp}_subplots.png'), bbox_inches='tight')
    plt.close()
    logging.info(f"Saved {metric} comparison plot for {data_name}, sparsity {sparsity_percent}%")

def plot_bar_comparisons(df):
    """
    Goal: Visualize mean metrics (NMSE, SNR, fit_time) as bar plots for comparing effectiveness and efficiency across methods and matrix types.
    Expected: Bar plots showing means with error bars (std) for each sparsity level.
    Handles: Grouped bars for methods (AMP, KAMP, DKAMP), faceted by sparsity_percent.
    Saves plots to plots subdirectory.
    """
    metrics = ['nmse', 'snr', 'fit_time']
    sparsity_levels = df['sparsity_percent'].unique()
    data_name = 'complex_signal'  # Assuming from data; adjust if needed
    timestamp = str(int(time.time()))
    
    for sparsity in sparsity_levels:
        subset = df[df['sparsity_percent'] == sparsity]
        for metric in metrics:
            plot_metric_comparison(subset, data_name, sparsity, metric, PLOTS_PATH, timestamp)
    print("Bar plots saved to plots/*_comparison_*_subplots.png")

def plot_box_distributions(df):
    """
    Goal: Show distributions of metrics to assess robustness (spread/variance) across trials.
    Expected: Box plots for NMSE, fit_time, etc., grouped by method and sparsity.
    Handles: Outliers and variance visualization for robustness evaluation.
    Saves to plots subdirectory.
    """
    metrics = ['nmse', 'snr', 'fit_time']
    for metric in metrics:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
        methods = df['method'].unique()
        if len(methods) != 2:
            logging.warning(f"Expected 2 methods for subplots, found {len(methods)}. Skipping box distribution for {metric}.")
            continue
        for i, method in enumerate(methods):
            subset = df[df['method'] == method]
            sns.boxplot(data=subset, x='sparsity_percent', y=metric, hue='matrix_type', ax=axs[i])
            axs[i].set_title(f'Distribution of {metric.upper()} for {method}')
            axs[i].set_xlabel('Sparsity Percent')
            axs[i].set_ylabel(metric.upper())
            axs[i].legend(title='Matrix Type')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, f'box_{metric}_distribution_subplots.png'))
        plt.close()
    print("Box plots with subplots saved to plots/box_*_distribution_subplots.png")

def plot_line_trends(df):
    """
    Goal: Illustrate trends in performance metrics as sparsity increases, for effectiveness and robustness.
    Expected: Line plots with confidence intervals, showing how metrics change with sparsity for each method and matrix type.
    Handles: Multi-line plots with hue for methods and style for matrix types.
    Saves to plots subdirectory.
    """
    metrics = ['nmse', 'snr', 'fit_time']
    for metric in metrics:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
        methods = df['method'].unique()
        if len(methods) != 2:
            logging.warning(f"Expected 2 methods for subplots, found {len(methods)}. Skipping line trends for {metric}.")
            continue
        for i, method in enumerate(methods):
            subset = df[df['method'] == method]
            sns.lineplot(data=subset, x='sparsity_percent', y=metric, hue='matrix_type', errorbar='sd', ax=axs[i])
            axs[i].set_title(f'Trends in {metric.upper()} for {method}')
            axs[i].set_xlabel('Sparsity Percent')
            axs[i].set_ylabel(metric.upper())
            axs[i].legend(title='Matrix Type')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, f'line_{metric}_trends_subplots.png'))
        plt.close()
    print("Line plots with subplots saved to plots/line_*_trends_subplots.png")

def plot_heatmaps(df):
    """
    Goal: Provide a compact overview of mean metrics across all combinations for quick comparison.
    Expected: Heatmaps of mean NMSE, fit_time, etc., with rows as sparsity/matrix and columns as methods.
    Handles: Color-coded visualization for identifying patterns in robustness and efficiency.
    Saves to plots subdirectory.
    """
    metrics = ['nmse', 'fit_time']
    for metric in metrics:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        methods = df['method'].unique()
        if len(methods) != 2:
            logging.warning(f"Expected 2 methods for subplots, found {len(methods)}. Skipping heatmap for {metric}.")
            continue
        for i, method in enumerate(methods):
            subset = df[df['method'] == method]
            pivot = subset.pivot_table(index=['sparsity_percent', 'matrix_type'], values=metric, aggfunc='mean')
            sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.4f', ax=axs[i])
            axs[i].set_title(f'Heatmap of Mean {metric.upper()} for {method}')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, f'heatmap_mean_{metric}_subplots.png'))
        plt.close()
    print("Heatmaps with subplots saved to plots/heatmap_mean_*_subplots.png")

def plot_scatter_tradeoffs(df):
    """
    Goal: Explore trade-offs between effectiveness (NMSE) and efficiency (fit_time).
    Expected: Scatter plots showing NMSE vs. fit_time, colored by method, sized by sparsity.
    Handles: Visualizing if better performance comes at higher computational cost.
    Saves to plots subdirectory.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
    methods = df['method'].unique()
    if len(methods) != 2:
        logging.warning(f"Expected 2 methods for subplots, found {len(methods)}. Skipping scatter tradeoffs.")
        return
    for i, method in enumerate(methods):
        subset = df[df['method'] == method]
        sns.scatterplot(data=subset, x='fit_time', y='nmse', hue='matrix_type', size='sparsity_percent', ax=axs[i])
        axs[i].set_title(f'Trade-off: NMSE vs. Fit Time for {method}')
        axs[i].set_xlabel('Fit Time (seconds)')
        axs[i].set_ylabel('NMSE')
        axs[i].legend(title='Matrix / Sparsity')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'scatter_nmse_vs_time_subplots.png'))
    plt.close()
    print("Scatter plot with subplots saved to plots/scatter_nmse_vs_time_subplots.png")

# Main execution
if __name__ == '__main__':
    df = load_data()
    df_cleaned = remove_outliers(df)
    descriptive_statistics(df_cleaned)
    statistical_tests(df_cleaned)
    plot_bar_comparisons(df_cleaned)
    plot_box_distributions(df_cleaned)
    plot_line_trends(df_cleaned)
    plot_heatmaps(df_cleaned)
    plot_scatter_tradeoffs(df_cleaned)
    print("All analyses and plots completed.")