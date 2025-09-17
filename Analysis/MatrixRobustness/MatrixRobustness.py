import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from itertools import combinations
from matplotlib import gridspec

# ----------------------- Setup -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set(style="whitegrid")

BASE_PATH = 'Analysis/MatrixRobustness'
os.makedirs(BASE_PATH, exist_ok=True)
PLOTS_PATH = os.path.join(BASE_PATH, 'plots')
STATS_PATH = os.path.join(BASE_PATH, 'stats')
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(STATS_PATH, exist_ok=True)

# Excel files paths (sample)
EXCEL_FILES = [
    r'ThesisExperiments\test_matrix_type_robustness\results\amp_kamp_all_modes_20250909_185243.xlsx',
    r'ThesisExperiments\test_matrix_type_robustness\results\amp_kamp_all_modes_20250908_115543.xlsx'
]

METHOD_ORDER = ["AMP", "KAMP", "DKAMP"]

# ----------------------- Metric Order -----------------------
METRIC_ORDER = ["nmse", "mse", "rmse", "snr", "peak_snr"]  # Exclude fit_time for now

# ----------------------- IO & Cleaning -----------------------

def load_data():
    dfs = []
    for file in EXCEL_FILES:
        if not os.path.exists(file):
            logging.warning(f"File not found: {file}")
            continue
        df = pd.read_excel(file)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No Excel files loaded. Check EXCEL_FILES paths.")

    combined_df = pd.concat(dfs, ignore_index=True)

    combined_df.columns = [c.strip().lower() for c in combined_df.columns]
    if 'method' in combined_df.columns:
        combined_df['method'] = combined_df['method'].str.upper().str.strip()
    else:
        raise ValueError("Expected a 'method' column in the input data.")

    expected_cols = {'nmse', 'mse', 'rmse', 'snr', 'peak_snr', 'fit_time'}
    present = expected_cols.intersection(combined_df.columns)
    if 'nmse' not in present or 'fit_time' not in present:
        raise ValueError("Expected at least 'nmse' and 'fit_time' columns in the data.")

    combined_df = combined_df.dropna(subset=['nmse', 'fit_time'])
    if 'trial' not in combined_df.columns:
        combined_df['trial'] = combined_df.groupby(['method', 'sparsity_percent', 'matrix_type']).cumcount()

    combined_df['method'] = pd.Categorical(combined_df['method'],
                                           categories=[m for m in METHOD_ORDER if m in combined_df['method'].unique()],
                                           ordered=True)
    return combined_df

def remove_outliers(df,
                    metrics=METRIC_ORDER,
                    group_by=('method', 'sparsity_percent', 'matrix_type')):
    cleaned_parts = []
    for _, group in df.groupby(list(group_by), dropna=False):
        g = group.copy()
        for metric in metrics:
            if metric not in g.columns:
                continue
            q1, q3 = g[metric].quantile([0.25, 0.75])
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            g = g[(g[metric] >= low) & (g[metric] <= high)]
        cleaned_parts.append(g)
    cleaned_df = pd.concat(cleaned_parts, ignore_index=True)
    logging.info(f"Removed outliers; original shape: {df.shape}, cleaned shape: {cleaned_df.shape}")
    return cleaned_df

# ----------------------- Plot helpers -----------------------
def plot_line_trends(df):
    """
    Goal: Illustrate trends in performance metrics as sparsity increases, for effectiveness and robustness.
    Expected: Line plots with confidence intervals, showing how metrics change with sparsity for each method and matrix type.
    Handles: Multi-line plots with hue for methods and style for matrix types.
    Saves to plots subdirectory.
    """
    metrics = [m for m in METRIC_ORDER if m in df.columns]  # Include all required metrics
    methods = [m for m in METHOD_ORDER if m in df['method'].unique()]
    for metric in metrics:
        fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
        if len(methods) != 2:
            logging.warning(f"Expected 2 methods for subplots, found {len(methods)}. Skipping line trends for {metric}.")
            plt.close(fig); continue
        for i, method in enumerate(methods):
            subset = df[df['method'] == method]
            sns.lineplot(data=subset, x='sparsity_percent', y=metric, hue='matrix_type',
                         errorbar='sd', marker='o', ax=axs[i])
            axs[i].set_title(f'Trends in {metric.upper()} for {method}')
            axs[i].set_xlabel('Sparsity Percent')
            axs[i].set_ylabel(metric.upper())
            axs[i].legend(title='Matrix Type')
            axs[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, f'line_{metric}_trends_subplots.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    print("Line plots with subplots saved to plots/line_*_trends_subplots.png")

def _draw_diagonal_break(ax, where='bottom', d=0.008):
    if where == 'bottom':
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
        ax.plot((-d, +d), (-d, +d), **kwargs)  # left
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # right
    elif where == 'top':
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # left
        ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # right

def plot_metric_comparison(df, data_name, sparsity_percent, metric, plots_dir, timestamp, log_scale=False):
    if df.empty or 'matrix_type' not in df.columns or 'method' not in df.columns:
        logging.warning(f"Skipping metric comparison plot for {data_name}, sparsity {sparsity_percent}%, metric {metric}: DataFrame is empty or missing required columns")
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
        if log_scale:
            axs[i].set_yscale('log')
    plot_filename = f'{metric}_comparison_{data_name}_sparsity_{sparsity_percent}_{timestamp}'
    if log_scale:
        plot_filename += '_log'
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, plot_filename + '_subplots.png'), bbox_inches='tight')
    plt.close()
    logging.info(f"Saved {metric} comparison plot for {data_name}, sparsity {sparsity_percent}%")

def plot_bar_comparisons(df, log_scale=False):
    metrics = METRIC_ORDER  # Use the METRIC_ORDER defined above
    sparsity_levels = sorted(df['sparsity_percent'].unique())
    data_name = 'complex_signal'  # Assuming from data; adjust if needed
    timestamp = str(int(time.time()))
    
    for sparsity in sparsity_levels:
        subset = df[df['sparsity_percent'] == sparsity]
        for metric in metrics:
            if metric not in subset.columns: 
                continue
            plot_metric_comparison(subset, data_name, sparsity, metric, PLOTS_PATH, timestamp, log_scale)
    print("Bar plots saved to plots/*_comparison_*_subplots.png")

def plot_box_distributions(df, metrics=METRIC_ORDER, dynamic_break_threshold=5.0, log_scale=False):
    methods = [m for m in METHOD_ORDER if m in df['method'].unique()]
    if len(methods) != 2:
        logging.warning(f"Expected 2 methods for subplots, found {len(methods)}. Skipping box distributions.")
        return

    for metric in metrics:
        fig = plt.figure(figsize=(18, 9))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], hspace=0.05, wspace=0.18)
        axes_for_method = {}

        for j, method in enumerate(methods):
            subset = df[df['method'] == method]
            y = subset[metric].dropna().values
            if len(y) == 0:
                continue

            p1, p90, p95, p99 = np.percentile(y, [1, 90, 95, 99])
            ymax = np.max(y)
            needs_break = (p99 > 0) and (ymax / max(p99, 1e-12) > dynamic_break_threshold)

            if needs_break:
                ax_top = fig.add_subplot(gs[0, j])
                ax_bot = fig.add_subplot(gs[1, j], sharex=ax_top)

                sns.boxplot(data=subset, x='sparsity_percent', y=metric, hue='matrix_type', ax=ax_top)
                sns.boxplot(data=subset, x='sparsity_percent', y=metric, hue='matrix_type', ax=ax_bot)

                low_min = max(0.0, p1 - 0.1*(p95 - p1))
                low_max = p95 + 0.1*(p95 - p1)
                high_min = max(p99 * 0.98, low_max * 1.05)
                high_max = ymax * 1.02

                ax_bot.set_ylim(low_min, low_max)
                ax_top.set_ylim(high_min, high_max)

                ax_top.set_title(f'Distribution of {metric.upper()} for {method}')
                ax_bot.set_xlabel('Sparsity Percent')
                ax_bot.set_ylabel(metric.upper())
                ax_top.set_xlabel('')
                ax_top.set_xticklabels([])

                _draw_diagonal_break(ax_top, 'bottom')
                _draw_diagonal_break(ax_bot, 'top')

                if log_scale:
                    ax_top.set_yscale('log')
                    ax_bot.set_yscale('log')

                axes_for_method[method] = (ax_top, ax_bot)
            else:
                ax_main = fig.add_subplot(gs[:, j])
                sns.boxplot(data=subset, x='sparsity_percent', y=metric, hue='matrix_type', ax=ax_main)

                ax_main.set_title(f'Distribution of {metric.upper()} for {method}')
                ax_main.set_xlabel('Sparsity Percent')
                ax_main.set_ylabel(metric.upper())
                ax_main.legend(title='Matrix Type', ncol=3, fontsize=9)

                if log_scale:
                    ax_main.set_yscale('log')

                axes_for_method[method] = (ax_main,)

        plt.tight_layout()
        plot_filename = f'box_{metric}_distribution_subplots'
        if log_scale:
            plot_filename += '_log'
        plt.savefig(os.path.join(PLOTS_PATH, plot_filename + '.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    print("Box plots with (auto) broken y-axis saved to plots/box_*_distribution_subplots.png")

# ----------------------- Main -----------------------
if __name__ == '__main__':
    df = load_data()
    df_cleaned = remove_outliers(df)
    #plot_bar_comparisons(df_cleaned)
    #plot_bar_comparisons(df_cleaned, log_scale=True)  # log scale is enabled
    plot_box_distributions(df_cleaned, metrics=METRIC_ORDER, dynamic_break_threshold=5.0, log_scale=True)  # log scale is enabled
    #plot_line_trends(df_cleaned)
    print("All analyses and plots completed.")
