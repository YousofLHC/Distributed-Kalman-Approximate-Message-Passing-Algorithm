import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================== 
# 1. SETUP & CONFIGURATION 
# ============================================================================== 

# Define file paths 
RESULTS_EXCEL_PATH = 'ThesisExperiments/DKAMP/results/distributed_kamp_all_modes_20250908_115532.xlsx'
ANALYSIS_OUTPUT_DIR = 'Analysis/Topology'

# Create output directory if it doesn't exist 
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# Plotting style configuration 
sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1) 
plt.rcParams['figure.figsize'] = (14, 8) 
plt.rcParams['figure.dpi'] = 100 
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================== 
# 2. DATA LOADING & PREPROCESSING 
# ============================================================================== 

def load_and_preprocess_data(excel_path): 
    """
    Loads the Excel file, preprocesses the data, and extracts the topology name.
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Error: The results file was not found at '{excel_path}'")

    print(f"Loading data from '{excel_path}'...")
    df = pd.read_excel(excel_path)
    print("Data loaded successfully.")

    # --- Extract Topology Name ---
    def extract_topology_name(filename):
        match = re.search(r'graph_\d+_(.*?)_adj\.txt', filename)
        if match:
            return match.group(1).replace('_', ' ')  # Replace underscores for better plot labels
        return "Unknown"

    df['topology_name'] = df['topology'].apply(extract_topology_name)
    print("Topology names extracted.")
    print(f"Found topologies: {df['topology_name'].unique().tolist()}")

    # Convert relevant columns to numeric, coercing errors
    numeric_cols = ['nmse_global', 'mse_global', 'rmse_global', 'snr_global', 'peak_snr_global', 
                    'mean_nmse_per_node', 'std_nmse_per_node', 'consensus_error', 'fit_time']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Removing outliers using Z-score
    df_no_outliers = remove_outliers(df, numeric_cols)

    return df_no_outliers


def remove_outliers(df, numeric_cols, threshold=3.0):
    """
    Removes outliers based on Z-score. 
    Any data point with a Z-score greater than the threshold is considered an outlier.
    """
    for col in numeric_cols:
        if col in df.columns:
            # Calculate the Z-score for the column
            z_scores = np.abs(stats.zscore(df[col].dropna()))  # Drop NaN values for Z-score calculation
            df = df[z_scores < threshold]  # Keep only the rows where Z-score is below the threshold
    print(f"Outliers removed based on Z-score (threshold: {threshold}).")
    return df

# ============================================================================== 
# 3. ANALYSIS & VISUALIZATION FUNCTIONS 
# ============================================================================== 

def plot_metric_comparison(df, data_type, metric, is_higher_better=False): 
    """
    Creates and saves a bar plot comparing a specific metric across all topologies,
    faceted by sparsity level.
    """
    print(f"Plotting comparison for metric: '{metric}'...")
    
    # Group by topology and sparsity, then calculate the mean of the metric
    grouped_data = df.groupby(['topology_name', 'sparsity_percent'])[metric].mean().reset_index()

    # Create the plot
    g = sns.catplot(
        data=grouped_data,
        x='topology_name',
        y=metric,
        col='sparsity_percent',
        kind='bar',
        height=6,
        aspect=1.5,
        palette='plasma',
        legend=False
    )

    # Customize plot aesthetics
    g.fig.suptitle(f'Comparison of {metric.upper()}  {data_type.replace("_", " ").title()}', y=1.03, fontsize=20)
    g.set_axis_labels("Topology", f"Average {metric.upper()}")
    g.set_titles("Sparsity = {col_name}%")
    g.set_xticklabels(rotation=45, ha='right')

    # Add value labels on top of bars
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=9)

    # Save the figure
    filename = f'{data_type}_comparison_{metric}.png'
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to '{output_path}'")

def perform_statistical_tests(df, metrics_to_rank): 
    """
    Performs statistical tests (e.g., Mann-Whitney U test or ANOVA) for each metric and reports results.
    """
    print("Performing statistical tests...")
    test_results = {}

    for metric in metrics_to_rank:
        # Perform Kruskal-Wallis test for non-parametric comparison between topologies
        grouped = df.groupby('topology_name')[metric].apply(list)  # Grouping by topology_name
        stat, p_value = stats.kruskal(*grouped)  # Using scipy's kruskal function
        test_results[metric] = {'statistic': stat, 'p_value': p_value}

        # Report the results of the statistical test
        print(f"Metric: {metric} - Kruskal-Wallis test statistic: {stat:.3f}, p-value: {p_value:.3f}")

    return test_results

def create_ranking_heatmap(df, data_type, metrics_to_rank): 
    """
    Creates a heatmap showing the rank of each topology for different metrics.
    Lower rank (e.g., 1) is better.
    """
    print("Creating ranking heatmap...")

    # Define which metrics are better when higher
    higher_is_better = ['snr_global', 'peak_snr_global']

    # Calculate mean values for each topology
    mean_df = df.groupby('topology_name')[metrics_to_rank].mean()

    # Calculate ranks
    rank_df = pd.DataFrame(index=mean_df.index)
    for metric in metrics_to_rank:
        ascending = metric not in higher_is_better
        rank_df[metric] = mean_df[metric].rank(ascending=ascending, method='min')

    # Perform statistical tests on the ranks
    test_results = perform_statistical_tests(df, metrics_to_rank)

    # Plot heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        rank_df,
        annot=True,
        cmap='YlGnBu',
        linewidths=.5,
        fmt='.0f',
        cbar_kws={'label': 'Performance Rank (1 is Best)'}
    )
    plt.title(f'Overall Performance Ranking ', fontsize=20, pad=20)
    plt.xlabel('Performance Metrics', fontsize=14)
    plt.ylabel('Topology', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Save the figure
    filename = f'{data_type}_performance_ranking_heatmap.png'
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap to '{output_path}'")

    return rank_df, test_results

def generate_summary_report(rank_df, test_results, data_type): 
    """
    Analyzes the ranking data and generates a text summary of the best topologies and statistical test results.
    """
    print("Generating summary report...")

    # Calculate average rank for each topology
    rank_df['average_rank'] = rank_df.mean(axis=1)
    best_overall_topology = rank_df['average_rank'].idxmin()

    summary = []
    summary.append("=" * 80)
    summary.append(f"TOPOLOGY PERFORMANCE ANALYSIS REPORT - Data Type: {data_type.replace('_', ' ').title()}")
    summary.append("=" * 80)
    summary.append("\n--- Overall Best Performing Topology ---")
    summary.append(f"Based on the average rank across all metrics, the best topology is: '{best_overall_topology}'")
    summary.append(f"with an average rank of {rank_df.loc[best_overall_topology, 'average_rank']:.2f}.\n")

    summary.append("--- Best Topology per Metric ---")
    for metric in rank_df.columns:
        if metric != 'average_rank':
            best_topology_for_metric = rank_df[metric].idxmin()
            summary.append(f"'{metric}': '{best_topology_for_metric}'")

    summary.append("\n--- Statistical Test Results ---")
    for metric, result in test_results.items():
        summary.append(f"Metric '{metric}' - Kruskal-Wallis Test: Statistic = {result['statistic']:.3f}, p-value = {result['p_value']:.3f}")

    summary.append("\n--- Detailed Ranking Table ---")
    summary.append(rank_df.sort_values('average_rank').to_string())
    summary.append("\n" + "=" * 80)

    report_str = "\n".join(summary)
    
    # Save the report to a text file
    filename = f'{data_type}_analysis_report.txt'
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, filename)
    with open(output_path, 'w') as f:
        f.write(report_str)
        
    print(f"Saved analysis report to '{output_path}'")
    return report_str


# ============================================================================== 
# 4. MAIN EXECUTION 
# ==============================================================================

def main(): 
    """
    Main function to run the complete analysis pipeline.
    """ 
    try: 
        # Load and preprocess data
        full_df = load_and_preprocess_data(RESULTS_EXCEL_PATH)

        # Case 1: Both 2d_signal and complex together
        df_combined = full_df[full_df['data_type'].isin(['2d_signal', 'complex'])].copy()
        if not df_combined.empty:
            print(f"Analyzing data for '2d_signal and complex' together...")
            metrics_to_analyze = ['nmse_global', 'rmse_global', 'snr_global', 'peak_snr_global', 'mse_global']  # including mse_global for combined analysis
            for metric in metrics_to_analyze:
                if metric in df_combined.columns and not df_combined[metric].isnull().all():
                    plot_metric_comparison(df_combined, '2d_signal_and_complex', metric)

            rank_df_combined, test_results_combined = create_ranking_heatmap(df_combined, '2d_signal_and_complex', metrics_to_analyze)
            generate_summary_report(rank_df_combined, test_results_combined, '2d_signal_and_complex')

        # Case 2: For complex, check mse_global
        df_complex = full_df[full_df['data_type'] == 'complex'].copy()
        if not df_complex.empty:
            print(f"Analyzing data for 'complex'...")
            metrics_to_analyze = ['mse_global']  # Only mse_global for complex data
            for metric in metrics_to_analyze:
                if metric in df_complex.columns and not df_complex[metric].isnull().all():
                    plot_metric_comparison(df_complex, 'complex', metric)

            rank_df_complex, test_results_complex = create_ranking_heatmap(df_complex, 'complex', metrics_to_analyze)
            generate_summary_report(rank_df_complex, test_results_complex, 'complex')

        print("\nAnalysis complete. All plots and reports have been saved to the 'Analysis/Topology' directory.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__": 
    main()
