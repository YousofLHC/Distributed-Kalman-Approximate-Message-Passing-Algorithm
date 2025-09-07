import numpy as np
import warnings
from typing import Literal
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef
)

# Changelog: Added principled zero-division handling with safe_divide function,
# avoiding +1 trick. Consistent with scikit-learn's zero_division parameter.

ZeroDivMode = Literal["epsilon", "zero", "nan", "raise"]
DEFAULT_ZERO_DIVISION_CLASSIF: ZeroDivMode = "zero"
DEFAULT_ZERO_DIVISION_RECON: ZeroDivMode = "nan"
DEFAULT_EPS = 1e-12


def safe_divide(
    num: np.ndarray | float,
    den: np.ndarray | float,
    *,
    mode: ZeroDivMode = "epsilon",
    epsilon: float = DEFAULT_EPS,
    where: np.ndarray | None = None,
    warn: bool = True
) -> np.ndarray | float:
    """
    Numerically stable division for metric computation.

    This function provides principled handling of division-by-zero in evaluation metrics,
    avoiding the statistically unjustified +1 trick used in some implementations.
    Instead, it follows common practices similar to scikit-learn's zero_division parameter,
    with additional epsilon-smoothing for numerical stability.

    Modes:
      - "epsilon": return num / (den + epsilon)   # preferred default for stability
      - "zero":    return 0.0 where den==0
      - "nan":     return np.nan where den==0
      - "raise":   raise ZeroDivisionError for any den==0

    If `where` is provided, apply mode only on entries where `den==0 & where`.
    When `warn=True` and any den==0 occurs (except mode='raise'), log a one-time warning.

    Parameters
    ----------
    num : np.ndarray | float
        Numerator.
    den : np.ndarray | float
        Denominator.
    mode : ZeroDivMode, default="epsilon"
        How to handle den==0.
    epsilon : float, default=1e-12
        Small value for epsilon mode.
    where : np.ndarray | None, default=None
        Optional mask for conditional application.
    warn : bool, default=True
        Whether to warn on zero denominators.

    Returns
    -------
    np.ndarray | float
        Result of division with zero-handling.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)

    zero_mask = den == 0
    if where is not None:
        zero_mask = zero_mask & where

    if np.any(zero_mask):
        if warn:
            warnings.warn(
                f"Zero denominator encountered in safe_divide. "
                f"Resolved using mode='{mode}'. "
                f"Consider using epsilon-smoothing for numerical stability.",
                UserWarning,
                stacklevel=2
            )
        if mode == "raise":
            raise ZeroDivisionError("Division by zero encountered.")
        elif mode == "zero":
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(zero_mask, 0.0, num / den)
        elif mode == "nan":
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(zero_mask, np.nan, num / den)
        elif mode == "epsilon":
            den_smooth = np.where(zero_mask, epsilon, den)
            result = num / den_smooth
        else:
            raise ValueError(f"Invalid mode: {mode}")
    else:
        result = num / den

    return result.item() if result.ndim == 0 else result

def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Calculate AUC-based evaluation metrics for anomaly detection.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_score : np.ndarray
        Predicted scores.

    Returns
    -------
    dict
        Dictionary containing ROC AUC and Precision-Recall AUC.
    """
    auc_roc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = np.trapezoid(precision, recall)
    return {'auc_roc': auc_roc, 'auc_pr': auc_pr}


def calculate_anomaly_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate anomaly detection metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.

    Returns
    -------
    dict
        Dictionary containing precision, recall, specificity, f1, kappa, mcc, balanced_accuracy, f2, g_mean.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Specificity (TNR)
    specificity = safe_divide(tn, tn + fp, mode=DEFAULT_ZERO_DIVISION_CLASSIF)

    # Additional rates
    fpr = safe_divide(fp, fp + tn, mode=DEFAULT_ZERO_DIVISION_CLASSIF)  # False Positive Rate
    fnr = safe_divide(fn, fn + tp, mode=DEFAULT_ZERO_DIVISION_CLASSIF)  # False Negative Rate
    fdr = safe_divide(fp, fp + tp, mode=DEFAULT_ZERO_DIVISION_CLASSIF)  # False Discovery Rate
    npv = safe_divide(tn, tn + fn, mode=DEFAULT_ZERO_DIVISION_CLASSIF)  # Negative Predictive Value
    for_ = safe_divide(fn, fn + tn, mode=DEFAULT_ZERO_DIVISION_CLASSIF)  # False Omission Rate

    # F2 score
    if precision == 0.0 and recall == 0.0:
        f2 = 0.0
    else:
        f2 = safe_divide(5 * precision * recall, 4 * precision + recall, mode=DEFAULT_ZERO_DIVISION_CLASSIF)

    # G-Mean
    g_mean = np.sqrt(recall * specificity)

    # Youden's J (Bookmaker Informedness)
    youdens_j = recall + specificity - 1

    # Markedness
    markedness = precision + npv - 1

    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'kappa': kappa,
        'mcc': mcc,
        'balanced_accuracy': balanced_acc,
        'f2': f2,
        'g_mean': g_mean,
        'fpr': fpr,
        'fnr': fnr,
        'fdr': fdr,
        'for': for_,
        'npv': npv,
        'youdens_j': youdens_j,
        'markedness': markedness
    }


def calculate_compressive_sensing_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate compressive sensing metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True signal.
    y_pred : np.ndarray
        Predicted signal.

    Returns
    -------
    dict
        Dictionary containing nmse, peak_snr.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    nmse = safe_divide(mse, np.var(y_true), mode=DEFAULT_ZERO_DIVISION_RECON)

    # SNR: Signal-to-Noise Ratio
    if mse == 0:
        snr = float('inf')
    else:
        mse_clamped = max(mse, DEFAULT_EPS)
        snr = 10 * np.log10(safe_divide(np.var(y_true), mse_clamped, mode=DEFAULT_ZERO_DIVISION_RECON))

    # Peak SNR (PSNR)
    if mse == 0:
        peak_snr = float('inf')
    else:
        mse_clamped = max(mse, DEFAULT_EPS)
        peak_snr = 20 * np.log10(safe_divide(np.max(np.abs(y_true))**2, mse_clamped, mode=DEFAULT_ZERO_DIVISION_RECON))

    return {'mse': mse, 'rmse': rmse, 'nmse': nmse, 'snr': snr, 'peak_snr': peak_snr}


def calculate_gmsd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Gradient Magnitude Similarity Deviation (GMSD).

    Parameters
    ----------
    y_true : np.ndarray
        True image.
    y_pred : np.ndarray
        Predicted image.

    Returns
    -------
    float
        GMSD value.
    """
    from scipy.ndimage import sobel

    # Compute gradients
    grad_true_x = sobel(y_true, axis=0)
    grad_true_y = sobel(y_true, axis=1)
    grad_pred_x = sobel(y_pred, axis=0)
    grad_pred_y = sobel(y_pred, axis=1)

    grad_true = np.sqrt(grad_true_x**2 + grad_true_y**2)
    grad_pred = np.sqrt(grad_pred_x**2 + grad_pred_y**2)

    # Compute quality map
    quality_map = (2 * grad_true * grad_pred + 0.01) / (grad_true**2 + grad_pred**2 + 0.01)

    # GMSD
    gmsd = np.std(quality_map)

    return gmsd


def calculate_fsim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Feature Similarity Index (FSIM).

    Parameters
    ----------
    y_true : np.ndarray
        True image.
    y_pred : np.ndarray
        Predicted image.

    Returns
    -------
    float
        FSIM value.
    """
    # Simplified FSIM implementation
    from scipy.ndimage import sobel, gaussian_filter

    # Compute phase congruency (simplified)
    pc_true = np.abs(sobel(y_true, axis=0)) + np.abs(sobel(y_true, axis=1))
    pc_pred = np.abs(sobel(y_pred, axis=0)) + np.abs(sobel(y_pred, axis=1))

    # Compute similarity
    pc_sim = (2 * pc_true * pc_pred + 0.01) / (pc_true**2 + pc_pred**2 + 0.01)

    # Gradient magnitude
    gm_true = np.sqrt(sobel(y_true, axis=0)**2 + sobel(y_true, axis=1)**2)
    gm_pred = np.sqrt(sobel(y_pred, axis=0)**2 + sobel(y_pred, axis=1)**2)
    gm_sim = (2 * gm_true * gm_pred + 0.01) / (gm_true**2 + gm_pred**2 + 0.01)

    # Combine
    sim = pc_sim * gm_sim
    fsim = np.mean(sim)

    return fsim


def calculate_vif(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Visual Information Fidelity (VIF).

    Parameters
    ----------
    y_true : np.ndarray
        True image.
    y_pred : np.ndarray
        Predicted image.

    Returns
    -------
    float
        VIF value.
    """
    # Simplified VIF implementation
    from scipy.ndimage import gaussian_filter

    # Compute mutual information (simplified)
    sigma = 1.0
    mu_true = gaussian_filter(y_true, sigma)
    mu_pred = gaussian_filter(y_pred, sigma)

    sigma_true = np.sqrt(gaussian_filter(y_true**2, sigma) - mu_true**2)
    sigma_pred = np.sqrt(gaussian_filter(y_pred**2, sigma) - mu_pred**2)
    sigma_tp = gaussian_filter(y_true * y_pred, sigma) - mu_true * mu_pred

    vif = np.mean((sigma_tp**2 + 0.01) / (sigma_true**2 * sigma_pred**2 + 0.01))

    return vif


def calculate_ssim(y_true: np.ndarray, y_pred: np.ndarray, data_range: float = None) -> float:
    """
    Calculate Structural Similarity Index Measure (SSIM).

    Parameters
    ----------
    y_true : np.ndarray
        True image.
    y_pred : np.ndarray
        Predicted image.
    data_range : float, optional
        Data range of the input image.

    Returns
    -------
    float
        SSIM value.
    """
    if data_range is None:
        data_range = np.max(y_true) - np.min(y_true)

    # Constants
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Means
    mu_true = np.mean(y_true)
    mu_pred = np.mean(y_pred)

    # Variances and covariance
    sigma_true = np.var(y_true)
    sigma_pred = np.var(y_pred)
    sigma_tp = np.cov(y_true.flatten(), y_pred.flatten())[0, 1]

    # SSIM
    numerator = (2 * mu_true * mu_pred + C1) * (2 * sigma_tp + C2)
    denominator = (mu_true**2 + mu_pred**2 + C1) * (sigma_true + sigma_pred + C2)

    ssim = numerator / denominator
    return ssim


def calculate_phase_transition(successes: list, sparsities: list, measurement_rates: list) -> dict:
    """
    Calculate phase transition data for compressive sensing.

    Parameters
    ----------
    successes : list
        List of success indicators (0 or 1).
    sparsities : list
        List of sparsity levels.
    measurement_rates : list
        List of measurement rates.

    Returns
    -------
    dict
        Dictionary with phase transition data.
    """
    # Group by parameters and compute success probability
    from collections import defaultdict

    data = defaultdict(list)
    for s, sp, mr in zip(successes, sparsities, measurement_rates):
        data[(sp, mr)].append(s)

    phase_data = {}
    for (sp, mr), succ_list in data.items():
        success_prob = np.mean(succ_list)
        phase_data[(sp, mr)] = success_prob

    return phase_data


def track_convergence(iterations: int, runtime: float, tolerance: float = None, error_history: list = None) -> dict:
    """
    Track convergence metrics.

    Parameters
    ----------
    iterations : int
        Number of iterations to converge.
    runtime : float
        Runtime in seconds.
    tolerance : float, optional
        Convergence tolerance used.
    error_history : list, optional
        List of error values over iterations.

    Returns
    -------
    dict
        Dictionary with convergence metrics.
    """
    metrics = {'iterations': iterations, 'runtime': runtime}

    if tolerance is not None:
        metrics['tolerance'] = tolerance

    if error_history is not None:
        metrics['final_error'] = error_history[-1] if error_history else None
        metrics['error_history'] = error_history
        if len(error_history) > 1:
            metrics['convergence_rate'] = np.mean(np.diff(error_history))

    return metrics


def log_results(results, filepath='results/logs/experiment_log.txt'):
    import time
    import os
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(f"Timestamp: {time.ctime()}\n")
        for method, metrics in results.items():
            f.write(f"{method}: AUC-ROC = {metrics['auc_roc']:.4f}, Time = {metrics['time']:.2f}s\n")
        f.write("-" * 50 + "\n")


def save_detailed_report(results, filepath='results/detailed_report.txt', format_type='txt'):
    """
    Save detailed comprehensive report of experimental results.

    Parameters
    ----------
    results : list
        List of result dictionaries from experiments
    filepath : str
        Path to save the report
    format_type : str
        Format type: 'txt', 'csv', or 'json'
    """
    import time
    import os
    import json
    import csv

    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if format_type == 'txt':
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DISTRIBUTED KALMAN APPROXIMATE MESSAGE PASSING - DETAILED REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {time.ctime()}\n")
            f.write(f"Total Experiments: {len(results)}\n\n")

            # Summary statistics
            if results:
                data_types = set(r.get('data_type', 'Unknown') for r in results)
                f.write("DATA TYPES PROCESSED:\n")
                for dt in data_types:
                    count = sum(1 for r in results if r.get('data_type') == dt)
                    f.write(f"  - {dt}: {count} experiments\n")
                f.write("\n")

                # Performance summary
                nmse_values = [r['nmse_global'] for r in results if 'nmse_global' in r]
                if nmse_values:
                    f.write("PERFORMANCE SUMMARY:\n")
                    f.write(f"  - Average NMSE: {np.mean(nmse_values):.6f}\n")
                    f.write(f"  - Min NMSE: {np.min(nmse_values):.6f}\n")
                    f.write(f"  - Max NMSE: {np.max(nmse_values):.6f}\n")
                    f.write(f"  - Std NMSE: {np.std(nmse_values):.6f}\n")
                    f.write("\n")

                consensus_errors = [r['consensus_error'] for r in results if 'consensus_error' in r]
                if consensus_errors:
                    f.write("CONSENSUS ANALYSIS:\n")
                    f.write(f"  - Average Consensus Error: {np.mean(consensus_errors):.6f}\n")
                    f.write(f"  - Min Consensus Error: {np.min(consensus_errors):.6f}\n")
                    f.write(f"  - Max Consensus Error: {np.max(consensus_errors):.6f}\n")
                    f.write("\n")

                fit_times = [r['fit_time'] for r in results if 'fit_time' in r]
                if fit_times:
                    f.write("TIMING ANALYSIS:\n")
                    f.write(f"  - Average Fit Time: {np.mean(fit_times):.4f}s\n")
                    f.write(f"  - Total Time: {np.sum(fit_times):.4f}s\n")
                    f.write(f"  - Min Fit Time: {np.min(fit_times):.4f}s\n")
                    f.write(f"  - Max Fit Time: {np.max(fit_times):.4f}s\n")
                    f.write("\n")

            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 80 + "\n")

            for i, result in enumerate(results, 1):
                f.write(f"\nExperiment {i}:\n")
                f.write(f"  Data Type: {result.get('data_type', 'N/A')}\n")
                f.write(f"  Graph File: {result.get('graph_file', 'N/A')}\n")
                f.write(f"  Nodes: {result.get('num_nodes', 'N/A')}\n")
                f.write(f"  Edges: {result.get('num_edges', 'N/A')}\n")
                f.write(f"  Density: {result.get('density', 'N/A'):.4f}\n")
                f.write(f"  Is DAG: {result.get('is_dag', 'N/A')}\n")
                f.write(f"  Global NMSE: {result.get('nmse_global', 'N/A'):.6f}\n")
                f.write(f"  Mean NMSE per Node: {result.get('mean_nmse_per_node', 'N/A'):.6f}\n")
                f.write(f"  Std NMSE per Node: {result.get('std_nmse_per_node', 'N/A'):.6f}\n")
                f.write(f"  Consensus Error: {result.get('consensus_error', 'N/A'):.6f}\n")
                f.write(f"  Bytes Transferred: {result.get('bytes', 'N/A')}\n")
                f.write(f"  Iterations: {result.get('iters', 'N/A')}\n")
                f.write(f"  Fit Time: {result.get('fit_time', 'N/A'):.4f}s\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("REPORT COMPLETE\n")
            f.write("=" * 80 + "\n")

    elif format_type == 'csv':
        if results:
            fieldnames = ['experiment_id', 'data_type', 'graph_file', 'num_nodes', 'num_edges',
                         'density', 'is_dag', 'nmse_global', 'mean_nmse_per_node', 'std_nmse_per_node',
                         'consensus_error', 'bytes', 'iters', 'fit_time']

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for i, result in enumerate(results, 1):
                    row = {'experiment_id': i}
                    row.update({k: result.get(k, 'N/A') for k in fieldnames[1:]})
                    writer.writerow(row)

    elif format_type == 'json':
        # Add metadata
        report_data = {
            'metadata': {
                'generated_at': time.ctime(),
                'total_experiments': len(results),
                'format_version': '1.0'
            },
            'summary': {},
            'results': results
        }

        # Add summary statistics
        if results:
            nmse_values = [r['nmse_global'] for r in results if 'nmse_global' in r]
            if nmse_values:
                report_data['summary']['nmse'] = {
                    'mean': float(np.mean(nmse_values)),
                    'min': float(np.min(nmse_values)),
                    'max': float(np.max(nmse_values)),
                    'std': float(np.std(nmse_values))
                }

            consensus_errors = [r['consensus_error'] for r in results if 'consensus_error' in r]
            if consensus_errors:
                report_data['summary']['consensus_error'] = {
                    'mean': float(np.mean(consensus_errors)),
                    'min': float(np.min(consensus_errors)),
                    'max': float(np.max(consensus_errors))
                }

            fit_times = [r['fit_time'] for r in results if 'fit_time' in r]
            if fit_times:
                report_data['summary']['timing'] = {
                    'mean_fit_time': float(np.mean(fit_times)),
                    'total_time': float(np.sum(fit_times)),
                    'min_fit_time': float(np.min(fit_times)),
                    'max_fit_time': float(np.max(fit_times))
                }

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)

    print(f"Detailed report saved to {filepath}")


def save_experiment_summary(results, filepath='results/experiment_summary.txt'):
    """
    Save a concise summary of experimental results.

    Parameters
    ----------
    results : list
        List of result dictionaries from experiments
    filepath : str
        Path to save the summary
    """
    import time
    import os

    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {time.ctime()}\n")
        f.write(f"Total Experiments: {len(results)}\n\n")

        if results:
            # Group by data type
            from collections import defaultdict
            by_data_type = defaultdict(list)
            for r in results:
                by_data_type[r.get('data_type', 'Unknown')].append(r)

            for data_type, type_results in by_data_type.items():
                f.write(f"Data Type: {data_type}\n")
                f.write(f"  Experiments: {len(type_results)}\n")

                nmse_vals = [r['nmse_global'] for r in type_results if 'nmse_global' in r]
                if nmse_vals:
                    f.write(f"  Avg NMSE: {np.mean(nmse_vals):.6f} ± {np.std(nmse_vals):.6f}\n")

                times = [r['fit_time'] for r in type_results if 'fit_time' in r]
                if times:
                    f.write(f"  Avg Time: {np.mean(times):.4f}s\n")

                f.write("\n")

        f.write("Top 5 Best Performing (by NMSE):\n")
        sorted_results = sorted(results, key=lambda x: x.get('nmse_global', float('inf')))
        for i, r in enumerate(sorted_results[:5], 1):
            f.write(f"{i}. {r.get('graph_file', 'N/A')}: NMSE={r.get('nmse_global', 'N/A'):.6f}\n")

        f.write("\nTop 5 Fastest:\n")
        sorted_by_time = sorted(results, key=lambda x: x.get('fit_time', float('inf')))
        for i, r in enumerate(sorted_by_time[:5], 1):
            f.write(f"{i}. {r.get('graph_file', 'N/A')}: {r.get('fit_time', 'N/A'):.4f}s\n")