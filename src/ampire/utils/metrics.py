import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, precision_score, recall_score,
    f1_score, cohen_kappa_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef
)

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

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Additional rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0  # False Discovery Rate
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    for_ = fn / (fn + tn) if (fn + tn) > 0 else 0  # False Omission Rate

    # F2 score
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

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
    nmse = mse / np.var(y_true) if np.var(y_true) > 0 else 0
    snr = 10 * np.log10(np.var(y_true) / mse) if mse > 0 else float('inf')
    peak_snr = 20 * np.log10(np.max(np.abs(y_true)) / np.sqrt(mse)) if mse > 0 else float('inf')

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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(f"Timestamp: {time.ctime()}\n")
        for method, metrics in results.items():
            f.write(f"{method}: AUC-ROC = {metrics['auc_roc']:.4f}, Time = {metrics['time']:.2f}s\n")
        f.write("-" * 50 + "\n")