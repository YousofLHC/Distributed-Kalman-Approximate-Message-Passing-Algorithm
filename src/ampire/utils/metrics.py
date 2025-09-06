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

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # F2 score
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    # G-Mean
    g_mean = np.sqrt(recall * specificity)

    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'kappa': kappa,
        'mcc': mcc,
        'balanced_accuracy': balanced_acc,
        'f2': f2,
        'g_mean': g_mean
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
    nmse = mse / np.var(y_true) if np.var(y_true) > 0 else 0
    peak_snr = 20 * np.log10(np.max(np.abs(y_true)) / np.sqrt(mse)) if mse > 0 else float('inf')

    return {'nmse': nmse, 'peak_snr': peak_snr}


def log_results(results, filepath='results/logs/experiment_log.txt'):
    import time
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(f"Timestamp: {time.ctime()}\n")
        for method, metrics in results.items():
            f.write(f"{method}: AUC-ROC = {metrics['auc_roc']:.4f}, Time = {metrics['time']:.2f}s\n")
        f.write("-" * 50 + "\n")