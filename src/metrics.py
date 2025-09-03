import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Calculate evaluation metrics for anomaly detection.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_score : np.ndarray
        Predicted scores.

    Returns
    -------
    dict
        Dictionary containing AUC-ROC and AUC-PR.
    """
    auc_roc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = np.trapz(precision, recall)
    return {'auc_roc': auc_roc, 'auc_pr': auc_pr}