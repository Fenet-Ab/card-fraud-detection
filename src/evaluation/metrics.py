import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculate fraud detection metrics.

    Parameters
    ----------
    y_true : actual labels
    y_prob : predicted fraud probabilities
    threshold : probability threshold for fraud classification
    """

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    alerts = int(np.sum(y_pred))

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": pr_auc,
        "Alerts": alerts,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }