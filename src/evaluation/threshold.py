import numpy as np


def apply_threshold(y_prob, threshold=0.5):
    """
    Convert fraud probabilities into class predictions.
    
    Probability >= threshold → Fraud
    Probability < threshold  → Legitimate
    """
    return (y_prob >= threshold).astype(int)


def count_alerts(y_prob, threshold):
    """
    Count how many transactions would be flagged
    as fraud at a given threshold.
    """
    y_pred = apply_threshold(
        y_prob,
        threshold
    )

    return int(np.sum(y_pred))


def find_threshold_for_alert_budget(
    y_prob,
    alert_budget=500
):
    """
    Find a threshold that keeps the number of alerts
    at or below the specified alert budget.

    A lower threshold creates more alerts.
    A higher threshold creates fewer alerts.
    """

    thresholds = np.linspace(
        0.01,
        0.99,
        99
    )

    valid_thresholds = []

    for threshold in thresholds:

        alerts = count_alerts(
            y_prob,
            threshold
        )

        if alerts <= alert_budget:
            valid_thresholds.append(
                (threshold, alerts)
            )

    if not valid_thresholds:
        return 0.99

    # Select the lowest threshold that satisfies
    # the alert budget, maximizing fraud sensitivity.
    best_threshold = min(
        valid_thresholds,
        key=lambda x: x[0]
    )[0]

    return best_threshold