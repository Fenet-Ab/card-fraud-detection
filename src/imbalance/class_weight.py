import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


def calculate_class_weights(y_train):
    """
    Calculate balanced sample weights for each training sample.
    """
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    return sample_weights


def create_custom_class_weights(y_train, fraud_weight=10):
    """
    Assign a custom weight to fraud samples.

    Legitimate transactions receive weight 1.
    Fraud transactions receive fraud_weight.
    """
    sample_weights = np.where(
        y_train == 1,
        fraud_weight,
        1
    )

    return sample_weights