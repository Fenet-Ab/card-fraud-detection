from sklearn.ensemble import GradientBoostingClassifier


def create_gradient_boosting_model(random_state=42):
    """
    Create the Gradient Boosting classifier used
    throughout the fraud detection experiments.
    """
    model = GradientBoostingClassifier(
        random_state=random_state
    )

    return model


def train_model(model, X_train, y_train, sample_weight=None):
    """
    Train the Gradient Boosting model.

    sample_weight is used for the class-weight experiment.
    """
    print("Training Gradient Boosting model...")

    if sample_weight is not None:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight
        )
    else:
        model.fit(
            X_train,
            y_train
        )

    print("Model training completed!")

    return model