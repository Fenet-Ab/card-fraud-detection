from imblearn.over_sampling import SMOTE


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE only to the training data.

    The test data must remain untouched.
    """
    smote = SMOTE(
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample( # type: ignore
        X_train,
        y_train
    )

    print("SMOTE applied successfully.")
    print("Class distribution after SMOTE:")
    print(y_resampled.value_counts())

    return X_resampled, y_resampled