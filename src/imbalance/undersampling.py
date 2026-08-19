from imblearn.under_sampling import RandomUnderSampler


def apply_undersampling(X_train, y_train, random_state=42):
    """
    Reduce the number of legitimate transactions
    in the training set.
    
    The test data remains untouched.
    """
    undersampler = RandomUnderSampler(
        random_state=random_state
    )

    X_resampled, y_resampled = undersampler.fit_resample( # type: ignore
        X_train,
        y_train
    )

    print("Undersampling applied successfully.")
    print("Class distribution after undersampling:")
    print(y_resampled.value_counts())

    return X_resampled, y_resampled