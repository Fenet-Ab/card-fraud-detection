import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Load the credit card fraud dataset.
    """
    df = pd.read_csv(file_path)

    print(f"Dataset loaded: {df.shape}")

    return df


def split_features_target(df):
    """
    Separate features (X) from target (y).
    
    Class:
        0 = Legitimate
        1 = Fraud
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def train_test_split_data(X, y, test_size=0.20, random_state=42):
    """
    Create a stratified train/test split.
    
    Stratification preserves the fraud/legitimate
    class distribution in both sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test