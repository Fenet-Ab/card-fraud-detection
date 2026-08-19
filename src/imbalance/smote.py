import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

df = pd.read_csv("../data/processed/creditcard_processed.csv")

print("Dataset shape:")
print(df.shape)

print("\nOriginal class distribution:")
print(df["Class"].value_counts())

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

model = GradientBoostingClassifier(
    random_state=42
)

print("\nTraining Gradient Boosting model...")

model.fit(
    X_train_smote,
    y_train_smote
)

print("Model training completed!")

y_prob = model.predict_proba(X_test)[:, 1]

print("\nFirst 10 fraud probabilities:")
print(y_prob[:10])

threshold = 0.5

y_pred = (y_prob >= threshold).astype(int)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

pr_auc = average_precision_score(y_test, y_prob)

print("\n================================")
print("        SMOTE RESULTS")
print("================================")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("================================")

precision_curve, recall_curve, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"SMOTE (PR-AUC = {pr_auc:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("SMOTE - Precision-Recall Curve")

plt.legend()
plt.grid()

plt.show()