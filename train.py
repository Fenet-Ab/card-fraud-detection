from src.data.preprocess import (
    load_data,
    split_features_target,
    train_test_split_data
)

from src.models.gradient_boosting import (
    create_gradient_boosting_model,
    train_model
)

from src.imbalance.smote import apply_smote
from src.imbalance.undersampling import apply_undersampling
from src.imbalance.class_weight import calculate_class_weights

from src.evaluation.metrics import calculate_metrics
from src.evaluation.threshold import find_threshold_for_alert_budget

import pandas as pd



# 1. LOAD DATA


df = load_data(
    "data/processed/creditcard_processed.csv"
)



# 2. SEPARATE FEATURES AND TARGET


X, y = split_features_target(df)



# 3. TRAIN / TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split_data(
    X,
    y
)

print("\nTraining data:", X_train.shape)
print("Test data:", X_test.shape)

print("\nTest class distribution:")
print(y_test.value_counts())


# 4. STORE RESULTS


all_results = []



# 5. BASELINE


print("\n" + "=" * 60)
print("BASELINE MODEL")
print("=" * 60)

baseline_model = create_gradient_boosting_model()

baseline_model = train_model(
    baseline_model,
    X_train,
    y_train
)

baseline_prob = baseline_model.predict_proba(
    X_test
)[:, 1]

baseline_results = calculate_metrics(
    y_test,
    baseline_prob,
    threshold=0.5
)

baseline_results["Method"] = "Baseline"

all_results.append(baseline_results)

print("\nBaseline results:")

for metric, value in baseline_results.items():
    print(f"{metric}: {value}")



# 6. SMOTE


print("\n" + "=" * 60)
print("SMOTE")
print("=" * 60)

X_train_smote, y_train_smote = apply_smote(
    X_train,
    y_train
)

smote_model = create_gradient_boosting_model()

smote_model = train_model(
    smote_model,
    X_train_smote,
    y_train_smote
)

smote_prob = smote_model.predict_proba(
    X_test
)[:, 1]

smote_results = calculate_metrics(
    y_test,
    smote_prob,
    threshold=0.5
)

smote_results["Method"] = "SMOTE"

all_results.append(smote_results)

print("\nSMOTE results:")

for metric, value in smote_results.items():
    print(f"{metric}: {value}")



# 7. UNDERSAMPLING


print("\n" + "=" * 60)
print("UNDERSAMPLING")
print("=" * 60)

X_train_under, y_train_under = apply_undersampling(
    X_train,
    y_train
)

undersampling_model = create_gradient_boosting_model()

undersampling_model = train_model(
    undersampling_model,
    X_train_under,
    y_train_under
)

undersampling_prob = undersampling_model.predict_proba(
    X_test
)[:, 1]

undersampling_results = calculate_metrics(
    y_test,
    undersampling_prob,
    threshold=0.5
)

undersampling_results["Method"] = "Undersampling"

all_results.append(undersampling_results)

print("\nUndersampling results:")

for metric, value in undersampling_results.items():
    print(f"{metric}: {value}")



# 8. CLASS WEIGHT


print("\n" + "=" * 60)
print("CLASS WEIGHT")
print("=" * 60)

sample_weights = calculate_class_weights(
    y_train
)

class_weight_model = create_gradient_boosting_model()

class_weight_model = train_model(
    class_weight_model,
    X_train,
    y_train,
    sample_weight=sample_weights
)

class_weight_prob = class_weight_model.predict_proba(
    X_test
)[:, 1]

class_weight_results = calculate_metrics(
    y_test,
    class_weight_prob,
    threshold=0.5
)

class_weight_results["Method"] = "Class Weight"

all_results.append(class_weight_results)

print("\nClass Weight results:")

for metric, value in class_weight_results.items():
    print(f"{metric}: {value}")


# 9. THRESHOLD TUNING


print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

# Use the BASELINE model.
# Threshold tuning does not train another model.

alert_budget = 500

tuned_threshold = find_threshold_for_alert_budget(
    baseline_prob,
    alert_budget=alert_budget
)

print(f"\nAlert budget: {alert_budget}")
print(f"Selected threshold: {tuned_threshold:.4f}")

threshold_results = calculate_metrics(
    y_test,
    baseline_prob,
    threshold=tuned_threshold
)

threshold_results["Method"] = "Threshold Tuning"

all_results.append(threshold_results)

print("\nThreshold tuning results:")

for metric, value in threshold_results.items():
    print(f"{metric}: {value}")



# 10. FINAL COMPARISON TABLE


comparison = pd.DataFrame(
    all_results
)


# Put Method first
columns = [
    "Method",
    "Precision",
    "Recall",
    "F1",
    "PR-AUC",
    "Alerts",
    "TN",
    "FP",
    "FN",
    "TP"
]

comparison = comparison[columns]


# 11. DISPLAY COMPARISON


print("\n\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False
    )
)


# 12. SAVE RESULTS


comparison.to_csv(
    "results/src_model_comparison.csv",
    index=False
)

print(
    "\nComparison saved to:"
    " results/src_model_comparison.csv"
)