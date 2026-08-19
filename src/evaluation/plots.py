import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve


def plot_precision_recall_curve(
    y_true,
    y_prob,
    pr_auc,
    title="Precision-Recall Curve"
):
    """
    Plot the Precision-Recall curve.
    """

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)

    plt.legend()
    plt.grid()

    plt.show()


def plot_model_comparison(
    comparison,
    metric="F1"
):
    """
    Plot a selected metric across fraud detection methods.
    """

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison["Method"],
        comparison[metric]
    )

    plt.xlabel("Method")
    plt.ylabel(metric)

    plt.title(
        f"{metric} Comparison"
    )

    plt.xticks(rotation=20)

    plt.grid(axis="y")

    plt.show()