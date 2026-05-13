"""
Stretch Tuesday — Calibration Analysis.

Reliability diagram + Expected Calibration Error (ECE).
"""

import numpy as np


def reliability_diagram(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """
    Bin predictions by max predicted probability; compute empirical accuracy per bin.

    Returns (bucket_centers, bucket_accuracies, bucket_counts), all length n_bins.
    """
    y_true = np.array(y_true)
    probs = np.array(probs)

    confidence = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)




    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bucket_accuracies = []
    bucket_counts = []

    for i in range(n_bins):
        lower= bin_edges[i]
        upper = bin_edges[i + 1]

        if i==n_bins - 1:
          mask=(confidence >= lower) & (confidence <= upper)
          
        else:
          mask=(confidence >= lower) & (confidence < upper)


          count = np.sum(mask)
        bucket_counts.append(count)

        if count > 0:
           acc=np.mean(preds[mask] == y_true[mask])
        else:
            acc=0.0
        bucket_accuracies.append(acc)


    return np.array(bucket_centers), np.array(bucket_accuracies), np.array(bucket_counts)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    ECE = sum over bins of (bucket_count / N) * |bucket_accuracy - bucket_confidence|.

    A perfectly calibrated model has ECE = 0.
    """
    y_true = np.array(y_true)
    probs = np.array(probs)

    confidence = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = np.sum(mask)
        if count > 0:
            acc = np.mean(preds[mask] == y_true[mask])
            avg_confidence = np.mean(confidence[mask])
            ece += (count / N) * abs(acc - avg_confidence)

    return ece


def plot_reliability(centers: np.ndarray, accs: np.ndarray, counts: np.ndarray, output_path: str) -> None:
    """Save a reliability diagram. Provided helper — do not modify."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / max(len(centers), 1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="black", alpha=0.8, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bucket center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
