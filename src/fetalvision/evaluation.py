
from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from fetalvision.data import CLASSES



def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Compute global and per-class classification metrics."""

    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Labels must be non-empty and have equal lengths.")

    labels = list(range(len(CLASSES)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }

    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in CLASSES
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
