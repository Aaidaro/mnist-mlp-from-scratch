
"""Evaluation helpers for summarizing and visualizing trained-model results."""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.metrics import accuracy_score, confusion_matrix_np, per_class_accuracy
from utils.visualization import (
    plot_confusion_matrix,
    plot_misclassified_samples,
    plot_per_class_accuracy,
)


def create_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Collect the main evaluation artifacts for a set of predictions."""
    confusion = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion,
        "per_class_accuracy": per_class_accuracy(confusion),
    }


def print_evaluation_report(report: dict[str, Any], split_name: str = "Test") -> None:
    """Print a compact text summary of overall and per-class performance."""
    print(f"{split_name} accuracy: {report['accuracy']:.4f}")
    print("-" * 40)
    print("Per-class accuracy:")
    for class_id, class_acc in enumerate(report["per_class_accuracy"]):
        print(f"Class {class_id}: {class_acc:.4f}")


def show_evaluation_plots(
    x_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    report: dict[str, Any],
) -> None:
    """Render the confusion matrix, per-class accuracy, and misclassified images."""
    plot_confusion_matrix(report["confusion_matrix"], title="Test Confusion Matrix")
    plot_per_class_accuracy(report["per_class_accuracy"], title="Per-Class Accuracy on Test Set")
    plot_misclassified_samples(x_test, y_true, y_pred)
