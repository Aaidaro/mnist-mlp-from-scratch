
"""Metric helpers used during training and evaluation."""

from __future__ import annotations

import numpy as np


def cross_entropy_loss(
    y_true_onehot: np.ndarray,
    y_pred_probs: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute the average multi-class cross-entropy loss."""
    clipped_probs = np.clip(y_pred_probs, eps, 1.0)
    return float(-np.sum(y_true_onehot * np.log(clipped_probs)) / y_true_onehot.shape[0])


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Measure the fraction of correctly classified samples."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Build a confusion matrix using NumPy only."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def per_class_accuracy(confusion_matrix: np.ndarray) -> np.ndarray:
    """Compute one accuracy value for each class from a confusion matrix."""
    correct = confusion_matrix.diagonal()
    totals = np.maximum(confusion_matrix.sum(axis=1), 1)
    return correct / totals
