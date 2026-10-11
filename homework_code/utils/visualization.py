
"""Plotting utilities for training curves and model diagnostics."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_history(history: dict[str, list[float]], title: str = "Training History") -> None:
    """Visualize loss, regularization, and accuracy across epochs."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    has_regularization = any(value > 0 for value in history["train_reg_loss"])
    ncols = 3 if has_regularization else 2

    plt.figure(figsize=(6 * ncols, 4))

    plt.subplot(1, ncols, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Total Loss")
    plt.legend()
    plt.grid(True)

    if has_regularization:
        plt.subplot(1, ncols, 2)
        plt.plot(epochs, history["train_reg_loss"], label="Train L2")
        plt.plot(epochs, history["val_reg_loss"], label="Validation L2")
        plt.xlabel("Epoch")
        plt.ylabel("L2 Penalty")
        plt.title(f"{title} - Regularization")
        plt.legend()
        plt.grid(True)
        accuracy_axis_index = 3
    else:
        accuracy_axis_index = 2

    plt.subplot(1, ncols, accuracy_axis_index)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion_matrix: np.ndarray, title: str = "Confusion Matrix") -> None:
    """Display a confusion matrix as a heatmap with cell counts."""
    plt.figure(figsize=(7, 6))
    plt.imshow(confusion_matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    num_classes = confusion_matrix.shape[0]
    plt.xticks(ticks=np.arange(num_classes), labels=np.arange(num_classes))
    plt.yticks(ticks=np.arange(num_classes), labels=np.arange(num_classes))

    for row in range(num_classes):
        for col in range(num_classes):
            plt.text(col, row, str(confusion_matrix[row, col]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_random_samples_by_class(
    features: np.ndarray,
    labels: np.ndarray,
    samples_per_class: int = 5,
    image_shape: tuple[int, int] = (28, 28),
) -> None:
    """Show a few random images for each class to inspect the dataset visually."""
    classes = np.unique(labels)
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(samples_per_class, len(classes)))

    if len(classes) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, class_id in enumerate(classes):
        class_indices = np.where(labels == class_id)[0]
        chosen_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        for col, sample_index in enumerate(chosen_indices):
            axes[row, col].imshow(features[sample_index].reshape(image_shape), cmap="gray")
            axes[row, col].axis("off")

    plt.suptitle("Random Samples from Each Class")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Compare label counts across the train, validation, and test splits."""
    classes = np.arange(max(y_train.max(), y_val.max(), y_test.max()) + 1)
    train_counts = np.bincount(y_train, minlength=len(classes))
    val_counts = np.bincount(y_val, minlength=len(classes))
    test_counts = np.bincount(y_test, minlength=len(classes))

    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(classes - width, train_counts, width=width, label="Train")
    plt.bar(classes, val_counts, width=width, label="Validation")
    plt.bar(classes + width, test_counts, width=width, label="Test")
    plt.xticks(classes)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution Across Dataset Splits")
    plt.legend(loc="upper left", ncol=3)
    plt.show()


def plot_per_class_accuracy(class_accuracy: np.ndarray, title: str = "Per-Class Accuracy") -> None:
    """Visualize how well the model performs on each class individually."""
    classes = np.arange(len(class_accuracy))

    plt.figure(figsize=(8, 4))
    plt.bar(classes, class_accuracy)
    plt.xticks(classes)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(axis="y")
    plt.show()


def plot_misclassified_samples(
    features: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_show: int = 12,
    image_shape: tuple[int, int] = (28, 28),
) -> None:
    """Display a subset of the incorrectly classified images."""
    misclassified_indices = np.where(y_pred != y_true)[0]

    if len(misclassified_indices) == 0:
        print("No misclassified samples were found.")
        return

    num_show = min(num_show, len(misclassified_indices))
    chosen_indices = np.random.choice(misclassified_indices, num_show, replace=False)

    plt.figure(figsize=(12, 8))
    for plot_index, sample_index in enumerate(chosen_indices, start=1):
        plt.subplot(3, 4, plot_index)
        plt.imshow(features[sample_index].reshape(image_shape), cmap="gray")
        plt.title(f"True: {y_true[sample_index]} | Pred: {y_pred[sample_index]}")
        plt.axis("off")

    plt.suptitle("Misclassified Test Images")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
