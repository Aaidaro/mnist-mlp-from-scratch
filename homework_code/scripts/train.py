
"""Training utilities for running experiments with the MLP classifier."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from data.data_loader import one_hot_encode, prepare_data
from models.model import MLP, build_model_from_config
from utils.metrics import accuracy_score, confusion_matrix_np, cross_entropy_loss
from utils.visualization import plot_confusion_matrix, plot_history


def iterate_minibatches(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
):
    """Yield shuffled mini-batches so the model can train in small chunks."""
    indices = rng.permutation(features.shape[0])
    for start_idx in range(0, features.shape[0], batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield features[batch_indices], labels[batch_indices]


def evaluate_model(
    model: MLP,
    features: np.ndarray,
    labels: np.ndarray,
    return_loss_parts: bool = False,
):
    """Evaluate one dataset split and optionally return separate CE and L2 terms."""
    num_classes = model.layer_dims[-1]
    labels_onehot = one_hot_encode(labels, num_classes)

    probabilities, _ = model.forward(features)
    ce_loss = cross_entropy_loss(labels_onehot, probabilities)
    reg_loss = model.l2_penalty(features.shape[0])
    total_loss = ce_loss + reg_loss
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(labels, predictions)

    if return_loss_parts:
        return total_loss, ce_loss, reg_loss, accuracy, predictions

    return total_loss, accuracy, predictions


def train_model(
    model: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 0.0,
    restore_best_weights: bool = True,
) -> dict[str, Any]:
    """Train the model and track losses and accuracies after every epoch."""
    rng = np.random.default_rng(seed)
    num_classes = model.layer_dims[-1]

    history: dict[str, Any] = {
        "train_loss": [],
        "train_ce_loss": [],
        "train_reg_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_ce_loss": [],
        "val_reg_loss": [],
        "val_acc": [],
        "stopped_epoch": None,
        "best_epoch": None,
        "best_val_loss": None,
    }

    best_val_loss = np.inf
    best_epoch = -1
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        for x_batch, y_batch in iterate_minibatches(x_train, y_train, batch_size, rng):
            y_batch_onehot = one_hot_encode(y_batch, num_classes)
            _, cache = model.forward(x_batch)
            gradients = model.backward(y_batch_onehot, cache)
            model.update_params(gradients)

        train_loss, train_ce_loss, train_reg_loss, train_acc, _ = evaluate_model(
            model,
            x_train,
            y_train,
            return_loss_parts=True,
        )
        val_loss, val_ce_loss, val_reg_loss, val_acc, _ = evaluate_model(
            model,
            x_val,
            y_val,
            return_loss_parts=True,
        )

        history["train_loss"].append(train_loss)
        history["train_ce_loss"].append(train_ce_loss)
        history["train_reg_loss"].append(train_reg_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_ce_loss"].append(val_ce_loss)
        history["val_reg_loss"].append(val_reg_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}, L2: {train_reg_loss:.4f}) | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} (CE: {val_ce_loss:.4f}, L2: {val_reg_loss:.4f}) | "
            f"Val Acc: {val_acc:.4f}"
        )

        if early_stopping:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = model.get_state()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best epoch was {best_epoch} with validation loss = {best_val_loss:.4f}."
                )
                history["stopped_epoch"] = epoch + 1
                break

    if early_stopping and restore_best_weights and best_state is not None:
        model.set_state(best_state)
        print(f"Restored the best model weights from epoch {best_epoch}.")

    history["best_epoch"] = best_epoch if early_stopping else len(history["val_loss"])
    history["best_val_loss"] = best_val_loss if early_stopping else min(history["val_loss"])
    return history


def run_experiment(
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    exp_name: str = "experiment",
    early_stopping: bool = False,
    patience: int = 5,
    min_delta: float = 0.0,
    plot_results: bool = True,
) -> dict[str, Any]:
    """Train one configuration, evaluate all splits, and return a full result dictionary."""
    x_train_p, x_val_p, x_test_p = prepare_data(
        x_train,
        x_val,
        x_test,
        normalization=config["preprocessing"]["normalization"],
    )

    model = build_model_from_config(config)
    history = train_model(
        model,
        x_train_p,
        y_train,
        x_val_p,
        y_val,
        epochs=config["train"]["epochs"],
        batch_size=config["train"]["batch_size"],
        seed=config["seed"],
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
    )

    train_loss, train_ce_loss, train_reg_loss, train_acc, train_preds = evaluate_model(
        model,
        x_train_p,
        y_train,
        return_loss_parts=True,
    )
    val_loss, val_ce_loss, val_reg_loss, val_acc, val_preds = evaluate_model(
        model,
        x_val_p,
        y_val,
        return_loss_parts=True,
    )
    test_loss, test_ce_loss, test_reg_loss, test_acc, test_preds = evaluate_model(
        model,
        x_test_p,
        y_test,
        return_loss_parts=True,
    )

    print(
        f"\n[{exp_name}] Final Validation -> "
        f"Loss: {val_loss:.4f} (CE: {val_ce_loss:.4f}, L2: {val_reg_loss:.4f}), Accuracy: {val_acc:.4f}"
    )
    print(
        f"[{exp_name}] Final Test       -> "
        f"Loss: {test_loss:.4f} (CE: {test_ce_loss:.4f}, L2: {test_reg_loss:.4f}), Accuracy: {test_acc:.4f}"
    )

    if early_stopping:
        print(
            f"[{exp_name}] Best Epoch: {history['best_epoch']} | "
            f"Stopped Epoch: {history['stopped_epoch']}"
        )

    num_classes = config["model"]["output_dim"]
    val_confusion = confusion_matrix_np(y_val, val_preds, num_classes=num_classes)
    test_confusion = confusion_matrix_np(y_test, test_preds, num_classes=num_classes)

    if plot_results:
        plot_history(history, title=exp_name)
        plot_confusion_matrix(val_confusion, title=f"{exp_name} - Validation Confusion Matrix")

    return {
        "config": copy.deepcopy(config),
        "model": model,
        "history": history,
        "x_train_processed": x_train_p,
        "x_val_processed": x_val_p,
        "x_test_processed": x_test_p,
        "train_loss": train_loss,
        "train_ce_loss": train_ce_loss,
        "train_reg_loss": train_reg_loss,
        "train_acc": train_acc,
        "train_preds": train_preds,
        "val_loss": val_loss,
        "val_ce_loss": val_ce_loss,
        "val_reg_loss": val_reg_loss,
        "val_acc": val_acc,
        "val_preds": val_preds,
        "val_confusion": val_confusion,
        "test_loss": test_loss,
        "test_ce_loss": test_ce_loss,
        "test_reg_loss": test_reg_loss,
        "test_acc": test_acc,
        "test_preds": test_preds,
        "test_confusion": test_confusion,
        "best_epoch": history["best_epoch"],
        "stopped_epoch": history["stopped_epoch"],
    }
