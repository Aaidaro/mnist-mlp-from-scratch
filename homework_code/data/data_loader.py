
"""Utilities for loading and preprocessing the MNIST dataset."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def resolve_relative_path(path_value: str | Path, base_dir: str | Path) -> Path:
    """Resolve a project-relative path without hardcoding machine-specific locations."""
    path = Path(path_value)
    return path if path.is_absolute() else Path(base_dir) / path


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file and return it as a dictionary."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_mnist(dataset_path: str | Path) -> tuple[np.ndarray, ...]:
    """Load the pickled MNIST splits from a compressed .pkl.gz file."""
    with gzip.open(dataset_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_dataset_from_config(config: dict[str, Any], base_dir: str | Path) -> tuple[np.ndarray, ...]:
    """Load the dataset path declared in the configuration file."""
    dataset_path = resolve_relative_path(config["data"]["mnist_path"], base_dir)
    return load_mnist(dataset_path)


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer class labels into one-hot encoded vectors."""
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def standardize_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize all splits using only the training-set statistics."""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)

    x_train_std = (x_train - mean) / std
    x_val_std = (x_val - mean) / std
    x_test_std = (x_test - mean) / std
    return x_train_std, x_val_std, x_test_std


def prepare_data(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    normalization: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the requested preprocessing strategy to the three data splits."""
    if normalization == "none":
        return x_train.copy(), x_val.copy(), x_test.copy()

    if normalization == "standardize":
        return standardize_splits(x_train, x_val, x_test)

    raise ValueError(
        f"Unsupported normalization '{normalization}'. Choose 'none' or 'standardize'."
    )
