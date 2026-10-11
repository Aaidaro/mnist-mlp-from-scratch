
"""Neural-network model definition for a NumPy-based MLP classifier."""

from __future__ import annotations

from typing import Any

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Apply the ReLU activation function element-wise."""
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Return the gradient of ReLU for backpropagation."""
    return (x > 0).astype(np.float64)


def softmax(x: np.ndarray) -> np.ndarray:
    """Convert logits into class probabilities in a numerically stable way."""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class MLP:
    """A fully connected multi-layer perceptron implemented only with NumPy."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        bias_init: float = 0.0,
        optimizer: str = "sgd",
        learning_rate: float = 1e-3,
        momentum_beta: float = 0.9,
        l2_lambda: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initialize the network parameters and optimizer state."""
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.num_layers = len(self.layer_dims) - 1

        self.optimizer = optimizer.lower()
        self.lr = learning_rate
        self.beta = momentum_beta
        self.l2_lambda = l2_lambda

        self.rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        self.velocity: dict[str, np.ndarray] = {}

        for layer_idx in range(1, self.num_layers + 1):
            in_dim = self.layer_dims[layer_idx - 1]
            out_dim = self.layer_dims[layer_idx]

            self.params[f"W{layer_idx}"] = self.rng.normal(
                loc=init_mean,
                scale=init_std,
                size=(in_dim, out_dim),
            )
            self.params[f"b{layer_idx}"] = np.full((1, out_dim), bias_init, dtype=np.float64)

            self.velocity[f"W{layer_idx}"] = np.zeros((in_dim, out_dim), dtype=np.float64)
            self.velocity[f"b{layer_idx}"] = np.zeros((1, out_dim), dtype=np.float64)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run the forward pass and keep intermediate activations for backpropagation."""
        cache = {"A0": x}
        activations = x

        for layer_idx in range(1, self.num_layers):
            z_value = activations @ self.params[f"W{layer_idx}"] + self.params[f"b{layer_idx}"]
            activations = relu(z_value)
            cache[f"Z{layer_idx}"] = z_value
            cache[f"A{layer_idx}"] = activations

        final_logits = (
            activations @ self.params[f"W{self.num_layers}"] + self.params[f"b{self.num_layers}"]
        )
        probabilities = softmax(final_logits)

        cache[f"Z{self.num_layers}"] = final_logits
        cache[f"A{self.num_layers}"] = probabilities
        return probabilities, cache

    def l2_penalty(self, batch_size: int) -> float:
        """Compute the L2 regularization term for the current weights."""
        penalty = 0.0
        for layer_idx in range(1, self.num_layers + 1):
            penalty += np.sum(self.params[f"W{layer_idx}"] ** 2)
        return float((self.l2_lambda / (2 * batch_size)) * penalty)

    def backward(
        self,
        y_true_onehot: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Backpropagate the output error and compute parameter gradients."""
        grads: dict[str, np.ndarray] = {}
        batch_size = y_true_onehot.shape[0]

        output_probs = cache[f"A{self.num_layers}"]
        prev_activation = cache[f"A{self.num_layers - 1}"] if self.num_layers > 1 else cache["A0"]

        d_logits = (output_probs - y_true_onehot) / batch_size
        grads[f"W{self.num_layers}"] = (
            prev_activation.T @ d_logits
            + (self.l2_lambda / batch_size) * self.params[f"W{self.num_layers}"]
        )
        grads[f"b{self.num_layers}"] = np.sum(d_logits, axis=0, keepdims=True)

        d_prev = d_logits @ self.params[f"W{self.num_layers}"].T

        for layer_idx in range(self.num_layers - 1, 0, -1):
            z_value = cache[f"Z{layer_idx}"]
            prev_activation = cache[f"A{layer_idx - 1}"] if layer_idx - 1 > 0 else cache["A0"]

            d_logits = d_prev * relu_grad(z_value)
            grads[f"W{layer_idx}"] = (
                prev_activation.T @ d_logits
                + (self.l2_lambda / batch_size) * self.params[f"W{layer_idx}"]
            )
            grads[f"b{layer_idx}"] = np.sum(d_logits, axis=0, keepdims=True)

            if layer_idx > 1:
                d_prev = d_logits @ self.params[f"W{layer_idx}"].T

        return grads

    def update_params(self, grads: dict[str, np.ndarray]) -> None:
        """Update weights and biases using SGD or momentum."""
        for layer_idx in range(1, self.num_layers + 1):
            weight_key = f"W{layer_idx}"
            bias_key = f"b{layer_idx}"

            if self.optimizer == "sgd":
                self.params[weight_key] -= self.lr * grads[weight_key]
                self.params[bias_key] -= self.lr * grads[bias_key]
            elif self.optimizer == "momentum":
                self.velocity[weight_key] = (
                    self.beta * self.velocity[weight_key] - self.lr * grads[weight_key]
                )
                self.velocity[bias_key] = (
                    self.beta * self.velocity[bias_key] - self.lr * grads[bias_key]
                )

                self.params[weight_key] += self.velocity[weight_key]
                self.params[bias_key] += self.velocity[bias_key]
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for a batch of samples."""
        probabilities, _ = self.forward(x)
        return probabilities

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the most likely class index for each sample."""
        probabilities = self.predict_proba(x)
        return np.argmax(probabilities, axis=1)

    def get_state(self) -> dict[str, dict[str, np.ndarray]]:
        """Create a copy of the parameters so the best model can be restored later."""
        return {
            "params": {name: value.copy() for name, value in self.params.items()},
            "velocity": {name: value.copy() for name, value in self.velocity.items()},
        }

    def set_state(self, state: dict[str, dict[str, np.ndarray]]) -> None:
        """Restore a previously saved parameter state."""
        self.params = {name: value.copy() for name, value in state["params"].items()}
        self.velocity = {name: value.copy() for name, value in state["velocity"].items()}


def build_model_from_config(config: dict[str, Any]) -> MLP:
    """Construct an MLP instance directly from the YAML configuration."""
    return MLP(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        output_dim=config["model"]["output_dim"],
        init_mean=config["init"]["mean"],
        init_std=config["init"]["std"],
        bias_init=config["init"]["bias"],
        optimizer=config["train"]["optimizer"],
        learning_rate=config["train"]["learning_rate"],
        momentum_beta=config["train"]["momentum_beta"],
        l2_lambda=config["train"].get("l2_lambda", 0.0),
        seed=config["seed"],
    )
