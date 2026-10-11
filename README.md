# MNIST Digit Classification with a Multilayer Perceptron from Scratch

**A NumPy-only neural network implementation with manual backpropagation, systematic hyperparameter experiments, and 97.49% test accuracy on MNIST.**

![Test-set confusion matrix](results/2.15/test%20confusion%20matrix.png)

## Overview

This project implements and evaluates a fully connected **multilayer perceptron (MLP) from scratch using NumPy** for handwritten digit recognition on the [MNIST dataset](https://yann.lecun.org/exdb/mnist/). Rather than relying on a deep-learning framework's automatic differentiation or built-in neural-network layers, the project implements forward propagation, gradient computation, parameter updates, and evaluation directly.

Developed as a deep learning course project, it investigates how learning rate, input standardization, batch size, optimization algorithm, network width/depth, L2 regularization, and early stopping affect model performance.

### Highlights

- **97.49% test accuracy** (9,749 / 10,000 correct; 251 misclassified) for the selected model.
- Manual implementation of dense layers, **ReLU**, **softmax**, cross-entropy loss, and **backpropagation**.
- Configurable mini-batch SGD and momentum, optional feature standardization, L2 regularization, and early stopping.
- Comparative experiments with saved training curves, confusion matrices, per-class accuracy, and misclassified examples.
- A reusable Python training/evaluation pipeline plus an exploratory Jupyter notebook.

## Dataset

MNIST contains grayscale images of digits **0–9**, each of size **28 × 28 pixels**. Images are represented as 784-dimensional vectors. The preprocessed dataset used by this project is stored as a gzipped Python pickle containing three splits:

| Split | Samples |
| --- | ---: |
| Training | 50,000 |
| Validation | 10,000 |
| Test | 10,000 |

The code expects `homework_code/data/dataset/mnist.pkl.gz`. **The dataset file is not included in this repository archive**; obtain or prepare the MNIST pickle in the three-split format described in [Data setup](#data-setup). The official MNIST dataset is available [here](https://yann.lecun.org/exdb/mnist/).

![Examples of MNIST digits](results/2.1/5%20samples.png)

## Model and mathematical formulation

For a mini-batch $X$, each hidden layer computes

$$
Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)}, \qquad
A^{(l)} = \mathrm{ReLU}(Z^{(l)}),
$$

where $A^{(0)} = X$ and $\mathrm{ReLU}(z) = \max(0,z)$. The final layer produces logits $Z^{(L)}$ and class probabilities via softmax:

$$
P(y=k\mid x) = \frac{\exp(z_k-\max_j z_j)}{\sum_{r=1}^{10}\exp(z_r-\max_j z_j)}.
$$

Training minimizes categorical cross-entropy with optional L2 regularization on the weight matrices:

$$
\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{10}y_{ik}\log(\hat{y}_{ik})
+ \frac{\lambda}{2m}\sum_{l=1}^{L}\lVert W^{(l)}\rVert_F^2,
$$

where $m$ is the mini-batch size. The output-layer gradient is

$$
\frac{\partial \mathcal{L}}{\partial Z^{(L)}} = \frac{\hat Y-Y}{m},
$$

and the model uses the chain rule to propagate gradients through hidden ReLU layers. For SGD, parameters are updated by $\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal{L}$. With momentum, the implementation uses

$$
v_t = \beta v_{t-1} - \eta\nabla_\theta\mathcal{L}, \qquad
\theta_{t+1} = \theta_t+v_t.
$$

## Experiments and results

All accuracy values below are reproduced from the saved outputs in `homework_code/notebooks/Final.ipynb`. These are **recorded experimental results**, not newly run benchmarks. Different tables correspond to different experiment settings; they should not be treated as a single controlled leaderboard.

### Final selected model

| Setting | Value |
| --- | --- |
| Architecture | **784 → 64 (ReLU) → 10 (softmax)** |
| Optimizer | Momentum SGD |
| Learning rate | `0.01` |
| Momentum coefficient | `0.9` |
| Batch size | `8` |
| L2 coefficient | `0.001` |
| Input normalization | None |
| Training epochs | `50` |
| Seed | `42` |
| **Validation accuracy** | **97.52%** |
| **Test accuracy** | **97.49%** |

The final configuration is also saved in `homework_code/config/selected_config.yaml`.

### Learning-rate comparison

| Learning rate | Validation accuracy |
| ---: | ---: |
| 0.00001 | 19.06% |
| 0.001 | 67.72% |
| 0.005 | 83.99% |
| 0.01 | 86.90% |
| 1.0 | 91.69% |

These values come from the notebook's learning-rate experiment; they **do not imply that a learning rate of 1.0 is generally optimal** across architectures and preprocessing choices.

### Mini-batch size

| Batch size | Validation accuracy |
| ---: | ---: |
| **8** | **91.80%** |
| 16 | 89.45% |
| 32 | 86.90% |
| 128 | 79.39% |

![Batch size 8 learning curves](results/2.10/bs%3D8.png)

### Optimizer comparison

| Optimizer | Validation accuracy |
| --- | ---: |
| SGD | 91.80% |
| **Momentum** | **94.33%** |

![Momentum learning curves](results/2.11/opt%20%3D%20momentum%20plots.png)

### Hidden-layer configurations

| Hidden layers | Validation accuracy |
| --- | ---: |
| [8] | 89.87% |
| [16] | 94.33% |
| **[64]** | **95.74%** |
| [16, 32] | 91.03% |

A wider single hidden layer performed best among these recorded comparisons. Its training accuracy (99.26%) exceeded its validation accuracy (95.74%), motivating regularization experiments.

### L2 regularization

| L2 coefficient $\lambda$ | Validation accuracy | Test accuracy |
| ---: | ---: | ---: |
| 0 | 95.74% | 95.37% |
| 0.0001 | 96.21% | 95.95% |
| **0.001** | **97.52%** | **97.49%** |

The notebook's L2 experiment shows improved generalization at $\lambda=0.001$ for the evaluated setup. Test-set values are reported here because they are present in the original experiment output; hyperparameters should generally be selected on the validation set, with the test set reserved for final evaluation.

### Early stopping

An additional early-stopping experiment restored weights from **epoch 14** after training stopped at **epoch 19** (patience 5); this separate run achieved **94.33% test accuracy**. Early stopping is implemented but **was not enabled for the final 97.49% selected-configuration run**.

### Final test-set evaluation

The selected model correctly predicted **9,749 of 10,000** test digits.

![Test confusion matrix](results/2.15/test%20confusion%20matrix.png)

**Accuracy by digit** (test set):

| Digit | Accuracy | Digit | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 98.78% | 5 | 96.41% |
| 1 | 99.03% | 6 | 98.96% |
| 2 | 97.19% | 7 | 96.01% |
| 3 | 97.33% | 8 | 96.20% |
| 4 | 97.66% | 9 | 97.13% |

![Per-class test accuracy](results/2.16/per%20class%20accuracy%20on%20test%20set.png)

**Error analysis:** The model misclassified **251** test images. The saved examples provide a qualitative view of ambiguous or difficult digits.

![Misclassified MNIST digits](results/2.19/missclassified%20test%20images.png)

## Repository structure

```text
.
├── homework_code/
│   ├── config/
│   │   ├── config.yaml             # Baseline and experiment search space
│   │   └── selected_config.yaml    # Selected final model
│   ├── data/
│   │   ├── data_loader.py          # Data loading, one-hot labels, standardization
│   │   └── dataset/                # Place mnist.pkl.gz here
│   ├── models/
│   │   └── model.py                # NumPy MLP, gradients, optimizers
│   ├── notebooks/
│   │   ├── exploratory.ipynb
│   │   └── Final.ipynb             # Experiments, outputs, and analysis
│   ├── scripts/
│   │   ├── main.py                 # CLI pipeline
│   │   ├── train.py                # Mini-batch training and early stopping
│   │   └── evaluate.py             # Evaluation reporting
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── results/                       # Saved plots and figures
└── README.md
```

## Getting started

### 1. Install dependencies

Use **Python 3.10+** (Python 3.12 was used for the compiled cache in the provided project). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install numpy matplotlib pyyaml jupyter
```

The training pipeline uses NumPy, Matplotlib, and PyYAML. Jupyter is only required to explore/run notebooks.

### 2. Data setup

Create or obtain a pickle file at `homework_code/data/dataset/mnist.pkl.gz` containing precisely:

```python
((x_train, y_train), (x_val, y_val), (x_test, y_test))
```

with split sizes 50,000 / 10,000 / 10,000, flattened inputs of shape `(N, 784)`, and integer labels 0–9. **This project does not include a script to download or convert the official MNIST IDX files into this pickle format.** Downloading the four raw MNIST files alone is not sufficient to run the pipeline unchanged.

### 3. Train and evaluate

Run these commands **inside `homework_code/`**:

```bash
cd homework_code

# Run the baseline configuration
python -m scripts.main --config config/config.yaml

# Run the selected final model configuration
python -m scripts.main --config config/selected_config.yaml

# Optionally avoid displaying plots
python -m scripts.main --config config/selected_config.yaml --skip-plots

# Optionally display data samples and class distribution
python -m scripts.main --config config/selected_config.yaml --dataset-plots
```

The CLI trains the configured model and prints training metrics and a test evaluation report. Figures appear interactively unless `--skip-plots` is set. The standalone CLI does **not** automatically rerun every hyperparameter sweep in the notebook.

### 4. Explore the experiments

From `homework_code/`, open `notebooks/Final.ipynb` in Jupyter to review the original experiment code, its stored outputs, and visualizations:

```bash
jupyter notebook notebooks/Final.ipynb
```

## Implementation notes and limitations

- The neural network is written with NumPy, **without PyTorch/TensorFlow or autograd for the model**.
- Images are flattened, so the MLP does not explicitly exploit spatial structure as a convolutional neural network would.
- Results come from the provided notebook's saved run; reruns can differ with dependency versions, numerical behavior, or changed configurations.
- The archive does not contain the actual MNIST pickle dataset or pretrained model weights. You must train the model to reproduce the predictions.
- Some large learning-rate/standardization combinations produced `NaN` loss in the experiment notebook, illustrating numerical instability.

## References

- [LeCun, Cortes, and Burges — The MNIST Database](https://yann.lecun.org/exdb/mnist/)
- [GitHub Docs — Writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)

---

*Deep learning course project — implementing, tuning, and analyzing neural networks from first principles.*
