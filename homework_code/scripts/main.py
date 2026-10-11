
"""Run the full training and evaluation pipeline from one configuration file.

Example:
    python -m scripts.main --config config/config.yaml
    python -m scripts.main --config config/selected_config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.data_loader import load_dataset_from_config, load_yaml_config
from scripts.evaluate import create_evaluation_report, print_evaluation_report, show_evaluation_plots
from scripts.train import run_experiment
from utils.visualization import plot_class_distribution, plot_random_samples_by_class


def parse_args() -> argparse.Namespace:
    """Read command-line arguments so different config files can be used easily."""
    parser = argparse.ArgumentParser(description="Train and evaluate the NumPy MLP pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Relative path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dataset-plots",
        action="store_true",
        help="Show dataset sample plots and class distribution before training.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip training and evaluation plots.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute data loading, training, and final evaluation in one place."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / args.config

    config = load_yaml_config(config_path)
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_from_config(config, project_root)

    print("Training set  :", x_train.shape, y_train.shape)
    print("Validation set:", x_val.shape, y_val.shape)
    print("Test set      :", x_test.shape, y_test.shape)

    if x_train.shape[1] != config["model"]["input_dim"]:
        raise ValueError("model.input_dim does not match the dataset feature size.")

    if args.dataset_plots and not args.skip_plots:
        plot_random_samples_by_class(x_train, y_train)
        plot_class_distribution(y_train, y_val, y_test)

    results = run_experiment(
        config=config,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        exp_name="MLP Experiment",
        early_stopping=False,
        plot_results=not args.skip_plots,
    )

    report = create_evaluation_report(
        y_true=y_test,
        y_pred=results["test_preds"],
        num_classes=config["model"]["output_dim"],
    )
    print_evaluation_report(report, split_name="Test")

    if not args.skip_plots:
        show_evaluation_plots(
            x_test=results["x_test_processed"],
            y_true=y_test,
            y_pred=results["test_preds"],
            report=report,
        )


if __name__ == "__main__":
    main()
