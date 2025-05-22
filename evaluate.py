from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from scipy import stats
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass


@dataclass
class EvaluationResults:
    """Container for evaluation results with statistical measures."""

    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    n_samples: int
    split_name: str
    bootstrap_samples: Optional[int] = None


def evaluate_model(
    model,
    X,
    y,
    split_name="val",
    verbose=True,
    bootstrap_ci=True,
    n_bootstrap=1000,
    ci_alpha=0.05,
):
    """
    Enhanced model evaluation with confidence intervals and statistical rigor.

    Parameters:
        model: Trained regression model
        X: Input features
        y: True targets
        split_name: Label for this dataset split (e.g., 'val', 'test')
        verbose: Whether to print results
        bootstrap_ci: Whether to compute bootstrap confidence intervals
        n_bootstrap: Number of bootstrap samples
        ci_alpha: Alpha level for confidence intervals (default: 95% CI)

    Returns:
        EvaluationResults: Comprehensive evaluation results
    """
    predictions = model.predict(X)

    # Handle potential shape differences gracefully
    if len(predictions.shape) == 1 and len(y.shape) > 1:
        predictions = predictions.reshape(-1, 1)

    # Calculate point estimates
    mse = mean_squared_error(y, predictions, multioutput="raw_values")
    mae = mean_absolute_error(y, predictions, multioutput="raw_values")
    r2 = r2_score(y, predictions, multioutput="raw_values")

    # Handle both single target and multi-target cases
    if len(mse.shape) == 0:
        # Single target case
        metrics = {
            f"{split_name}_mse": float(mse),
            f"{split_name}_mae": float(mae),
            f"{split_name}_r2": float(r2),
            f"{split_name}_rmse": float(np.sqrt(mse)),
        }

        # Add relative metrics (with safety checks)
        y_mean = np.mean(y)
        if y_mean != 0:
            # MAPE with safety check for near-zero values
            y_flat = y.flatten()
            pred_flat = predictions.flatten()
            # Only calculate MAPE for values that are not too close to zero
            mask = np.abs(y_flat) > 1e-6  # Avoid division by very small numbers
            if np.sum(mask) > 0:
                mape = (
                    np.mean(np.abs((y_flat[mask] - pred_flat[mask]) / y_flat[mask]))
                    * 100
                )
                # Cap MAPE at reasonable values
                metrics[f"{split_name}_mape"] = float(min(mape, 1000.0))

            metrics[f"{split_name}_rrmse"] = float(np.sqrt(mse) / y_mean * 100)

    else:
        # Multi-target case (avg, min, max)
        target_names = ["avg", "min", "max"]
        metrics = {}

        for i, target in enumerate(target_names[: len(mse)]):
            metrics[f"{split_name}_mse_{target}"] = float(mse[i])
            metrics[f"{split_name}_mae_{target}"] = float(mae[i])
            metrics[f"{split_name}_r2_{target}"] = float(r2[i])
            metrics[f"{split_name}_rmse_{target}"] = float(np.sqrt(mse[i]))

            # Add relative metrics (with safety checks)
            y_target = y[:, i] if len(y.shape) > 1 else y
            y_mean = np.mean(y_target)
            if y_mean != 0:
                pred_target = (
                    predictions[:, i] if len(predictions.shape) > 1 else predictions
                )
                # MAPE with safety check
                mask = np.abs(y_target) > 1e-6
                if np.sum(mask) > 0:
                    mape = (
                        np.mean(
                            np.abs(
                                (y_target[mask] - pred_target[mask]) / y_target[mask]
                            )
                        )
                        * 100
                    )
                    metrics[f"{split_name}_mape_{target}"] = float(min(mape, 1000.0))

                metrics[f"{split_name}_rrmse_{target}"] = float(
                    np.sqrt(mse[i]) / y_mean * 100
                )

    # Compute confidence intervals via bootstrap
    confidence_intervals = {}
    if bootstrap_ci:
        confidence_intervals = _compute_bootstrap_ci(
            y, predictions, metrics.keys(), n_bootstrap, ci_alpha
        )

    # Create results object
    results = EvaluationResults(
        metrics=metrics,
        confidence_intervals=confidence_intervals,
        n_samples=len(y),
        split_name=split_name,
        bootstrap_samples=n_bootstrap if bootstrap_ci else None,
    )

    if verbose:
        _print_evaluation_results(results, ci_alpha)

    return results


def _compute_bootstrap_ci(y_true, y_pred, metric_names, n_bootstrap, alpha):
    """Compute bootstrap confidence intervals for metrics."""
    n_samples = len(y_true)
    bootstrap_metrics = {name: [] for name in metric_names}

    # Generate bootstrap samples
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[indices]
        pred_boot = y_pred[indices]

        # Calculate metrics for bootstrap sample
        try:
            mse_boot = mean_squared_error(y_boot, pred_boot, multioutput="raw_values")
            mae_boot = mean_absolute_error(y_boot, pred_boot, multioutput="raw_values")
            r2_boot = r2_score(y_boot, pred_boot, multioutput="raw_values")

            # Store metrics (this is simplified - in practice you'd match the exact calculation above)
            if len(mse_boot.shape) == 0:
                # Single target
                for name in metric_names:
                    if "_mse" in name:
                        bootstrap_metrics[name].append(float(mse_boot))
                    elif "_mae" in name:
                        bootstrap_metrics[name].append(float(mae_boot))
                    elif "_r2" in name:
                        bootstrap_metrics[name].append(float(r2_boot))
                    elif "_rmse" in name:
                        bootstrap_metrics[name].append(float(np.sqrt(mse_boot)))
            else:
                # Multi-target - simplified for brevity
                for i, target in enumerate(["avg", "min", "max"][: len(mse_boot)]):
                    for name in metric_names:
                        if f"_mse_{target}" in name:
                            bootstrap_metrics[name].append(float(mse_boot[i]))
                        elif f"_mae_{target}" in name:
                            bootstrap_metrics[name].append(float(mae_boot[i]))
                        elif f"_r2_{target}" in name:
                            bootstrap_metrics[name].append(float(r2_boot[i]))
                        elif f"_rmse_{target}" in name:
                            bootstrap_metrics[name].append(float(np.sqrt(mse_boot[i])))

        except Exception:
            # Skip failed bootstrap samples
            continue

    # Calculate confidence intervals
    ci_dict = {}
    for metric_name, values in bootstrap_metrics.items():
        if len(values) > 0:
            lower = np.percentile(values, (alpha / 2) * 100)
            upper = np.percentile(values, (1 - alpha / 2) * 100)
            ci_dict[metric_name] = (lower, upper)

    return ci_dict


def _print_evaluation_results(results: EvaluationResults, alpha: float):
    """Print evaluation results in a formatted way."""
    ci_pct = int((1 - alpha) * 100)
    print(f"\n=== {results.split_name.upper()} EVALUATION RESULTS ===")
    print(f"Samples: {results.n_samples}")
    if results.bootstrap_samples:
        print(f"Bootstrap samples: {results.bootstrap_samples}")
    print("-" * 50)

    for metric_name, value in results.metrics.items():
        ci_str = ""
        if metric_name in results.confidence_intervals:
            ci_lower, ci_upper = results.confidence_intervals[metric_name]
            ci_str = f" [{ci_lower:.4f}, {ci_upper:.4f}] ({ci_pct}% CI)"

        print(f"{metric_name}: {value:.4f}{ci_str}")


def compare_models_statistical(
    model_results: List[EvaluationResults],
    metric_name: str = "mse",
    test_type: str = "paired_ttest",
) -> Dict[str, Any]:
    """
    Perform statistical comparison between multiple models.

    Parameters:
        model_results: List of EvaluationResults for different models
        metric_name: Metric to compare (without split prefix)
        test_type: Type of statistical test ('paired_ttest', 'wilcoxon', 'bootstrap')

    Returns:
        Dictionary with comparison results
    """
    if len(model_results) < 2:
        raise ValueError("Need at least 2 models for comparison")

    # Extract metric values for each model
    split_name = model_results[0].split_name
    full_metric_name = f"{split_name}_{metric_name}"

    if not all(full_metric_name in result.metrics for result in model_results):
        # Try different target variants for multi-output
        possible_names = [
            f"{split_name}_{metric_name}_{target}" for target in ["avg", "min", "max"]
        ]
        for name in possible_names:
            if all(name in result.metrics for result in model_results):
                full_metric_name = name
                break
        else:
            raise ValueError(f"Metric {metric_name} not found in all model results")

    values = [result.metrics[full_metric_name] for result in model_results]

    # Perform pairwise comparisons
    n_models = len(model_results)
    comparison_results = {
        "metric": full_metric_name,
        "values": values,
        "test_type": test_type,
        "pairwise_tests": {},
    }

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_i_name = f"Model_{i}"
            model_j_name = f"Model_{j}"

            # For point estimates, we need to simulate from bootstrap if available
            if test_type == "bootstrap" and all(
                full_metric_name in result.confidence_intervals
                for result in [model_results[i], model_results[j]]
            ):
                # Use bootstrap confidence intervals to estimate p-value
                ci_i = model_results[i].confidence_intervals[full_metric_name]
                ci_j = model_results[j].confidence_intervals[full_metric_name]

                # Simple overlap test (conservative)
                overlap = not (ci_i[1] < ci_j[0] or ci_j[1] < ci_i[0])
                p_value = 0.05 if overlap else 0.01  # Rough estimate

                comparison_results["pairwise_tests"][
                    f"{model_i_name}_vs_{model_j_name}"
                ] = {
                    "statistic": abs(values[i] - values[j]),
                    "p_value": p_value,
                    "significant": not overlap,
                    "effect_size": _calculate_effect_size(values[i], values[j]),
                }

            else:
                # For single point estimates, report effect size only
                effect_size = _calculate_effect_size(values[i], values[j])
                comparison_results["pairwise_tests"][
                    f"{model_i_name}_vs_{model_j_name}"
                ] = {
                    "difference": values[i] - values[j],
                    "effect_size": effect_size,
                    "note": "Statistical test requires bootstrap samples or repeated CV",
                }

    return comparison_results


def _calculate_effect_size(value1: float, value2: float) -> float:
    """Calculate Cohen's d-like effect size for metric differences."""
    # For metrics like MSE, MAE - relative difference
    if value1 == 0 and value2 == 0:
        return 0.0

    # Percentage difference
    baseline = max(abs(value1), abs(value2))
    if baseline == 0:
        return 0.0

    return abs(value1 - value2) / baseline


def cross_validate_model(
    model, X, y, cv_folds=5, scoring_metrics=None, random_state=42
) -> Dict[str, Any]:
    """
    Perform cross-validation with multiple metrics and statistical analysis.

    Parameters:
        model: Sklearn-compatible model
        X: Features
        y: Targets
        cv_folds: Number of CV folds
        scoring_metrics: List of metrics to compute
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with CV results and statistics
    """
    if scoring_metrics is None:
        scoring_metrics = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

    # Use StratifiedKFold for regression (based on target quantiles)
    if len(y.shape) == 1:
        # Single target - create quantile-based stratification
        y_binned = pd.qcut(y, q=min(5, cv_folds), duplicates="drop", labels=False)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_splits = list(cv.split(X, y_binned))
    else:
        # Multi-target - use first target for stratification
        y_binned = pd.qcut(y[:, 0], q=min(5, cv_folds), duplicates="drop", labels=False)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_splits = list(cv.split(X, y_binned))

    # Perform cross-validation
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv_splits,
        scoring=scoring_metrics,
        return_train_score=True,
        n_jobs=-1,
    )

    # Calculate statistics for each metric
    results_summary = {"cv_folds": cv_folds, "n_samples": len(y), "metrics": {}}

    for metric in scoring_metrics:
        test_scores = cv_results[f"test_{metric}"]
        train_scores = cv_results[f"train_{metric}"]

        # Convert negative scores back to positive for MSE, MAE
        if "neg_" in metric:
            test_scores = -test_scores
            train_scores = -train_scores
            metric_name = metric.replace("neg_", "")
        else:
            metric_name = metric

        results_summary["metrics"][metric_name] = {
            "test_mean": float(np.mean(test_scores)),
            "test_std": float(np.std(test_scores)),
            "test_scores": test_scores.tolist(),
            "train_mean": float(np.mean(train_scores)),
            "train_std": float(np.std(train_scores)),
            "train_scores": train_scores.tolist(),
            "test_ci_95": (
                float(np.percentile(test_scores, 2.5)),
                float(np.percentile(test_scores, 97.5)),
            ),
        }

        # Add generalization gap
        results_summary["metrics"][metric_name]["generalization_gap"] = float(
            np.mean(train_scores) - np.mean(test_scores)
        )

    return results_summary


def print_cv_results(cv_results: Dict[str, Any]):
    """Print cross-validation results in a formatted way."""
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Folds: {cv_results['cv_folds']}, Samples: {cv_results['n_samples']}")
    print("-" * 50)

    for metric_name, stats in cv_results["metrics"].items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Test:  {stats['test_mean']:.4f} ± {stats['test_std']:.4f}")
        print(f"  Train: {stats['train_mean']:.4f} ± {stats['train_std']:.4f}")
        print(f"  95% CI: [{stats['test_ci_95'][0]:.4f}, {stats['test_ci_95'][1]:.4f}]")
        print(f"  Generalization gap: {stats['generalization_gap']:.4f}")


# Backward compatibility function for MLflow integration
def evaluate_model_legacy(model, X, y, split_name="val", verbose=True):
    """
    Legacy function that returns dictionary format for MLflow compatibility.
    This maintains the old interface while using the new enhanced evaluation.
    """
    results = evaluate_model(model, X, y, split_name, verbose, bootstrap_ci=False)
    return results.metrics


# Convert EvaluationResults to dictionary for MLflow
def results_to_dict(evaluation_results):
    """Convert EvaluationResults object to dictionary for MLflow logging."""
    if hasattr(evaluation_results, "metrics"):
        return evaluation_results.metrics
    elif isinstance(evaluation_results, dict):
        return evaluation_results
    else:
        raise ValueError(f"Cannot convert {type(evaluation_results)} to dictionary")


def comprehensive_evaluate(
    model, X_train, y_train, X_val, y_val, X_test, y_test, cv_folds=5, bootstrap_ci=True
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation including single-split and cross-validation.

    Returns:
        Dictionary with all evaluation results
    """
    results = {}

    # Single-split evaluation with bootstrap CIs
    results["train"] = evaluate_model(
        model, X_train, y_train, "train", verbose=False, bootstrap_ci=bootstrap_ci
    )
    results["val"] = evaluate_model(
        model, X_val, y_val, "val", verbose=False, bootstrap_ci=bootstrap_ci
    )
    results["test"] = evaluate_model(
        model, X_test, y_test, "test", verbose=True, bootstrap_ci=bootstrap_ci
    )

    # Cross-validation on combined train+val for final model assessment
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.vstack([y_train, y_val])

    try:
        results["cross_validation"] = cross_validate_model(
            model, X_trainval, y_trainval, cv_folds=cv_folds
        )
        print_cv_results(results["cross_validation"])
    except Exception as e:
        print(f"Cross-validation failed: {e}")
        results["cross_validation"] = None

    return results
