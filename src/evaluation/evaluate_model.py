"""
Evaluate trained model and generate visualizations.
- Prediction vs actual plots
- Residual analysis
- Feature importance (XGBoost and SHAP)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import DATA_PROCESSED, MODELS_DIR
from src.modeling.train_model import (
    load_prepared_data, prepare_train_test_split, calculate_prediction_intervals
)

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_latest_model():
    """Load the most recent trained model and quantile models if available."""
    model_path = MODELS_DIR / "latest_model.pkl"
    metadata_path = MODELS_DIR / "latest_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train a model first."
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Model trained on: {metadata.get('timestamp', 'unknown')}")
    
    # Load quantile models if available
    quantile_models = None
    if metadata.get('has_prediction_intervals', False):
        quantile_models = {}
        for quantile in [0.05, 0.95]:
            q_path = MODELS_DIR / f"latest_quantile_model_{int(quantile*100)}.pkl"
            if q_path.exists():
                with open(q_path, 'rb') as f:
                    quantile_models[quantile] = pickle.load(f)
                logger.info(f"Loaded quantile model for {quantile*100}th percentile")
    
    return model, metadata, quantile_models


def plot_predictions_vs_actual(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series,
    title: str = "Predictions vs Actual",
    save_path: Path = None
):
    """Plot predicted vs actual EPS growth over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series plot
    axes[0].plot(dates, y_true, label='Actual', marker='o', linewidth=2)
    axes[0].plot(dates, y_pred, label='Predicted', marker='s', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('EPS Growth (6M)')
    axes[0].set_title(f'{title} - Time Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.6, s=50)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual EPS Growth')
    axes[1].set_ylabel('Predicted EPS Growth')
    axes[1].set_title(f'{title} - Scatter Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def plot_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series,
    save_path: Path = None
):
    """Plot residual analysis."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Residuals over time
    axes[0, 0].plot(dates, residuals, marker='o', alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals vs predicted
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals vs Predicted')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residuals plot to {save_path}")
    
    plt.show()


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 20,
    save_path: Path = None
):
    """Plot XGBoost feature importance."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importance (XGBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.show()


def plot_shap_importance(
    model,
    X: pd.DataFrame,
    top_n: int = 20,
    save_path: Path = None
):
    """Plot SHAP feature importance."""
    logger.info("Calculating SHAP values (this may take a while)...")
    
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    
    # Sample data for faster computation (use all if small dataset)
    if len(X) > 100:
        X_sample = X.sample(100, random_state=42)
    else:
        X_sample = X
    
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=top_n)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP plot to {save_path}")
    
    plt.show()
    
    return explainer, shap_values


def calculate_directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Calculate directional accuracy (percentage of correct up/down predictions).
    
    Parameters:
    -----------
    y_true : pd.Series
        True target values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary with directional accuracy metrics
    """
    # Calculate actual and predicted directions
    actual_direction = np.sign(y_true.diff().dropna())
    pred_direction = np.sign(pd.Series(y_pred).diff().dropna())
    
    # Align indices
    common_idx = actual_direction.index.intersection(pred_direction.index)
    actual_dir_aligned = actual_direction.loc[common_idx]
    pred_dir_aligned = pred_direction.loc[common_idx]
    
    # Calculate accuracy
    correct = (actual_dir_aligned == pred_dir_aligned).sum()
    total = len(actual_dir_aligned)
    accuracy = correct / total if total > 0 else 0
    
    metrics = {
        'directional_accuracy': accuracy,
        'correct_predictions': int(correct),
        'total_predictions': total
    }
    
    logger.info(f"Directional Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    
    return metrics


def plot_prediction_intervals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    intervals: pd.DataFrame,
    dates: pd.Series,
    save_path: Path = None
):
    """Plot predictions with confidence intervals."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot actual
    ax.plot(dates, y_true, label='Actual', marker='o', linewidth=2, color='black')
    
    # Plot predictions
    ax.plot(dates, y_pred, label='Predicted', marker='s', linewidth=2, alpha=0.7, color='blue')
    
    # Plot confidence intervals
    if 'lower' in intervals.columns and 'upper' in intervals.columns:
        ax.fill_between(
            dates,
            intervals['lower'],
            intervals['upper'],
            alpha=0.3,
            color='blue',
            label='90% Prediction Interval'
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('EPS Growth (6M)')
    ax.set_title('Predictions with Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction intervals plot to {save_path}")
    
    plt.show()


def calculate_quantile_metrics(
    y_true: pd.Series,
    intervals: pd.DataFrame
) -> dict:
    """
    Calculate metrics for prediction intervals.
    
    Parameters:
    -----------
    y_true : pd.Series
        True target values
    intervals : pd.DataFrame
        DataFrame with prediction, lower, upper columns
    
    Returns:
    --------
    dict
        Quantile metrics
    """
    if 'lower' not in intervals.columns or 'upper' not in intervals.columns:
        return {}
    
    # Coverage: percentage of actual values within interval
    coverage = ((y_true >= intervals['lower']) & (y_true <= intervals['upper'])).mean()
    
    # Interval width
    interval_width = (intervals['upper'] - intervals['lower']).mean()
    
    # Mean absolute interval width
    mae_width = (intervals['upper'] - intervals['lower']).abs().mean()
    
    metrics = {
        'coverage': coverage,
        'mean_interval_width': interval_width,
        'mae_interval_width': mae_width
    }
    
    logger.info(f"Interval Coverage: {coverage*100:.2f}%")
    logger.info(f"Mean Interval Width: {interval_width:.4f}")
    
    return metrics


def main():
    """Main evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Evaluating NUE EPS Growth Prediction Model")
    logger.info("=" * 60)
    
    # Load model
    model, metadata, quantile_models = load_latest_model()
    
    # Load data
    df = load_prepared_data()
    X_train, X_test, y_train, y_test, train_dates, test_dates = prepare_train_test_split(df)
    
    # Fill missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate directional accuracy
    train_dir_metrics = calculate_directional_accuracy(y_train, y_train_pred)
    test_dir_metrics = calculate_directional_accuracy(y_test, y_test_pred)
    
    # Calculate prediction intervals if available
    test_intervals = None
    if quantile_models:
        test_intervals = calculate_prediction_intervals(
            model, X_test, quantile_models, method="quantile"
        )
        quantile_metrics = calculate_quantile_metrics(y_test, test_intervals)
    
    # Create output directory
    output_dir = Path("notebooks") / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot predictions vs actual
    plot_predictions_vs_actual(
        y_test, y_test_pred, test_dates,
        title="Test Set: Predictions vs Actual",
        save_path=output_dir / "predictions_test.png"
    )
    
    # Plot residuals
    plot_residuals(
        y_test, y_test_pred, test_dates,
        save_path=output_dir / "residuals_test.png"
    )
    
    # Feature importance
    plot_feature_importance(
        model, list(X_train.columns),
        save_path=output_dir / "feature_importance.png"
    )
    
    # SHAP importance
    try:
        plot_shap_importance(
            model, X_test,
            save_path=output_dir / "shap_importance.png"
        )
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
    
    # Plot prediction intervals if available
    if test_intervals is not None:
        plot_prediction_intervals(
            y_test, y_test_pred, test_intervals, test_dates,
            save_path=output_dir / "prediction_intervals.png"
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()

