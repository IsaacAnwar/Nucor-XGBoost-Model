"""
Train XGBoost model for NUE 6-month EPS growth prediction.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import (
    DATA_PROCESSED, MODELS_DIR, TRAIN_START, TRAIN_END,
    TEST_START, TEST_END, XGBOOST_PARAMS
)

logger = logging.getLogger(__name__)


def load_prepared_data() -> pd.DataFrame:
    """Load the prepared features dataset."""
    data_path = DATA_PROCESSED / "features_ready.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Prepared features not found at {data_path}. "
            "Please run src/preprocessing/prepare_features.py first."
        )
    
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    # Handle timezone-aware dates - convert to timezone-naive
    if pd.api.types.is_datetime64_any_dtype(df['Date']):
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
    else:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
    
    logger.info(f"Loaded prepared data: {df.shape}")
    return df


def prepare_train_test_split(df: pd.DataFrame) -> tuple:
    """
    Split data chronologically into train and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, train_dates, test_dates)
    """
    # Ensure Date column is datetime type (handle timezone-aware dates)
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # If Date column is timezone-aware, convert to timezone-naive
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Convert date strings to Timestamps for comparison
    train_start = pd.to_datetime(TRAIN_START)
    train_end = pd.to_datetime(TRAIN_END)
    test_start = pd.to_datetime(TEST_START)
    test_end = pd.to_datetime(TEST_END)
    
    # Filter by date ranges
    train_mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
    test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    logger.info(f"Train set: {len(train_df)} samples ({TRAIN_START} to {TRAIN_END})")
    logger.info(f"Test set: {len(test_df)} samples ({TEST_START} to {TEST_END})")
    
    # Check if target variable exists
    if 'EPS_Growth_6M' not in df.columns:
        error_msg = (
            "Target variable 'EPS_Growth_6M' not found in dataset. "
            "This usually means EPS data was not successfully fetched from Yahoo Finance. "
            "The fundamentals data may only contain price data as a fallback. "
            "Please check:\n"
            "1. Internet connection\n"
            "2. Yahoo Finance API availability\n"
            "3. Try re-running data acquisition: python src/data_acquisition/main.py"
        )
        logger.error(error_msg)
        raise KeyError(error_msg)
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['Date', 'EPS_Growth_6M']]
    
    if len(feature_cols) == 0:
        error_msg = "No features found in dataset. Please check data preparation step."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['EPS_Growth_6M'].copy()
    train_dates = train_df['Date'].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df['EPS_Growth_6M'].copy()
    test_dates = test_df['Date'].copy()
    
    # Check if we have any valid target values
    if y_train.isna().all():
        error_msg = (
            "All target values are missing in training set. "
            "This means EPS data was not available to create the target variable. "
            "Please ensure fundamentals data includes EPS column."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Handle missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test
    
    return X_train, X_test, y_train, y_test, train_dates, test_dates


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: list = [0.05, 0.95],
    params: dict = None
) -> dict:
    """
    Train quantile regression models for prediction intervals.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    quantiles : list
        List of quantiles to predict (e.g., [0.05, 0.95] for 90% interval)
    params : dict, optional
        Base XGBoost parameters
    
    Returns:
    --------
    dict
        Dictionary mapping quantile to trained model
    """
    if params is None:
        params = XGBOOST_PARAMS.copy()
    
    logger.info(f"Training quantile models for quantiles: {quantiles}")
    quantile_models = {}
    
    for quantile in quantiles:
        q_params = params.copy()
        q_params['objective'] = f'reg:quantileerror'
        q_params['quantile_alpha'] = quantile
        
        model = xgb.XGBRegressor(**q_params)
        model.fit(X_train, y_train)
        quantile_models[quantile] = model
        
        logger.info(f"Trained quantile model for {quantile*100}th percentile")
    
    return quantile_models


def calculate_prediction_intervals(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    quantile_models: dict = None,
    method: str = "quantile"
) -> pd.DataFrame:
    """
    Calculate prediction intervals.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Main trained model
    X : pd.DataFrame
        Features
    quantile_models : dict, optional
        Dictionary of quantile models
    method : str
        Method: "quantile" or "bootstrap"
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: prediction, lower, upper
    """
    predictions = model.predict(X)
    results = pd.DataFrame({'prediction': predictions})
    
    if method == "quantile" and quantile_models:
        lower_quantile = min(quantile_models.keys())
        upper_quantile = max(quantile_models.keys())
        
        results['lower'] = quantile_models[lower_quantile].predict(X)
        results['upper'] = quantile_models[upper_quantile].predict(X)
    else:
        # Bootstrap method: use residual standard deviation
        # This is a simplified version - in practice, you'd bootstrap the residuals
        std_estimate = predictions.std() * 0.1  # Rough estimate
        results['lower'] = predictions - 1.96 * std_estimate  # 95% interval
        results['upper'] = predictions + 1.96 * std_estimate
    
    return results


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None
) -> xgb.XGBRegressor:
    """
    Train XGBoost regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_val : pd.DataFrame, optional
        Validation features
    y_val : pd.Series, optional
        Validation target
    params : dict, optional
        XGBoost parameters
    
    Returns:
    --------
    xgb.XGBRegressor
        Trained model
    """
    if params is None:
        params = XGBOOST_PARAMS.copy()
    
    logger.info("Training XGBoost model...")
    logger.info(f"Parameters: {params}")
    
    model = xgb.XGBRegressor(**params)
    
    # Fit with early stopping if validation set provided
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100
        )
    else:
        model.fit(X_train, y_train)
    
    logger.info("Model training complete")
    return model


def evaluate_model(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    set_name: str = "dataset"
) -> dict:
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained model
    X : pd.DataFrame
        Features
    y : pd.Series
        True target values
    set_name : str
        Name of dataset (for logging)
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Mean_Actual': y.mean(),
        'Mean_Predicted': y_pred.mean(),
        'Std_Actual': y.std(),
        'Std_Predicted': y_pred.std()
    }
    
    logger.info(f"\n{set_name} Metrics:")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R²:   {r2:.4f}")
    
    return metrics


def cross_validate_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> dict:
    """
    Perform time-series cross-validation.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    n_splits : int
        Number of CV folds
    
    Returns:
    --------
    dict
        Cross-validation metrics
    """
    logger.info(f"\nPerforming time-series cross-validation ({n_splits} folds)...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = {'MAE': [], 'RMSE': [], 'R2': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        model_cv = train_xgboost_model(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
        metrics_cv = evaluate_model(model_cv, X_val_cv, y_val_cv, f"Fold {fold+1}")
        
        cv_scores['MAE'].append(metrics_cv['MAE'])
        cv_scores['RMSE'].append(metrics_cv['RMSE'])
        cv_scores['R2'].append(metrics_cv['R2'])
    
    cv_summary = {
        'MAE_mean': np.mean(cv_scores['MAE']),
        'MAE_std': np.std(cv_scores['MAE']),
        'RMSE_mean': np.mean(cv_scores['RMSE']),
        'RMSE_std': np.std(cv_scores['RMSE']),
        'R2_mean': np.mean(cv_scores['R2']),
        'R2_std': np.std(cv_scores['R2'])
    }
    
    logger.info(f"\nCross-Validation Summary:")
    logger.info(f"  MAE:  {cv_summary['MAE_mean']:.4f} ± {cv_summary['MAE_std']:.4f}")
    logger.info(f"  RMSE: {cv_summary['RMSE_mean']:.4f} ± {cv_summary['RMSE_std']:.4f}")
    logger.info(f"  R²:   {cv_summary['R2_mean']:.4f} ± {cv_summary['R2_std']:.4f}")
    
    return cv_summary


def save_model(
    model: xgb.XGBRegressor,
    metrics: dict,
    feature_names: list,
    model_dir: Path = None,
    quantile_models: dict = None
):
    """
    Save trained model and metadata.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained model
    metrics : dict
        Evaluation metrics
    feature_names : list
        List of feature names
    model_dir : Path, optional
        Directory to save model
    """
    if model_dir is None:
        model_dir = MODELS_DIR
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"xgb_model_{timestamp}.pkl"
    metadata_path = model_dir / f"model_metadata_{timestamp}.json"
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to: {model_path}")
    
    # Save quantile models if provided
    quantile_model_paths = {}
    if quantile_models:
        for quantile, q_model in quantile_models.items():
            q_path = model_dir / f"quantile_model_{int(quantile*100)}_{timestamp}.pkl"
            with open(q_path, 'wb') as f:
                pickle.dump(q_model, f)
            quantile_model_paths[quantile] = str(q_path)
        logger.info(f"Saved {len(quantile_models)} quantile models")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'metrics': metrics,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'model_params': model.get_params(),
        'has_prediction_intervals': quantile_models is not None,
        'quantile_models': quantile_model_paths if quantile_models else None
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    # Save latest model reference
    latest_path = model_dir / "latest_model.pkl"
    latest_metadata_path = model_dir / "latest_metadata.json"
    
    with open(latest_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save latest quantile models
    if quantile_models:
        for quantile, q_model in quantile_models.items():
            latest_q_path = model_dir / f"latest_quantile_model_{int(quantile*100)}.pkl"
            with open(latest_q_path, 'wb') as f:
                pickle.dump(q_model, f)
    
    with open(latest_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info("Saved latest model reference")


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Training XGBoost Model for NUE EPS Growth Prediction")
    logger.info("=" * 60)
    
    # Load data
    df = load_prepared_data()
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, train_dates, test_dates = prepare_train_test_split(df)
    
    # Optional: Use last 20% of training data as validation
    val_size = int(len(X_train) * 0.2)
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    
    # Train model
    model = train_xgboost_model(X_train_fit, y_train_fit, X_val, y_val)
    
    # Train quantile models for prediction intervals
    quantile_models = None
    try:
        quantile_models = train_quantile_models(
            X_train_fit.fillna(X_train_fit.median()),
            y_train_fit,
            quantiles=[0.05, 0.95]  # 90% prediction interval
        )
        logger.info("Successfully trained quantile models for prediction intervals")
    except Exception as e:
        logger.warning(f"Could not train quantile models: {e}")
    
    # Evaluate on train and test
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Optional: Cross-validation
    # cv_metrics = cross_validate_time_series(X_train, y_train, n_splits=5)
    
    # Save model
    save_model(model, test_metrics, list(X_train.columns), quantile_models=quantile_models)
    
    logger.info("\n" + "=" * 60)
    logger.info("Model training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()

