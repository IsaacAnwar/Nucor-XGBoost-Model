"""
Hyperparameter tuning using Optuna with time-series cross-validation.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import HYPERPARAMETER_SEARCH_SPACE, XGBOOST_PARAMS

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5
) -> float:
    """
    Objective function for Optuna optimization.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_splits : int
        Number of CV folds
    
    Returns:
    --------
    float
        Mean MAE across CV folds
    """
    # Suggest hyperparameters
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", *HYPERPARAMETER_SEARCH_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *HYPERPARAMETER_SEARCH_SPACE["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *HYPERPARAMETER_SEARCH_SPACE["learning_rate"]),
        "subsample": trial.suggest_float("subsample", *HYPERPARAMETER_SEARCH_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *HYPERPARAMETER_SEARCH_SPACE["colsample_bytree"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *HYPERPARAMETER_SEARCH_SPACE["min_child_weight"]),
        "gamma": trial.suggest_float("gamma", *HYPERPARAMETER_SEARCH_SPACE["gamma"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *HYPERPARAMETER_SEARCH_SPACE["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *HYPERPARAMETER_SEARCH_SPACE["reg_lambda"]),
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_train_cv = X_train.iloc[train_idx]
        X_val_cv = X_train.iloc[val_idx]
        y_train_cv = y_train.iloc[train_idx]
        y_val_cv = y_train.iloc[val_idx]
        
        # Fill missing values
        X_train_cv = X_train_cv.fillna(X_train_cv.median())
        X_val_cv = X_val_cv.fillna(X_train_cv.median())
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_cv, y_train_cv)
        
        # Predict and evaluate
        y_pred = model.predict(X_val_cv)
        mae = mean_absolute_error(y_val_cv, y_pred)
        cv_scores.append(mae)
    
    return np.mean(cv_scores)


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
    study_name: str = "xgb_optimization"
) -> dict:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_trials : int
        Number of optimization trials
    n_splits : int
        Number of CV folds
    study_name : str
        Name for Optuna study
    
    Returns:
    --------
    dict
        Best hyperparameters
    """
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, n_splits),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    })
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best CV score (MAE): {study.best_value:.4f}")
    
    return best_params


def get_best_params_from_study(study_name: str = "xgb_optimization") -> dict:
    """
    Load best parameters from a previous Optuna study.
    
    Parameters:
    -----------
    study_name : str
        Name of the study
    
    Returns:
    --------
    dict
        Best hyperparameters
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=None)
        best_params = study.best_params.copy()
        best_params.update({
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        })
        return best_params
    except Exception as e:
        logger.warning(f"Could not load study {study_name}: {e}")
        logger.info("Using default parameters")
        return XGBOOST_PARAMS.copy()

