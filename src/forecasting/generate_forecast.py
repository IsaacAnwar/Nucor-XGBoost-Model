"""
Generate 6-month ahead EPS growth forecast using trained model.
Output formatted for DCF comparison.
"""
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import DATA_PROCESSED, MODELS_DIR, TICKER
from src.modeling.train_model import load_prepared_data

logger = logging.getLogger(__name__)


def load_latest_model():
    """Load the most recent trained model and metadata."""
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
    
    return model, metadata


def get_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the latest quarter's features for forecasting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Prepared features dataset
    
    Returns:
    --------
    pd.DataFrame
        Latest quarter's features
    """
    # Sort by date and get latest
    df_sorted = df.sort_values('Date')
    latest = df_sorted.iloc[-1:].copy()
    
    logger.info(f"Latest quarter: {latest['Date'].values[0]}")
    
    return latest


def generate_forecast(model, features: pd.DataFrame, feature_names: list) -> dict:
    """
    Generate 6-month ahead EPS growth forecast.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained model
    features : pd.DataFrame
        Latest quarter's features
    feature_names : list
        List of feature names expected by model
    
    Returns:
    --------
    dict
        Forecast results
    """
    # Prepare features in correct order
    X_forecast = features[feature_names].copy()
    
    # Fill missing values with median (should use training median in production)
    X_forecast = X_forecast.fillna(X_forecast.median())
    
    # Generate prediction
    predicted_growth = model.predict(X_forecast)[0]
    
    logger.info(f"Predicted 6-month EPS growth: {predicted_growth:.4f} ({predicted_growth*100:.2f}%)")
    
    return {
        'predicted_growth': predicted_growth,
        'predicted_growth_pct': predicted_growth * 100,
        'forecast_date': datetime.now().strftime("%Y-%m-%d"),
        'latest_quarter': features['Date'].values[0] if 'Date' in features.columns else None
    }


def translate_to_eps_level(
    current_eps: float,
    predicted_growth: float
) -> dict:
    """
    Translate growth rate to EPS level.
    
    Parameters:
    -----------
    current_eps : float
        Current quarter EPS
    predicted_growth : float
        Predicted 6-month growth rate
    
    Returns:
    --------
    dict
        EPS projections
    """
    eps_6m = current_eps * (1 + predicted_growth)
    
    return {
        'current_eps': current_eps,
        'predicted_eps_6m': eps_6m,
        'eps_change': eps_6m - current_eps,
        'eps_change_pct': predicted_growth * 100
    }


def format_dcf_comparison(
    forecast_results: dict,
    eps_results: dict = None
) -> str:
    """
    Format forecast results for DCF comparison.
    
    Parameters:
    -----------
    forecast_results : dict
        Forecast results from generate_forecast
    eps_results : dict, optional
        EPS level results
    
    Returns:
    --------
    str
        Formatted report
    """
    report = []
    report.append("=" * 60)
    report.append("NUE 6-Month EPS Growth Forecast")
    report.append("=" * 60)
    report.append("")
    report.append(f"Forecast Date: {forecast_results['forecast_date']}")
    if forecast_results.get('latest_quarter'):
        report.append(f"Latest Quarter: {forecast_results['latest_quarter']}")
    report.append("")
    report.append("PREDICTION:")
    report.append(f"  6-Month EPS Growth Rate: {forecast_results['predicted_growth_pct']:.2f}%")
    report.append("")
    
    if eps_results:
        report.append("EPS PROJECTION:")
        report.append(f"  Current EPS: ${eps_results['current_eps']:.2f}")
        report.append(f"  Projected EPS (6M): ${eps_results['predicted_eps_6m']:.2f}")
        report.append(f"  Change: ${eps_results['eps_change']:.2f} ({eps_results['eps_change_pct']:.2f}%)")
        report.append("")
    
    report.append("=" * 60)
    report.append("")
    report.append("DCF INTEGRATION NOTES:")
    report.append("  - Compare this growth rate with your DCF's near-term earnings assumption")
    report.append("  - Use as a consistency check for your terminal growth assumptions")
    report.append("  - Consider this as a data-driven estimate of earnings acceleration")
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def save_forecast(
    forecast_results: dict,
    eps_results: dict = None,
    output_dir: Path = None
):
    """Save forecast results to JSON and text file."""
    if output_dir is None:
        output_dir = Path("notebooks") / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    results = {
        'forecast': forecast_results,
        'eps_projection': eps_results
    }
    
    json_path = output_dir / f"forecast_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved forecast JSON to {json_path}")
    
    # Save formatted report
    report = format_dcf_comparison(forecast_results, eps_results)
    txt_path = output_dir / f"forecast_{timestamp}.txt"
    with open(txt_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved forecast report to {txt_path}")
    
    # Print to console
    print("\n" + report)


def main():
    """Main forecasting pipeline."""
    logger.info("=" * 60)
    logger.info("Generating 6-Month EPS Growth Forecast for NUE")
    logger.info("=" * 60)
    
    # Load model
    model, metadata = load_latest_model()
    feature_names = metadata['feature_names']
    logger.info(f"Loaded model with {len(feature_names)} features")
    
    # Load prepared data
    df = load_prepared_data()
    
    # Get latest features
    latest_features = get_latest_features(df)
    
    # Generate forecast
    forecast_results = generate_forecast(model, latest_features, feature_names)
    
    # Get current EPS if available
    eps_results = None
    if 'EPS' in latest_features.columns:
        current_eps = latest_features['EPS'].values[0]
        if not pd.isna(current_eps):
            eps_results = translate_to_eps_level(current_eps, forecast_results['predicted_growth'])
    
    # Save and display results
    save_forecast(forecast_results, eps_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("Forecast generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()

