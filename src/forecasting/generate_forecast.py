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
from src.modeling.train_model import (
    load_prepared_data, calculate_prediction_intervals
)

logger = logging.getLogger(__name__)


def load_latest_model():
    """Load the most recent trained model, quantile models, and metadata."""
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


def generate_forecast(
    model,
    features: pd.DataFrame,
    feature_names: list,
    quantile_models: dict = None
) -> dict:
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
    
    # Generate prediction intervals if available
    intervals = None
    if quantile_models:
        intervals_df = calculate_prediction_intervals(
            model, X_forecast, quantile_models, method="quantile"
        )
        intervals = {
            'lower': intervals_df['lower'].iloc[0],
            'upper': intervals_df['upper'].iloc[0]
        }
    
    logger.info(f"Predicted 6-month EPS growth: {predicted_growth:.4f} ({predicted_growth*100:.2f}%)")
    if intervals:
        logger.info(f"90% Confidence Interval: [{intervals['lower']:.4f}, {intervals['upper']:.4f}]")
    
    result = {
        'predicted_growth': predicted_growth,
        'predicted_growth_pct': predicted_growth * 100,
        'forecast_date': datetime.now().strftime("%Y-%m-%d"),
        'latest_quarter': features['Date'].values[0] if 'Date' in features.columns else None,
        'intervals': intervals
    }
    
    return result


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


def scenario_analysis(
    model,
    base_features: pd.DataFrame,
    feature_names: list,
    scenarios: dict = None
) -> dict:
    """
    Generate forecasts under different macro scenarios.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained model
    base_features : pd.DataFrame
        Base feature set
    feature_names : list
        Feature names
    scenarios : dict
        Dictionary mapping scenario names to feature adjustments
        e.g., {'optimistic': {'PMI_Change': +5}, 'pessimistic': {'PMI_Change': -5}}
    
    Returns:
    --------
    dict
        Scenario forecasts
    """
    if scenarios is None:
        scenarios = {
            'baseline': {},
            'optimistic': {'PMI_Change': 5, 'Steel_HRC_Return_3M': 0.1},
            'pessimistic': {'PMI_Change': -5, 'Steel_HRC_Return_3M': -0.1}
        }
    
    scenario_results = {}
    X_base = base_features[feature_names].copy().fillna(base_features[feature_names].median())
    
    for scenario_name, adjustments in scenarios.items():
        X_scenario = X_base.copy()
        
        # Apply adjustments
        for feature, adjustment in adjustments.items():
            if feature in X_scenario.columns:
                X_scenario[feature] = X_scenario[feature] + adjustment
                logger.info(f"Scenario {scenario_name}: Adjusted {feature} by {adjustment}")
        
        # Generate forecast
        forecast = model.predict(X_scenario)[0]
        scenario_results[scenario_name] = {
            'forecast': forecast,
            'forecast_pct': forecast * 100,
            'adjustments': adjustments
        }
    
    return scenario_results


def sensitivity_analysis(
    model,
    base_features: pd.DataFrame,
    feature_names: list,
    key_features: list = None,
    variation: float = 0.1
) -> dict:
    """
    Perform sensitivity analysis on key features.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        Trained model
    base_features : pd.DataFrame
        Base feature set
    feature_names : list
        Feature names
    key_features : list
        Features to vary (if None, uses top important features)
    variation : float
        Percentage variation (e.g., 0.1 = ±10%)
    
    Returns:
    --------
    dict
        Sensitivity results
    """
    X_base = base_features[feature_names].copy().fillna(base_features[feature_names].median())
    base_forecast = model.predict(X_base)[0]
    
    if key_features is None:
        # Use top 5 features by importance
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        key_features = feature_importance.head(5)['feature'].tolist()
    
    sensitivity_results = {}
    
    for feature in key_features:
        if feature not in X_base.columns:
            continue
        
        base_value = X_base[feature].iloc[0]
        
        # Vary feature
        X_high = X_base.copy()
        X_low = X_base.copy()
        
        if base_value != 0:
            X_high[feature] = base_value * (1 + variation)
            X_low[feature] = base_value * (1 - variation)
        else:
            X_high[feature] = variation
            X_low[feature] = -variation
        
        forecast_high = model.predict(X_high)[0]
        forecast_low = model.predict(X_low)[0]
        
        sensitivity_results[feature] = {
            'base_value': base_value,
            'forecast_base': base_forecast,
            'forecast_high': forecast_high,
            'forecast_low': forecast_low,
            'sensitivity': (forecast_high - forecast_low) / (2 * variation) if variation > 0 else 0
        }
    
    return sensitivity_results


def format_dcf_comparison(
    forecast_results: dict,
    eps_results: dict = None,
    scenarios: dict = None,
    sensitivity: dict = None
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
    
    if forecast_results.get('intervals'):
        intervals = forecast_results['intervals']
        report.append(f"  90% Confidence Interval: [{intervals['lower']*100:.2f}%, {intervals['upper']*100:.2f}%]")
    report.append("")
    
    if eps_results:
        report.append("EPS PROJECTION:")
        report.append(f"  Current EPS: ${eps_results['current_eps']:.2f}")
        report.append(f"  Projected EPS (6M): ${eps_results['predicted_eps_6m']:.2f}")
        report.append(f"  Change: ${eps_results['eps_change']:.2f} ({eps_results['eps_change_pct']:.2f}%)")
        report.append("")
    
    if scenarios:
        report.append("SCENARIO ANALYSIS:")
        for scenario_name, scenario_data in scenarios.items():
            report.append(f"  {scenario_name.capitalize()}: {scenario_data['forecast_pct']:.2f}%")
        report.append("")
    
    if sensitivity:
        report.append("SENSITIVITY ANALYSIS (Top Features):")
        for feature, sens_data in list(sensitivity.items())[:5]:
            report.append(f"  {feature}:")
            report.append(f"    ±10% change → Forecast: [{sens_data['forecast_low']*100:.2f}%, {sens_data['forecast_high']*100:.2f}%]")
        report.append("")
    
    report.append("=" * 60)
    report.append("")
    report.append("DCF INTEGRATION NOTES:")
    report.append("  - Compare this growth rate with your DCF's near-term earnings assumption")
    report.append("  - Use as a consistency check for your terminal growth assumptions")
    report.append("  - Consider this as a data-driven estimate of earnings acceleration")
    report.append("  - Confidence intervals provide uncertainty quantification")
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)


def save_forecast(
    forecast_results: dict,
    eps_results: dict = None,
    output_dir: Path = None,
    scenarios: dict = None,
    sensitivity: dict = None
):
    """Save forecast results to JSON and text file."""
    if output_dir is None:
        output_dir = Path("notebooks") / "forecasts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    results = {
        'forecast': forecast_results,
        'eps_projection': eps_results,
        'scenarios': scenarios,
        'sensitivity': sensitivity
    }
    
    json_path = output_dir / f"forecast_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved forecast JSON to {json_path}")
    
    # Save formatted report
    report = format_dcf_comparison(forecast_results, eps_results, scenarios, sensitivity)
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
    model, metadata, quantile_models = load_latest_model()
    feature_names = metadata['feature_names']
    logger.info(f"Loaded model with {len(feature_names)} features")
    
    # Load prepared data
    df = load_prepared_data()
    
    # Get latest features
    latest_features = get_latest_features(df)
    
    # Generate forecast with confidence intervals
    forecast_results = generate_forecast(model, latest_features, feature_names, quantile_models)
    
    # Get current EPS if available
    eps_results = None
    if 'EPS' in latest_features.columns:
        current_eps = latest_features['EPS'].values[0]
        if not pd.isna(current_eps):
            eps_results = translate_to_eps_level(current_eps, forecast_results['predicted_growth'])
    
    # Scenario analysis
    scenarios = None
    try:
        scenarios = scenario_analysis(model, latest_features, feature_names)
        logger.info("Generated scenario analysis")
    except Exception as e:
        logger.warning(f"Scenario analysis failed: {e}")
    
    # Sensitivity analysis
    sensitivity = None
    try:
        sensitivity = sensitivity_analysis(model, latest_features, feature_names)
        logger.info("Generated sensitivity analysis")
    except Exception as e:
        logger.warning(f"Sensitivity analysis failed: {e}")
    
    # Save and display results
    save_forecast(forecast_results, eps_results, scenarios=scenarios, sensitivity=sensitivity)
    
    logger.info("\n" + "=" * 60)
    logger.info("Forecast generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()

