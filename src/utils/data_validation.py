"""
Data validation utilities for checking data quality and completeness.
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def validate_date_range(
    df: pd.DataFrame,
    date_col: str = 'Date',
    expected_start: str = None,
    expected_end: str = None
) -> Tuple[bool, str]:
    """
    Validate that date range is as expected.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date column
    date_col : str
        Name of date column
    expected_start : str
        Expected start date (YYYY-MM-DD)
    expected_end : str
        Expected end date (YYYY-MM-DD)
    
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, message)
    """
    if date_col not in df.columns:
        return False, f"Date column '{date_col}' not found"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    df[date_col] = pd.to_datetime(df[date_col])
    actual_start = df[date_col].min()
    actual_end = df[date_col].max()
    
    issues = []
    
    if expected_start:
        expected_start_dt = pd.to_datetime(expected_start)
        if actual_start > expected_start_dt:
            issues.append(f"Start date {actual_start} is later than expected {expected_start}")
    
    if expected_end:
        expected_end_dt = pd.to_datetime(expected_end)
        if actual_end < expected_end_dt:
            issues.append(f"End date {actual_end} is earlier than expected {expected_end}")
    
    if issues:
        return False, "; ".join(issues)
    
    return True, f"Date range valid: {actual_start} to {actual_end}"


def check_missing_critical_features(
    df: pd.DataFrame,
    critical_features: List[str]
) -> Tuple[bool, List[str]]:
    """
    Check if critical features are missing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    critical_features : List[str]
        List of critical feature names
    
    Returns:
    --------
    Tuple[bool, List[str]]
        (all_present, missing_features)
    """
    missing = [feat for feat in critical_features if feat not in df.columns]
    
    if missing:
        return False, missing
    
    return True, []


def check_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.5,
    exclude_cols: List[str] = None
) -> Dict[str, float]:
    """
    Check for columns with high percentage of missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
    threshold : float
        Threshold for missing value percentage (0.5 = 50%)
    exclude_cols : List[str]
        Columns to exclude from check
    
    Returns:
    --------
    Dict[str, float]
        Dictionary mapping column names to missing percentage
    """
    if exclude_cols is None:
        exclude_cols = []
    
    missing_pct = {}
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        pct_missing = df[col].isna().sum() / len(df)
        if pct_missing > threshold:
            missing_pct[col] = pct_missing
    
    return missing_pct


def validate_data_quality(
    df: pd.DataFrame,
    date_col: str = 'Date',
    critical_features: List[str] = None,
    expected_start: str = None,
    expected_end: str = None,
    missing_threshold: float = 0.5
) -> Dict[str, any]:
    """
    Comprehensive data quality validation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    date_col : str
        Name of date column
    critical_features : List[str]
        List of critical features
    expected_start : str
        Expected start date
    expected_end : str
        Expected end date
    missing_threshold : float
        Threshold for missing values
    
    Returns:
    --------
    Dict[str, any]
        Validation results
    """
    results = {
        'is_valid': True,
        'issues': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        results['is_valid'] = False
        results['issues'].append("DataFrame is empty")
        return results
    
    # Validate date range
    if date_col in df.columns:
        is_valid, message = validate_date_range(df, date_col, expected_start, expected_end)
        if not is_valid:
            results['issues'].append(f"Date range validation: {message}")
        else:
            results['warnings'].append(f"Date range: {message}")
    else:
        results['warnings'].append(f"Date column '{date_col}' not found")
    
    # Check critical features
    if critical_features:
        all_present, missing = check_missing_critical_features(df, critical_features)
        if not all_present:
            results['is_valid'] = False
            results['issues'].append(f"Missing critical features: {missing}")
    
    # Check missing values
    missing_pct = check_missing_values(df, threshold=missing_threshold, exclude_cols=[date_col])
    if missing_pct:
        results['warnings'].append(
            f"Columns with >{missing_threshold*100}% missing values: {list(missing_pct.keys())}"
        )
    
    # Check for duplicate dates
    if date_col in df.columns:
        duplicates = df[date_col].duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate dates")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = 0
    for col in numeric_cols:
        inf_values = np.isinf(df[col]).sum()
        if inf_values > 0:
            inf_count += inf_values
            results['warnings'].append(f"Column {col} has {inf_values} infinite values")
    
    if results['issues']:
        results['is_valid'] = False
    
    return results


def log_validation_results(results: Dict[str, any]):
    """
    Log validation results.
    
    Parameters:
    -----------
    results : Dict[str, any]
        Validation results from validate_data_quality
    """
    if results['is_valid']:
        logger.info("Data validation passed")
    else:
        logger.error("Data validation failed")
        for issue in results['issues']:
            logger.error(f"  Issue: {issue}")
    
    for warning in results['warnings']:
        logger.warning(f"  Warning: {warning}")

