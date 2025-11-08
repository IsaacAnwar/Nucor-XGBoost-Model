"""
Technical indicators for market data analysis.
Includes moving averages, RSI, MACD, etc.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_moving_average(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate simple moving average.
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    window : int
        Window size (e.g., 50 for 50-day MA)
    
    Returns:
    --------
    pd.Series
        Moving average series
    """
    return series.rolling(window=window).mean()


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate exponential moving average.
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    window : int
        Window size
    
    Returns:
    --------
    pd.Series
        EMA series
    """
    return series.ewm(span=window, adjust=False).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    window : int
        Window size (default 14)
    
    Returns:
    --------
    pd.Series
        RSI values (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Parameters:
    -----------
    series : pd.Series
        Price series
    fast : int
        Fast EMA period (default 12)
    slow : int
        Slow EMA period (default 26)
    signal : int
        Signal line EMA period (default 9)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: MACD, Signal, Histogram
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    result = pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Histogram': histogram
    })
    
    return result


def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str,
    prefix: str = ""
) -> pd.DataFrame:
    """
    Add technical indicators to a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    price_col : str
        Name of the price column
    prefix : str
        Prefix for indicator column names
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    if price_col not in df.columns:
        logger.warning(f"Price column {price_col} not found")
        return df
    
    result_df = df.copy()
    price_series = df[price_col]
    
    # Moving averages
    result_df[f"{prefix}MA_50"] = calculate_moving_average(price_series, 50)
    result_df[f"{prefix}MA_200"] = calculate_moving_average(price_series, 200)
    
    # EMA
    result_df[f"{prefix}EMA_12"] = calculate_ema(price_series, 12)
    result_df[f"{prefix}EMA_26"] = calculate_ema(price_series, 26)
    
    # RSI
    result_df[f"{prefix}RSI"] = calculate_rsi(price_series, 14)
    
    # MACD
    macd_data = calculate_macd(price_series)
    for col in macd_data.columns:
        result_df[f"{prefix}{col}"] = macd_data[col]
    
    # Price relative to moving averages
    result_df[f"{prefix}Price_MA50_Ratio"] = price_series / result_df[f"{prefix}MA_50"]
    result_df[f"{prefix}Price_MA200_Ratio"] = price_series / result_df[f"{prefix}MA_200"]
    
    logger.info(f"Added technical indicators with prefix '{prefix}'")
    
    return result_df

