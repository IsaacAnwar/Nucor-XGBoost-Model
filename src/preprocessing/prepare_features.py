"""
Prepare features for XGBoost model training.
- Merge all data sources
- Create lagged features
- Engineer derived features
- Create target variable (6-month EPS growth)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import DATA_RAW, DATA_PROCESSED, TICKER

logger = logging.getLogger(__name__)


def load_latest_data(data_dir: Path) -> tuple:
    """
    Load the most recent data snapshot from data/raw/.
    
    Returns:
    --------
    tuple
        (fundamentals_df, market_df, macro_df, commodity_df, peer_df)
    """
    # Find latest snapshot directory
    snapshot_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != ".gitkeep"]
    
    if not snapshot_dirs:
        raise FileNotFoundError(f"No data snapshots found in {data_dir}")
    
    latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"Loading data from snapshot: {latest_snapshot.name}")
    
    # Load each dataset
    fundamentals_path = latest_snapshot / f"{TICKER}_fundamentals.csv"
    market_path = latest_snapshot / "market_data.csv"
    macro_path = latest_snapshot / "macro_data.csv"
    commodity_path = latest_snapshot / "commodity_data.csv"
    peer_path = latest_snapshot / "peer_data.csv"
    
    fundamentals = pd.DataFrame()
    market = pd.DataFrame()
    macro = pd.DataFrame()
    commodity = pd.DataFrame()
    peer = pd.DataFrame()
    
    if fundamentals_path.exists():
        fundamentals = pd.read_csv(fundamentals_path, parse_dates=['Date'])
    else:
        logger.warning(f"Fundamentals file not found: {fundamentals_path}")
    
    if market_path.exists():
        market = pd.read_csv(market_path, parse_dates=['Date'])
    else:
        logger.warning(f"Market data file not found: {market_path}")
    
    if macro_path.exists():
        macro = pd.read_csv(macro_path, parse_dates=['Date'])
    else:
        logger.warning(f"Macro data file not found: {macro_path}")
    
    if commodity_path.exists():
        commodity = pd.read_csv(commodity_path, parse_dates=['Date'])
    else:
        logger.warning(f"Commodity data file not found: {commodity_path}")
    
    if peer_path.exists():
        peer = pd.read_csv(peer_path, parse_dates=['Date'])
    else:
        logger.warning(f"Peer data file not found: {peer_path}")
    
    return fundamentals, market, macro, commodity, peer


def create_target_variable(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Create 6-month ahead EPS growth target variable.
    
    Formula: (EPS_{t+2} - EPS_t) / |EPS_t|
    
    Parameters:
    -----------
    fundamentals : pd.DataFrame
        Fundamentals data with EPS column
    
    Returns:
    --------
    pd.DataFrame
        Fundamentals with added 'EPS_Growth_6M' column
    """
    if 'EPS' not in fundamentals.columns:
        logger.error("EPS column not found in fundamentals")
        return fundamentals
    
    fundamentals = fundamentals.sort_values('Date').copy()
    
    # Calculate 2-quarter ahead EPS growth
    fundamentals['EPS_Growth_6M'] = (
        fundamentals['EPS'].shift(-2) - fundamentals['EPS']
    ) / fundamentals['EPS'].abs()
    
    # Replace inf and NaN values
    fundamentals['EPS_Growth_6M'] = fundamentals['EPS_Growth_6M'].replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Created target variable: {fundamentals['EPS_Growth_6M'].notna().sum()} valid observations")
    
    return fundamentals


def engineer_fundamental_features(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived fundamental features.
    
    Features:
    - Revenue growth (QoQ, YoY)
    - Margin trends
    - Leverage ratios
    - R&D intensity (if available)
    """
    fundamentals = fundamentals.sort_values('Date').copy()
    
    # Revenue growth
    if 'Revenue' in fundamentals.columns:
        fundamentals['Revenue_Growth_QoQ'] = fundamentals['Revenue'].pct_change(fill_method=None)
        fundamentals['Revenue_Growth_YoY'] = fundamentals['Revenue'].pct_change(4, fill_method=None)
        fundamentals['Revenue_Growth_2Q'] = fundamentals['Revenue'].pct_change(2, fill_method=None)
    
    # Margin trends
    if 'GrossMargin' in fundamentals.columns:
        fundamentals['GrossMargin_Change'] = fundamentals['GrossMargin'].diff()
    if 'OperatingMargin' in fundamentals.columns:
        fundamentals['OperatingMargin_Change'] = fundamentals['OperatingMargin'].diff()
    if 'NetMargin' in fundamentals.columns:
        fundamentals['NetMargin_Change'] = fundamentals['NetMargin'].diff()
    
    # Leverage and financial health
    if 'TotalDebt' in fundamentals.columns and 'TotalEquity' in fundamentals.columns:
        fundamentals['DebtToEquity'] = fundamentals['TotalDebt'] / fundamentals['TotalEquity']
        fundamentals['DebtToEquity_Change'] = fundamentals['DebtToEquity'].diff()
    
    if 'TotalAssets' in fundamentals.columns and 'TotalEquity' in fundamentals.columns:
        fundamentals['EquityRatio'] = fundamentals['TotalEquity'] / fundamentals['TotalAssets']
    
    # EPS trends
    if 'EPS' in fundamentals.columns:
        fundamentals['EPS_Growth_QoQ'] = fundamentals['EPS'].pct_change(fill_method=None)
        fundamentals['EPS_Growth_YoY'] = fundamentals['EPS'].pct_change(4, fill_method=None)
    
    return fundamentals


def engineer_market_features(market: pd.DataFrame, ticker: str = "NUE") -> pd.DataFrame:
    """
    Create derived market features.
    
    Features:
    - Price returns (3M, 6M)
    - Volatility trends
    - Beta vs S&P 500
    - Relative strength vs market
    """
    market = market.sort_values('Date').copy()
    
    # Relative strength vs S&P 500
    if f"{ticker}_Return_3M" in market.columns and "^GSPC_Return_3M" in market.columns:
        market['Relative_Strength_3M'] = (
            market[f"{ticker}_Return_3M"] - market["^GSPC_Return_3M"]
        )
    if f"{ticker}_Return_6M" in market.columns and "^GSPC_Return_6M" in market.columns:
        market['Relative_Strength_6M'] = (
            market[f"{ticker}_Return_6M"] - market["^GSPC_Return_6M"]
        )
    
    # Volatility trends
    vol_col = f"{ticker}_Volatility"
    if vol_col in market.columns:
        market[f"{ticker}_Volatility_Change"] = market[vol_col].diff()
        market[f"{ticker}_Volatility_MA"] = market[vol_col].rolling(4).mean()
    
    # Volume trends (if available)
    vol_col = f"{ticker}_Volume"
    if vol_col in market.columns:
        market[f"{ticker}_Volume_MA"] = market[vol_col].rolling(4).mean()
        market[f"{ticker}_Volume_Ratio"] = market[vol_col] / market[f"{ticker}_Volume_MA"]
    return market


def engineer_commodity_features(commodity: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived commodity features.
    
    Features:
    - Commodity price returns
    - Steel price momentum
    - Energy cost indicators
    """
    commodity = commodity.sort_values('Date').copy()
    
    # Additional momentum features
    if 'Steel_HRC_Price' in commodity.columns:
        commodity['Steel_HRC_Momentum'] = commodity['Steel_HRC_Price'].pct_change(2)  # 2-quarter momentum
    
    if 'Oil_WTI_Price' in commodity.columns:
        commodity['Oil_WTI_Momentum'] = commodity['Oil_WTI_Price'].pct_change(2)
    
    return commodity


def engineer_peer_features(fundamentals: pd.DataFrame, peer: pd.DataFrame) -> pd.DataFrame:
    """
    Create relative metrics comparing NUE to peer averages.
    
    Features:
    - Revenue growth vs peers
    - Margin vs peers
    - EPS growth vs peers
    """
    if peer.empty:
        logger.warning("Peer data is empty, skipping peer feature engineering")
        return fundamentals
    
    fundamentals = fundamentals.sort_values('Date').copy()
    peer = peer.sort_values('Date').copy()
    
    # Merge peer data
    merged = fundamentals.merge(
        peer,
        on='Date',
        how='left',
        suffixes=('', '_peer')
    )
    
    # Calculate relative metrics
    if 'Revenue' in merged.columns and 'Peer_Revenue_Mean' in merged.columns:
        merged['Revenue_vs_Peer'] = merged['Revenue'] / merged['Peer_Revenue_Mean']
        merged['Revenue_Growth_vs_Peer'] = (
            merged.get('Revenue_Growth_QoQ', 0) - merged.get('Peer_Revenue_Growth_QoQ_Mean', 0)
        )
    
    if 'EPS' in merged.columns and 'Peer_EPS_Mean' in merged.columns:
        merged['EPS_vs_Peer'] = merged['EPS'] / merged['Peer_EPS_Mean']
        merged['EPS_Growth_vs_Peer'] = (
            merged.get('EPS_Growth_QoQ', 0) - merged.get('Peer_EPS_Growth_QoQ_Mean', 0)
        )
    
    if 'GrossMargin' in merged.columns and 'Peer_GrossMargin_Mean' in merged.columns:
        merged['GrossMargin_vs_Peer'] = merged['GrossMargin'] - merged['Peer_GrossMargin_Mean']
    
    if 'OperatingMargin' in merged.columns and 'Peer_OperatingMargin_Mean' in merged.columns:
        merged['OperatingMargin_vs_Peer'] = merged['OperatingMargin'] - merged['Peer_OperatingMargin_Mean']
    
    return merged


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between key variables.
    
    Features:
    - PMI × Steel Prices
    - PMI × Revenue Growth
    - Steel Prices × Revenue
    """
    df = df.copy()
    
    # PMI × Steel Prices interaction
    pmi_col = None
    steel_col = None
    
    for col in df.columns:
        if 'PMI' in col and 'Change' not in col and 'YoY' not in col:
            pmi_col = col
        if 'Steel_HRC' in col and 'Price' in col and 'Return' not in col and 'YoY' not in col:
            steel_col = col
    
    if pmi_col and steel_col:
        df['PMI_Steel_Interaction'] = df[pmi_col] * df[steel_col]
        logger.info("Added PMI × Steel Prices interaction")
    
    # PMI × Revenue Growth
    if pmi_col and 'Revenue_Growth_QoQ' in df.columns:
        df['PMI_RevenueGrowth_Interaction'] = df[pmi_col] * df['Revenue_Growth_QoQ']
    
    # Steel Prices × Revenue
    if steel_col and 'Revenue' in df.columns:
        df['Steel_Revenue_Interaction'] = df[steel_col] * df['Revenue']
    
    # Seasonal features (quarter of year)
    if 'Date' in df.columns:
        df['Quarter'] = pd.to_datetime(df['Date']).dt.quarter
        # One-hot encode quarters
        for q in [1, 2, 3, 4]:
            df[f'Q{q}'] = (df['Quarter'] == q).astype(int)
    
    return df


def merge_all_data(
    fundamentals: pd.DataFrame,
    market: pd.DataFrame,
    macro: pd.DataFrame,
    commodity: pd.DataFrame = None,
    peer: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Merge all data sources on Date (quarter-end).
    
    Parameters:
    -----------
    fundamentals : pd.DataFrame
        Fundamental data
    market : pd.DataFrame
        Market data
    macro : pd.DataFrame
        Macroeconomic data
    
    Returns:
    --------
    pd.DataFrame
        Merged dataset with all features
    """
    logger.info("Merging all data sources...")
    
    # Start with fundamentals as base
    if fundamentals.empty:
        logger.error("Fundamentals data is empty")
        return pd.DataFrame()
    
    # Ensure all Date columns are datetime type before merging
    if 'Date' in fundamentals.columns:
        if not pd.api.types.is_datetime64_any_dtype(fundamentals['Date']):
            fundamentals['Date'] = pd.to_datetime(fundamentals['Date'], utc=True)
        if fundamentals['Date'].dt.tz is not None:
            fundamentals['Date'] = fundamentals['Date'].dt.tz_localize(None)
    
    merged = fundamentals.copy()
    
    # Merge market data
    if not market.empty:
        if 'Date' in market.columns:
            if not pd.api.types.is_datetime64_any_dtype(market['Date']):
                market['Date'] = pd.to_datetime(market['Date'], utc=True)
            if market['Date'].dt.tz is not None:
                market['Date'] = market['Date'].dt.tz_localize(None)
        
        merged = merged.merge(
            market,
            on='Date',
            how='left',
            suffixes=('', '_market')
        )
        logger.info(f"Merged market data: {merged.shape}")
    else:
        logger.warning("Market data is empty, skipping merge")
    
    # Merge macro data
    if not macro.empty:
        if 'Date' in macro.columns:
            if not pd.api.types.is_datetime64_any_dtype(macro['Date']):
                macro['Date'] = pd.to_datetime(macro['Date'], utc=True)
            if macro['Date'].dt.tz is not None:
                macro['Date'] = macro['Date'].dt.tz_localize(None)
        
        merged = merged.merge(
            macro,
            on='Date',
            how='left',
            suffixes=('', '_macro')
        )
        logger.info(f"Merged macro data: {merged.shape}")
    else:
        logger.warning("Macro data is empty, skipping merge")
    
    # Merge commodity data
    if commodity is not None and not commodity.empty:
        if 'Date' in commodity.columns:
            if not pd.api.types.is_datetime64_any_dtype(commodity['Date']):
                commodity['Date'] = pd.to_datetime(commodity['Date'], utc=True)
            if commodity['Date'].dt.tz is not None:
                commodity['Date'] = commodity['Date'].dt.tz_localize(None)
        
        merged = merged.merge(
            commodity,
            on='Date',
            how='left',
            suffixes=('', '_commodity')
        )
        logger.info(f"Merged commodity data: {merged.shape}")
    else:
        logger.warning("Commodity data is empty, skipping merge")
    
    # Merge peer data
    if peer is not None and not peer.empty:
        if 'Date' in peer.columns:
            if not pd.api.types.is_datetime64_any_dtype(peer['Date']):
                peer['Date'] = pd.to_datetime(peer['Date'], utc=True)
            if peer['Date'].dt.tz is not None:
                peer['Date'] = peer['Date'].dt.tz_localize(None)
        
        merged = merged.merge(
            peer,
            on='Date',
            how='left',
            suffixes=('', '_peer')
        )
        logger.info(f"Merged peer data: {merged.shape}")
    else:
        logger.warning("Peer data is empty, skipping merge")
    
    # Sort by date
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Final merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    
    return merged


def lag_features(df: pd.DataFrame, target_col: str = 'EPS_Growth_6M') -> pd.DataFrame:
    """
    Lag all explanatory features by one quarter to avoid look-ahead bias.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset
    target_col : str
        Name of target variable (not lagged)
    
    Returns:
    --------
    pd.DataFrame
        Dataset with lagged features
    """
    logger.info("Lagging features by one quarter...")
    
    df = df.sort_values('Date').copy()
    
    # Identify feature columns (exclude Date and target)
    feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
    
    # Create lagged versions
    for col in feature_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
    
    # Drop original non-lagged features (keep target as is)
    cols_to_drop = [col for col in feature_cols if f"{col}_lag1" in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    logger.info(f"Created {len([c for c in df.columns if '_lag1' in c])} lagged features")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Handle missing values
    - Winsorize outliers
    - Remove rows with missing target
    """
    logger.info("Cleaning data...")
    
    initial_rows = len(df)
    
    # Remove rows where target is missing
    if 'EPS_Growth_6M' in df.columns:
        df = df[df['EPS_Growth_6M'].notna()].copy()
        logger.info(f"Removed {initial_rows - len(df)} rows with missing target")
    
    # Forward fill missing values for some columns (time series continuity)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    # Winsorize extreme outliers in growth/ratio variables
    growth_cols = [col for col in df.columns if 'Growth' in col or 'Change' in col or 'Return' in col]
    for col in growth_cols:
        if col in df.columns:
            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q01, upper=q99)
    
    # Drop columns with too many missing values (>50%)
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} columns with >50% missing values")
    
    logger.info(f"Final cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


def prepare_features() -> pd.DataFrame:
    """
    Main function to prepare all features.
    
    Returns:
    --------
    pd.DataFrame
        Prepared dataset ready for modeling
    """
    logger.info("=" * 60)
    logger.info("Starting feature preparation")
    logger.info("=" * 60)
    
    # Load data
    fundamentals, market, macro, commodity, peer = load_latest_data(DATA_RAW)
    
    # Create target variable
    fundamentals = create_target_variable(fundamentals)
    
    # Engineer features
    fundamentals = engineer_fundamental_features(fundamentals)
    market = engineer_market_features(market, TICKER)
    
    # Engineer commodity features
    if not commodity.empty:
        commodity = engineer_commodity_features(commodity)
    
    # Engineer peer relative features
    fundamentals = engineer_peer_features(fundamentals, peer)
    
    # Merge all data
    merged = merge_all_data(fundamentals, market, macro, commodity, peer)
    
    # Add interaction features
    merged = add_interaction_features(merged)
    
    # Lag features
    merged = lag_features(merged)
    
    # Clean data
    merged = clean_data(merged)
    
    # Save processed data
    output_path = DATA_PROCESSED / "features_ready.csv"
    merged.to_csv(output_path, index=False)
    logger.info(f"\nSaved prepared features to: {output_path}")
    
    return merged


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = prepare_features()
    logger.info(f"\nFeature preparation complete!")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

