"""
Fetch macroeconomic data from FRED (Federal Reserve Economic Data).
"""
import pandas as pd
from fredapi import Fred
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import FRED_API_KEY, FRED_SERIES

logger = logging.getLogger(__name__)


def fetch_macro_data(
    start_date: str = "2010-01-01",
    end_date: str = None,
    series_ids: dict = None,
    resample_to_quarter: bool = True
) -> pd.DataFrame:
    """
    Fetch macroeconomic indicators from FRED.
    
    Parameters:
    -----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD), defaults to today
    series_ids : dict
        Dictionary mapping series IDs to descriptive names
        If None, uses default FRED_SERIES from config
    resample_to_quarter : bool
        If True, resample monthly data to quarter-end
    
    Returns:
    --------
    pd.DataFrame
        Macroeconomic indicators with date index
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if series_ids is None:
        series_ids = FRED_SERIES
    
    if not FRED_API_KEY:
        logger.error("FRED_API_KEY not found. Please set it in .env file")
        logger.info("Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return pd.DataFrame()
    
    logger.info(f"Fetching macro data from FRED from {start_date} to {end_date}")
    
    try:
        fred = Fred(api_key=FRED_API_KEY)
    except Exception as e:
        logger.error(f"Error initializing FRED client: {e}")
        return pd.DataFrame()
    
    all_data = {}
    
    for series_id, description in series_ids.items():
        try:
            logger.info(f"Fetching {series_id}: {description}")
            data = fred.get_series(series_id, start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data for {series_id}")
                continue
            
            # Use description as column name
            all_data[description] = data
            
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            continue
    
    if not all_data:
        logger.error("No macro data fetched")
        return pd.DataFrame()
    
    # Combine into single DataFrame
    macro_df = pd.DataFrame(all_data)
    macro_df.index.name = 'Date'
    
    # Resample to quarter-end if requested
    if resample_to_quarter:
        logger.info("Resampling to quarter-end")
        macro_df = macro_df.resample('QE').last()  # QE = quarter end (replaces deprecated 'Q')
    
    # Calculate year-over-year changes for key indicators
    if 'Consumer Price Index for All Urban Consumers: All Items' in macro_df.columns:
        cpi_col = 'Consumer Price Index for All Urban Consumers: All Items'
        macro_df['CPI_YoY'] = macro_df[cpi_col].pct_change(4) * 100  # 4 quarters = 1 year
    
    if 'Industrial Production Index' in macro_df.columns:
        ip_col = 'Industrial Production Index'
        macro_df['IP_YoY'] = macro_df[ip_col].pct_change(4) * 100
    
    # Calculate changes for interest rates
    if '10-Year Treasury Constant Maturity Rate' in macro_df.columns:
        rate_col = '10-Year Treasury Constant Maturity Rate'
        macro_df['10Y_Change'] = macro_df[rate_col].diff()
    
    # Calculate PMI changes (QoQ and YoY)
    if 'ISM Manufacturing: PMI Composite Index' in macro_df.columns:
        pmi_col = 'ISM Manufacturing: PMI Composite Index'
        macro_df['PMI_Change'] = macro_df[pmi_col].diff()  # Quarter-over-quarter change
        macro_df['PMI_YoY'] = macro_df[pmi_col].pct_change(4) * 100  # Year-over-year % change
    
    # Calculate changes for Durable Goods Orders
    if "Manufacturers' New Orders: Durable Goods" in macro_df.columns:
        dgo_col = "Manufacturers' New Orders: Durable Goods"
        macro_df['DGO_YoY'] = macro_df[dgo_col].pct_change(4) * 100
    
    # Calculate changes for Construction Spending
    if 'Total Construction Spending' in macro_df.columns:
        cons_col = 'Total Construction Spending'
        macro_df['Construction_YoY'] = macro_df[cons_col].pct_change(4) * 100
    
    logger.info(f"Successfully fetched {len(macro_df)} periods of macro data")
    return macro_df.reset_index()


def save_macro_data(df: pd.DataFrame, output_dir: Path):
    """Save macro data to CSV."""
    output_path = output_dir / "macro_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved macro data to {output_path}")

