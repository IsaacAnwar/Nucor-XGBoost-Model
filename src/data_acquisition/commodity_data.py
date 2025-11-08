"""
Fetch commodity data (steel prices, iron ore, energy) from Yahoo Finance and other sources.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_commodity_data(
    start_date: str = "2010-01-01",
    end_date: str = None,
    resample_to_quarter: bool = True
) -> pd.DataFrame:
    """
    Fetch commodity price data for steel, iron ore, and energy.
    
    Parameters:
    -----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD), defaults to today
    resample_to_quarter : bool
        If True, resample daily data to quarter-end
    
    Returns:
    --------
    pd.DataFrame
        Commodity data with date index
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Fetching commodity data from {start_date} to {end_date}")
    
    all_data = {}
    
    # Steel prices - Hot Rolled Coil futures
    steel_tickers = {
        "HRC=F": "Steel_HRC_Price",  # Hot Rolled Coil futures
    }
    
    # Iron ore - try to find available tickers
    # Note: Direct iron ore futures may not be available on Yahoo Finance
    # Alternative: Use steel-related ETFs or indices
    iron_ore_tickers = {
        # Iron ore is often tracked via commodity indices or steel-related instruments
        # We'll use steel ETF as a proxy if direct iron ore data isn't available
    }
    
    # Energy prices
    energy_tickers = {
        "CL=F": "Oil_WTI_Price",  # Crude Oil WTI futures
        "NG=F": "Gas_NaturalGas_Price",  # Natural Gas futures
    }
    
    # Combine all tickers
    all_tickers = {**steel_tickers, **iron_ore_tickers, **energy_tickers}
    
    for ticker, col_name in all_tickers.items():
        try:
            logger.info(f"Fetching {ticker} ({col_name})...")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No data for {ticker}")
                continue
            
            # Use adjusted close if available, otherwise close
            if 'Adj Close' in hist.columns:
                price_col = hist['Adj Close']
            elif 'Adj. Close' in hist.columns:
                price_col = hist['Adj. Close']
            else:
                price_col = hist['Close']
                logger.warning(f"Using 'Close' instead of 'Adj Close' for {ticker}")
            
            all_data[col_name] = price_col
            
            # Calculate returns
            all_data[f"{col_name}_Return"] = price_col.pct_change()
            all_data[f"{col_name}_Return_3M"] = price_col.pct_change(63)  # ~3 months
            all_data[f"{col_name}_Return_6M"] = price_col.pct_change(126)  # ~6 months
            
            # Calculate rolling volatility
            all_data[f"{col_name}_Volatility"] = price_col.pct_change().rolling(30).std()
            
            logger.info(f"Successfully fetched {len(hist)} days of data for {ticker}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            continue
    
    # Try to get iron ore data from alternative sources
    # Some commodity data might be available via FRED or other sources
    # For now, we'll note that iron ore data may need manual sourcing
    
    if not all_data:
        logger.error("No commodity data fetched")
        return pd.DataFrame()
    
    # Combine into single DataFrame
    commodity_df = pd.DataFrame(all_data)
    commodity_df.index.name = 'Date'
    
    # Resample to quarter-end if requested
    if resample_to_quarter:
        logger.info("Resampling to quarter-end")
        commodity_df = commodity_df.resample('QE').last()  # QE = quarter end
    
    # Calculate year-over-year changes for key commodities
    if 'Steel_HRC_Price' in commodity_df.columns:
        steel_col = 'Steel_HRC_Price'
        commodity_df['Steel_HRC_YoY'] = commodity_df[steel_col].pct_change(4) * 100
    
    if 'Oil_WTI_Price' in commodity_df.columns:
        oil_col = 'Oil_WTI_Price'
        commodity_df['Oil_WTI_YoY'] = commodity_df[oil_col].pct_change(4) * 100
    
    if 'Gas_NaturalGas_Price' in commodity_df.columns:
        gas_col = 'Gas_NaturalGas_Price'
        commodity_df['Gas_NaturalGas_YoY'] = commodity_df[gas_col].pct_change(4) * 100
    
    logger.info(f"Successfully fetched {len(commodity_df)} periods of commodity data")
    return commodity_df.reset_index()


def save_commodity_data(df: pd.DataFrame, output_dir: Path):
    """Save commodity data to CSV."""
    output_path = output_dir / "commodity_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved commodity data to {output_path}")

