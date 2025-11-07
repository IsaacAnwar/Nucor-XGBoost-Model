"""
Fetch market data (stock prices, indices, ETFs) from Yahoo Finance.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fetch_market_data(
    tickers: list,
    start_date: str = "2010-01-01",
    end_date: str = None,
    resample_to_quarter: bool = True
) -> pd.DataFrame:
    """
    Fetch daily market data and optionally resample to quarter-end.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols (e.g., ['NUE', '^GSPC', 'SLX'])
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD), defaults to today
    resample_to_quarter : bool
        If True, resample daily data to quarter-end
    
    Returns:
    --------
    pd.DataFrame
        Market data with columns for each ticker's close price and returns
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Fetching market data for {tickers} from {start_date} to {end_date}")
    
    all_data = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No data for {ticker}")
                continue
            
            # Use adjusted close for returns calculation
            all_data[f"{ticker}_Close"] = hist['Close']
            all_data[f"{ticker}_AdjClose"] = hist['Adj Close']
            all_data[f"{ticker}_Volume"] = hist['Volume']
            
            # Calculate returns
            all_data[f"{ticker}_Return"] = hist['Adj Close'].pct_change()
            all_data[f"{ticker}_Return_3M"] = hist['Adj Close'].pct_change(63)  # ~3 months
            all_data[f"{ticker}_Return_6M"] = hist['Adj Close'].pct_change(126)  # ~6 months
            
            # Calculate rolling volatility (30-day)
            all_data[f"{ticker}_Volatility"] = hist['Adj Close'].pct_change().rolling(30).std()
            
            logger.info(f"Successfully fetched {len(hist)} days of data for {ticker}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            continue
    
    if not all_data:
        logger.error("No market data fetched")
        return pd.DataFrame()
    
    # Combine into single DataFrame
    market_df = pd.DataFrame(all_data)
    
    # Resample to quarter-end if requested
    if resample_to_quarter:
        logger.info("Resampling to quarter-end")
        market_df = market_df.resample('Q').last()
        market_df.index.name = 'Date'
    
    return market_df.reset_index()


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> pd.Series:
    """
    Calculate rolling beta vs market.
    
    Parameters:
    -----------
    stock_returns : pd.Series
        Stock return series
    market_returns : pd.Series
        Market return series (e.g., S&P 500)
    
    Returns:
    --------
    pd.Series
        Rolling 252-day beta
    """
    # Align series
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    # Calculate rolling covariance and variance
    rolling_cov = aligned['stock'].rolling(252).cov(aligned['market'])
    rolling_var = aligned['market'].rolling(252).var()
    
    beta = rolling_cov / rolling_var
    return beta


def save_market_data(df: pd.DataFrame, output_dir: Path):
    """Save market data to CSV."""
    output_path = output_dir / "market_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved market data to {output_path}")

