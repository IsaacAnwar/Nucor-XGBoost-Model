"""
Fetch market data (stock prices, indices, ETFs) from Yahoo Finance.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.technical_indicators import add_technical_indicators

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
            # Handle different column name variations
            if 'Adj Close' in hist.columns:
                adj_close_col = 'Adj Close'
            elif 'Adj. Close' in hist.columns:
                adj_close_col = 'Adj. Close'
            else:
                adj_close_col = 'Close'  # Fallback to Close if Adj Close not available
                logger.warning(f"Using 'Close' instead of 'Adj Close' for {ticker}")
            
            all_data[f"{ticker}_Close"] = hist['Close']
            all_data[f"{ticker}_AdjClose"] = hist[adj_close_col]
            if 'Volume' in hist.columns:
                all_data[f"{ticker}_Volume"] = hist['Volume']
            
            # Calculate returns
            all_data[f"{ticker}_Return"] = hist[adj_close_col].pct_change()
            all_data[f"{ticker}_Return_3M"] = hist[adj_close_col].pct_change(63)  # ~3 months
            all_data[f"{ticker}_Return_6M"] = hist[adj_close_col].pct_change(126)  # ~6 months
            
            # Calculate rolling volatility (30-day)
            all_data[f"{ticker}_Volatility"] = hist[adj_close_col].pct_change().rolling(30).std()
            
            logger.info(f"Successfully fetched {len(hist)} days of data for {ticker}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            continue
    
    if not all_data:
        logger.error("No market data fetched")
        return pd.DataFrame()
    
    # Combine into single DataFrame
    market_df = pd.DataFrame(all_data)
    
    # Add technical indicators for each ticker before resampling
    if not resample_to_quarter:  # Only add technical indicators to daily data
        for ticker in tickers:
            close_col = f"{ticker}_Close"
            if close_col in market_df.columns:
                try:
                    market_df = add_technical_indicators(
                        market_df,
                        price_col=close_col,
                        prefix=f"{ticker}_"
                    )
                except Exception as e:
                    logger.warning(f"Error adding technical indicators for {ticker}: {e}")
    
    # Calculate beta if we have both stock and market data
    # Find stock ticker (first ticker that's not market index)
    stock_ticker = None
    market_ticker = None
    
    for ticker in tickers:
        if ticker.startswith('^') or ticker in ['SPY', 'SPX']:
            market_ticker = ticker
        elif stock_ticker is None:
            stock_ticker = ticker
    
    if stock_ticker and market_ticker:
        stock_return_col = f"{stock_ticker}_Return"
        market_return_col = f"{market_ticker}_Return"
        
        if stock_return_col in market_df.columns and market_return_col in market_df.columns:
            logger.info(f"Calculating rolling beta for {stock_ticker} vs {market_ticker}")
            try:
                beta_series = calculate_beta(
                    market_df[stock_return_col],
                    market_df[market_return_col]
                )
                market_df[f"{stock_ticker}_Beta"] = beta_series
                logger.info("Beta calculation complete")
            except Exception as e:
                logger.warning(f"Error calculating beta: {e}")
    
    # Resample to quarter-end if requested
    if resample_to_quarter:
        logger.info("Resampling to quarter-end")
        market_df = market_df.resample('QE').last()  # QE = quarter end (replaces deprecated 'Q')
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

