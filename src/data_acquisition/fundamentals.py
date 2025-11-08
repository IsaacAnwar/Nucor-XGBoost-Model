"""
Fetch Nucor (NUE) fundamental data from Yahoo Finance and FinancialModelingPrep.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from pathlib import Path
import requests
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import FMP_API_KEY

logger = logging.getLogger(__name__)


def fetch_fundamentals_fmp(ticker: str = "NUE", start_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Fetch quarterly fundamental data using FinancialModelingPrep API.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date for data collection (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame
        Quarterly fundamental metrics
    """
    if not FMP_API_KEY:
        logger.warning("FMP_API_KEY not found. Skipping FMP data fetch.")
        return pd.DataFrame()
    
    logger.info(f"Fetching fundamentals for {ticker} from FinancialModelingPrep")
    
    try:
        # Fetch income statement
        income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
        params = {
            'period': 'quarter',
            'apikey': FMP_API_KEY,
            'limit': 100  # Get last 100 quarters
        }
        
        response = requests.get(income_url, params=params, timeout=10)
        response.raise_for_status()
        income_data = response.json()
        
        if not income_data or len(income_data) == 0:
            logger.warning(f"No income statement data from FMP for {ticker}")
            return pd.DataFrame()
        
        # Fetch balance sheet
        balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}"
        balance_response = requests.get(balance_url, params=params, timeout=10)
        balance_response.raise_for_status()
        balance_data = balance_response.json()
        
        # Convert to DataFrames
        income_df = pd.DataFrame(income_data)
        balance_df = pd.DataFrame(balance_data) if balance_data else pd.DataFrame()
        
        # Process income statement
        if income_df.empty:
            return pd.DataFrame()
        
        # Convert date column
        income_df['date'] = pd.to_datetime(income_df['date'])
        income_df = income_df.sort_values('date')
        
        # Filter by start date
        income_df = income_df[income_df['date'] >= start_date]
        
        # Create fundamentals DataFrame
        fundamentals = pd.DataFrame()
        fundamentals['Date'] = income_df['date'].values
        
        # Map FMP fields to our standard fields
        field_mapping = {
            'revenue': 'Revenue',
            'grossProfit': 'GrossProfit',
            'operatingIncome': 'OperatingIncome',
            'netIncome': 'NetIncome',
            'eps': 'EPS',
        }
        
        for fmp_field, our_field in field_mapping.items():
            if fmp_field in income_df.columns:
                fundamentals[our_field] = income_df[fmp_field].values
        
        # Process balance sheet if available
        if not balance_df.empty:
            balance_df['date'] = pd.to_datetime(balance_df['date'])
            balance_df = balance_df.sort_values('date')
            balance_df = balance_df[balance_df['date'] >= start_date]
            
            # Merge balance sheet data
            for date in fundamentals['Date']:
                balance_row = balance_df[balance_df['date'] == date]
                if not balance_row.empty:
                    idx = fundamentals[fundamentals['Date'] == date].index[0]
                    if 'totalDebt' in balance_row.columns:
                        fundamentals.loc[idx, 'TotalDebt'] = balance_row['totalDebt'].values[0]
                    if 'totalStockholdersEquity' in balance_row.columns:
                        fundamentals.loc[idx, 'TotalEquity'] = balance_row['totalStockholdersEquity'].values[0]
                    if 'totalAssets' in balance_row.columns:
                        fundamentals.loc[idx, 'TotalAssets'] = balance_row['totalAssets'].values[0]
        
        # Calculate derived metrics
        if 'Revenue' in fundamentals.columns and 'GrossProfit' in fundamentals.columns:
            fundamentals['GrossMargin'] = (fundamentals['GrossProfit'] / fundamentals['Revenue']) * 100
        if 'Revenue' in fundamentals.columns and 'OperatingIncome' in fundamentals.columns:
            fundamentals['OperatingMargin'] = (fundamentals['OperatingIncome'] / fundamentals['Revenue']) * 100
        if 'Revenue' in fundamentals.columns and 'NetIncome' in fundamentals.columns:
            fundamentals['NetMargin'] = (fundamentals['NetIncome'] / fundamentals['Revenue']) * 100
        
        if 'TotalDebt' in fundamentals.columns and 'TotalEquity' in fundamentals.columns:
            fundamentals['DebtToEquity'] = fundamentals['TotalDebt'] / fundamentals['TotalEquity']
        
        logger.info(f"Successfully fetched {len(fundamentals)} quarters of fundamental data from FMP")
        return fundamentals
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from FMP API: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error in FMP fetch: {e}")
        return pd.DataFrame()


def fetch_fundamentals(ticker: str = "NUE", start_date: str = "2010-01-01") -> pd.DataFrame:
    """
    Fetch quarterly fundamental data for a ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date for data collection (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame
        Quarterly fundamental metrics with columns:
        - Date (quarter-end)
        - Revenue
        - GrossProfit
        - OperatingIncome
        - NetIncome
        - EPS
        - TotalDebt
        - TotalEquity
        - TotalAssets
    """
    logger.info(f"Fetching fundamentals for {ticker} from {start_date}")
    
    # Try FMP first if API key is available
    if FMP_API_KEY:
        logger.info("Attempting to fetch from FinancialModelingPrep API...")
        fmp_data = fetch_fundamentals_fmp(ticker, start_date)
        if not fmp_data.empty and 'EPS' in fmp_data.columns:
            logger.info("Successfully fetched fundamentals from FMP")
            return fmp_data
        else:
            logger.info("FMP data incomplete, falling back to Yahoo Finance")
    
    # Fallback to Yahoo Finance
    stock = yf.Ticker(ticker)
    
    # Get quarterly financials
    try:
        quarterly_financials = stock.quarterly_financials
        quarterly_balance = stock.quarterly_balance_sheet
        quarterly_cashflow = stock.quarterly_cashflow
        
        # Get earnings data (deprecated, try income statement instead)
        try:
            earnings = stock.quarterly_earnings
        except:
            earnings = None
        
        if quarterly_financials.empty:
            logger.warning(f"Insufficient data for {ticker}, trying alternative method")
            return _fetch_alternative_fundamentals(ticker, start_date)
        
        # Combine data
        fundamentals = pd.DataFrame()
        fundamentals['Date'] = pd.to_datetime(quarterly_financials.columns)
        fundamentals = fundamentals.set_index('Date').sort_index()
        
        # Income statement metrics
        if 'Total Revenue' in quarterly_financials.index:
            fundamentals['Revenue'] = quarterly_financials.loc['Total Revenue'].values
        if 'Gross Profit' in quarterly_financials.index:
            fundamentals['GrossProfit'] = quarterly_financials.loc['Gross Profit'].values
        if 'Operating Income' in quarterly_financials.index:
            fundamentals['OperatingIncome'] = quarterly_financials.loc['Operating Income'].values
        if 'Net Income' in quarterly_financials.index:
            fundamentals['NetIncome'] = quarterly_financials.loc['Net Income'].values
        
        # Balance sheet metrics
        if not quarterly_balance.empty:
            if 'Total Debt' in quarterly_balance.index:
                fundamentals['TotalDebt'] = quarterly_balance.loc['Total Debt'].values
            elif 'Long Term Debt' in quarterly_balance.index:
                fundamentals['TotalDebt'] = quarterly_balance.loc['Long Term Debt'].values
            if 'Total Stockholder Equity' in quarterly_balance.index:
                fundamentals['TotalEquity'] = quarterly_balance.loc['Total Stockholder Equity'].values
            if 'Total Assets' in quarterly_balance.index:
                fundamentals['TotalAssets'] = quarterly_balance.loc['Total Assets'].values
        
        # Try to get EPS from earnings, or calculate from Net Income and shares
        if earnings is not None and not earnings.empty:
            fundamentals['EPS'] = earnings['Earnings'].values
        elif 'NetIncome' in fundamentals.columns:
            # Try to get shares outstanding from info
            try:
                info = stock.info
                shares_outstanding = info.get('sharesOutstanding', None)
                if shares_outstanding:
                    fundamentals['EPS'] = fundamentals['NetIncome'] / shares_outstanding
                    logger.info("Calculated EPS from Net Income and shares outstanding")
                else:
                    logger.warning("Could not calculate EPS - shares outstanding not available")
            except:
                logger.warning("Could not fetch shares outstanding to calculate EPS")
        
        if 'EPS' not in fundamentals.columns:
            logger.warning(f"EPS data not available for {ticker} - model training will fail without it")
        
        # Filter by start date
        fundamentals = fundamentals[fundamentals.index >= start_date]
        
        # Calculate derived metrics
        fundamentals['GrossMargin'] = (fundamentals['GrossProfit'] / fundamentals['Revenue']) * 100
        fundamentals['OperatingMargin'] = (fundamentals['OperatingIncome'] / fundamentals['Revenue']) * 100
        fundamentals['NetMargin'] = (fundamentals['NetIncome'] / fundamentals['Revenue']) * 100
        
        if 'TotalDebt' in fundamentals.columns and 'TotalEquity' in fundamentals.columns:
            fundamentals['DebtToEquity'] = fundamentals['TotalDebt'] / fundamentals['TotalEquity']
        
        logger.info(f"Successfully fetched {len(fundamentals)} quarters of fundamental data")
        return fundamentals.reset_index()
        
    except Exception as e:
        logger.error(f"Error fetching fundamentals: {e}")
        return _fetch_alternative_fundamentals(ticker, start_date)


def _fetch_alternative_fundamentals(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Alternative method using historical data download.
    """
    logger.info("Using alternative method for fundamentals")
    
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, interval="3mo")
    
    if hist.empty:
        logger.error(f"No data available for {ticker}")
        return pd.DataFrame()
    
    # Get quarterly earnings estimates
    try:
        info = stock.info
        fundamentals = pd.DataFrame()
        fundamentals['Date'] = hist.index
        fundamentals['Price'] = hist['Close'].values
        
        # Note: This is a fallback - actual fundamentals may need manual entry
        # or use of FinancialModelingPrep API
        logger.warning("Using price data as proxy - consider using FMP API for full fundamentals")
        
        return fundamentals.reset_index()
    except Exception as e:
        logger.error(f"Alternative method failed: {e}")
        return pd.DataFrame()


def save_fundamentals(df: pd.DataFrame, ticker: str, output_dir: Path):
    """Save fundamentals data to CSV."""
    output_path = output_dir / f"{ticker}_fundamentals.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved fundamentals to {output_path}")

