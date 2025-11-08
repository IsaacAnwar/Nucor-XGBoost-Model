"""
Fetch peer company fundamental data for relative metrics.
"""
import pandas as pd
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_acquisition.fundamentals import fetch_fundamentals
from config.config import FMP_API_KEY

logger = logging.getLogger(__name__)


def fetch_peer_fundamentals(
    peer_tickers: list,
    start_date: str = "2010-01-01"
) -> pd.DataFrame:
    """
    Fetch fundamentals for peer companies and aggregate for relative metrics.
    
    Parameters:
    -----------
    peer_tickers : list
        List of peer company tickers (e.g., ['STLD', 'X', 'CLF'])
    start_date : str
        Start date for data collection (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame
        Aggregated peer fundamentals with date index
        Columns include averages for Revenue, EPS, Margins, etc.
    """
    logger.info(f"Fetching peer fundamentals for {peer_tickers} from {start_date}")
    
    all_peer_data = {}
    
    for ticker in peer_tickers:
        try:
            logger.info(f"Fetching fundamentals for peer: {ticker}")
            peer_fundamentals = fetch_fundamentals(ticker, start_date)
            
            if peer_fundamentals.empty:
                logger.warning(f"No data fetched for peer {ticker}")
                continue
            
            all_peer_data[ticker] = peer_fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching data for peer {ticker}: {e}")
            continue
    
    if not all_peer_data:
        logger.warning("No peer data fetched")
        return pd.DataFrame()
    
    # Aggregate peer data by date
    # Get all unique dates
    all_dates = set()
    for peer_df in all_peer_data.values():
        if 'Date' in peer_df.columns:
            all_dates.update(peer_df['Date'].values)
    
    if not all_dates:
        logger.warning("No dates found in peer data")
        return pd.DataFrame()
    
    all_dates = sorted(list(all_dates))
    
    # Create aggregated DataFrame
    aggregated = pd.DataFrame()
    aggregated['Date'] = pd.to_datetime(all_dates)
    aggregated = aggregated.sort_values('Date').reset_index(drop=True)
    
    # Aggregate metrics across peers
    metrics_to_aggregate = [
        'Revenue', 'EPS', 'GrossMargin', 'OperatingMargin', 'NetMargin',
        'Revenue_Growth_QoQ', 'Revenue_Growth_YoY', 'EPS_Growth_QoQ', 'EPS_Growth_YoY'
    ]
    
    for metric in metrics_to_aggregate:
        values_by_date = {}
        
        for date in all_dates:
            values = []
            for ticker, peer_df in all_peer_data.items():
                if 'Date' in peer_df.columns and metric in peer_df.columns:
                    date_rows = peer_df[peer_df['Date'] == date]
                    if not date_rows.empty:
                        value = date_rows[metric].iloc[0]
                        if pd.notna(value):
                            values.append(value)
            
            if values:
                values_by_date[date] = {
                    'mean': pd.Series(values).mean(),
                    'median': pd.Series(values).median(),
                    'std': pd.Series(values).std() if len(values) > 1 else 0
                }
        
        # Add aggregated columns
        if values_by_date:
            aggregated[f"Peer_{metric}_Mean"] = aggregated['Date'].map(
                lambda d: values_by_date.get(d, {}).get('mean', None)
            )
            aggregated[f"Peer_{metric}_Median"] = aggregated['Date'].map(
                lambda d: values_by_date.get(d, {}).get('median', None)
            )
            if any(v.get('std', 0) > 0 for v in values_by_date.values()):
                aggregated[f"Peer_{metric}_Std"] = aggregated['Date'].map(
                    lambda d: values_by_date.get(d, {}).get('std', None)
                )
    
    logger.info(f"Successfully aggregated peer data: {len(aggregated)} quarters")
    return aggregated


def save_peer_data(df: pd.DataFrame, output_dir: Path):
    """Save peer data to CSV."""
    output_path = output_dir / "peer_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved peer data to {output_path}")

