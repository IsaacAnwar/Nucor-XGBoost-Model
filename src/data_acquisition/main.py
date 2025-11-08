"""
Main script to fetch all data sources for NUE EPS prediction.
"""
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import (
    DATA_RAW, TICKER, MARKET_INDEX, SECTOR_ETF,
    TRAIN_START, FRED_SERIES, PEER_TICKERS
)

from src.data_acquisition.fundamentals import fetch_fundamentals, save_fundamentals
from src.data_acquisition.market_data import fetch_market_data, save_market_data
from src.data_acquisition.macro_data import fetch_macro_data, save_macro_data
from src.data_acquisition.commodity_data import fetch_commodity_data, save_commodity_data
from src.data_acquisition.peer_data import fetch_peer_fundamentals, save_peer_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Fetch all data sources and save to data/raw/"""
    logger.info("=" * 60)
    logger.info("Starting data acquisition for NUE EPS prediction")
    logger.info("=" * 60)
    
    # Create timestamp for this data snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = DATA_RAW / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch fundamentals
    logger.info("\n1. Fetching fundamentals...")
    fundamentals = fetch_fundamentals(TICKER, TRAIN_START)
    if not fundamentals.empty:
        save_fundamentals(fundamentals, TICKER, snapshot_dir)
    else:
        logger.warning("No fundamentals data fetched")
    
    # 2. Fetch market data
    logger.info("\n2. Fetching market data...")
    market_tickers = [TICKER, MARKET_INDEX, SECTOR_ETF]
    market_data = fetch_market_data(market_tickers, TRAIN_START, resample_to_quarter=True)
    if not market_data.empty:
        save_market_data(market_data, snapshot_dir)
    else:
        logger.warning("No market data fetched")
    
    # 3. Fetch macro data
    logger.info("\n3. Fetching macroeconomic data...")
    macro_data = fetch_macro_data(TRAIN_START, resample_to_quarter=True)
    if not macro_data.empty:
        save_macro_data(macro_data, snapshot_dir)
    else:
        logger.warning("No macro data fetched")
    
    # 4. Fetch commodity data
    logger.info("\n4. Fetching commodity data...")
    commodity_data = fetch_commodity_data(TRAIN_START, resample_to_quarter=True)
    if not commodity_data.empty:
        save_commodity_data(commodity_data, snapshot_dir)
    else:
        logger.warning("No commodity data fetched")
    
    # 5. Fetch peer company data
    logger.info("\n5. Fetching peer company fundamentals...")
    peer_data = fetch_peer_fundamentals(PEER_TICKERS, TRAIN_START)
    if not peer_data.empty:
        save_peer_data(peer_data, snapshot_dir)
    else:
        logger.warning("No peer data fetched")
    
    logger.info("\n" + "=" * 60)
    logger.info("Data acquisition complete!")
    logger.info(f"Data saved to: {snapshot_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

