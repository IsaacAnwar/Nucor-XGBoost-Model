"""
Configuration settings for NUE EPS Growth Prediction project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
# Try loading from project root first, then current directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()  # Fallback to default behavior

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

# Ticker symbols
TICKER = "NUE"  # Nucor
MARKET_INDEX = "^GSPC"  # S&P 500
SECTOR_ETF = "SLX"  # Steel ETF

# Data date ranges
TRAIN_START = "2010-01-01"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# FRED Series IDs for macroeconomic data
FRED_SERIES = {
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
    "MANEMP": "All Employees, Manufacturing",
    "INDPRO": "Industrial Production Index",
    "UMCSENT": "University of Michigan: Consumer Sentiment",
    "PAYEMS": "All Employees, Total Nonfarm",
}

# Model parameters
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

# Evaluation metrics
EVAL_METRICS = ["mae", "rmse", "r2"]

# Logging
LOG_LEVEL = "INFO"

