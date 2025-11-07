# Nucor (NUE) 6-Month EPS Growth Prediction

Predict Nucor's 6-month-ahead EPS growth rate using historical financial, market, and macroeconomic data. This model supports DCF analysis by providing data-driven estimates of near-term earnings acceleration.

## 🎯 Project Overview

This project uses XGBoost regression to predict Nucor's EPS growth rate 6 months ahead (2 quarters) based on:
- **Company Fundamentals**: Quarterly income statement metrics (EPS, revenue, margins, leverage)
- **Market Data**: Stock prices, S&P 500, Steel ETF (SLX)
- **Macroeconomic Indicators**: 10-year yield, CPI, PMI, Industrial Production
- **Sector/Commodity Data**: Steel industry proxies

## 📁 Project Structure

```
NUE_xgboost/
├── data/
│   ├── raw/              # Raw data snapshots
│   └── processed/        # Cleaned and merged datasets
├── src/
│   ├── data_acquisition/ # Data fetching modules
│   ├── preprocessing/    # Feature engineering
│   ├── modeling/         # Model training
│   ├── evaluation/       # Model validation
│   └── forecasting/      # Prediction generation
├── models/              # Saved model artifacts
├── notebooks/           # Jupyter notebooks for exploration
├── config/              # Configuration files
└── requirements.txt     # Python dependencies
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Get a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html

3. (Optional) Get a FinancialModelingPrep API key from https://financialmodelingprep.com/

4. Add your keys to `.env`:
   ```
   FRED_API_KEY=your_fred_key_here
   FMP_API_KEY=your_fmp_key_here
   ```

### 3. Run the Pipeline

```bash
# 1. Acquire data
python src/data_acquisition/main.py

# 2. Prepare features
python src/preprocessing/prepare_features.py

# 3. Train model
python src/modeling/train_model.py

# 4. Evaluate model
python src/evaluation/evaluate_model.py

# 5. Generate forecast
python src/forecasting/generate_forecast.py
```

## 📊 Data Sources

- **Yahoo Finance** (yfinance): Market data, no API key required
- **FRED** (fredapi): Macroeconomic indicators, free API key required
- **FinancialModelingPrep** (optional): Clean fundamentals data

## 🔧 Key Features

- **Temporal Alignment**: All data resampled to quarter-end timestamps
- **Feature Lagging**: Explanatory variables shifted by one quarter to avoid look-ahead bias
- **Time-Series Validation**: Chronological train/test split with walk-forward validation
- **SHAP Integration**: Feature importance and interpretability
- **DCF Integration**: Outputs formatted for DCF model comparison

## 📈 Model Output

The model predicts:
- **6-month EPS growth rate**: `(EPS_{t+2} - EPS_t) / |EPS_t|`
- **Feature importance**: Which drivers matter most
- **Forecast with confidence**: Latest quarter prediction for DCF comparison

## 📝 Notes

- Data frequency: Fundamentals (quarterly), Market (daily → quarterly), Macro (monthly → quarterly)
- Training period: 2010-2022 (configurable)
- Test period: 2023-2024 (configurable)
- Model: XGBoost Regressor with hyperparameter tuning

