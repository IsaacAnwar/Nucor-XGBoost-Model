# Quick Start Guide

This guide will help you get started with the NUE EPS Growth Prediction project.

## Prerequisites

- Python 3.10 or higher
- pip package manager

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up API Keys

1. **FRED API Key (Required for macro data)**
   - Go to https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign up for a free account
   - Generate an API key
   - Copy `env.example` to `.env`:
     ```bash
     copy env.example .env
     ```
   - Edit `.env` and add your FRED API key:
     ```
     FRED_API_KEY=your_key_here
     ```

2. **FinancialModelingPrep API Key (Optional)**
   - Sign up at https://financialmodelingprep.com/
   - Add your key to `.env` if you want cleaner fundamentals data

## Step 3: Run the Pipeline

### Option A: Run Complete Pipeline

```bash
python run_pipeline.py
```

This will execute all steps:
1. Data acquisition
2. Feature preparation
3. Model training
4. Model evaluation
5. Forecast generation

### Option B: Run Steps Individually

```bash
# 1. Fetch data
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

## Step 4: Review Results

- **Model metrics**: Check console output for MAE, RMSE, R² scores
- **Visualizations**: See `notebooks/evaluation_results/` for plots
- **Forecast**: See `notebooks/forecasts/` for latest predictions

## Understanding the Output

### Model Performance Metrics

- **MAE (Mean Absolute Error)**: Average prediction error in EPS growth units
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more
- **R² (R-squared)**: Proportion of variance explained (1.0 = perfect)

### Forecast Output

The forecast script generates:
- **6-Month EPS Growth Rate**: Predicted percentage change
- **EPS Projection**: Estimated EPS level 6 months ahead
- **DCF Integration Notes**: How to use this in your DCF model

## Troubleshooting

### "FRED_API_KEY not found"
- Make sure you created a `.env` file (not just `env.example`)
- Verify the key is correctly formatted in `.env`

### "No data fetched"
- Check your internet connection
- Verify ticker symbols are correct in `config/config.py`
- Some data sources may have rate limits

### "Model not found"
- Make sure you've run the training step before evaluation/forecasting
- Check that `models/latest_model.pkl` exists

### Import Errors
- Make sure you're running scripts from the project root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

## Next Steps

1. **Experiment with features**: Add more macro indicators in `config/config.py`
2. **Tune hyperparameters**: Adjust `XGBOOST_PARAMS` in `config/config.py`
3. **Extend time range**: Modify `TRAIN_START`, `TRAIN_END` in `config/config.py`
4. **Add more companies**: Extend to peer analysis

## Data Sources

- **Yahoo Finance**: Market data (no API key needed)
- **FRED**: Macroeconomic indicators (free API key required)
- **FinancialModelingPrep**: Optional cleaner fundamentals

## Project Structure

```
NUE_xgboost/
├── data/
│   ├── raw/              # Raw data snapshots (timestamped)
│   └── processed/        # Cleaned features
├── src/
│   ├── data_acquisition/ # Fetch data from APIs
│   ├── preprocessing/    # Feature engineering
│   ├── modeling/         # Train XGBoost model
│   ├── evaluation/      # Model validation & plots
│   └── forecasting/     # Generate predictions
├── models/              # Saved model artifacts
├── notebooks/           # Results and analysis
└── config/              # Configuration settings
```

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review error messages in console output
3. Verify all dependencies are correctly installed

