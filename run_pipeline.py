"""
Main pipeline script to run the complete NUE EPS prediction workflow.
"""
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete pipeline."""
    logger.info("=" * 60)
    logger.info("NUE EPS Growth Prediction - Complete Pipeline")
    logger.info("=" * 60)
    
    steps = [
        ("1. Data Acquisition", "src.data_acquisition.main", "main"),
        ("2. Feature Preparation", "src.preprocessing.prepare_features", "prepare_features"),
        ("3. Model Training", "src.modeling.train_model", "main"),
        ("4. Model Evaluation", "src.evaluation.evaluate_model", "main"),
        ("5. Generate Forecast", "src.forecasting.generate_forecast", "main"),
    ]
    
    for step_name, module_path, function_name in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {step_name}")
        logger.info(f"{'='*60}\n")
        
        try:
            module = __import__(module_path, fromlist=[function_name])
            func = getattr(module, function_name)
            func()
        except Exception as e:
            logger.error(f"Error in {step_name}: {e}")
            logger.error("Pipeline stopped. Please fix the error and rerun.")
            sys.exit(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete! All steps executed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

