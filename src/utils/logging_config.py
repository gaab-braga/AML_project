import logging
import sys

def setup_logging():
    """Configure logging for AML project."""
    # Clear any existing handlers
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.WARNING)
    
    # Disable logs from specific libraries
    for name in ['src', 'sklearn', 'xgboost', 'lightgbm', 'src.modeling', 'pandas', 'numpy']:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        logger.propagate = False