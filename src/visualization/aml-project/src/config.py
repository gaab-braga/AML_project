# Configuration settings for the AML project

import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'

# File names
FEATURES_FILE_PKL = 'features_with_patterns_sampled.pkl'

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
OPTUNA_TRIALS = 30

# Logging configuration
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'