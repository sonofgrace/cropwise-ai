from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
EXTERNAL_DATA_DIR = ROOT_DIR / "data" / "external"

RAW_DATA_PATH = RAW_DATA_DIR / "Crop_recommendation.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "crop_recommendation_clean.csv"

# Model paths
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "crop_model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
TUNED_MODEL_PATH = MODEL_DIR / "crop_model_tuned.joblib"
TUNED_MODEL_METADATA_PATH = MODEL_DIR / "tuned_model_metadata.json"

# Reports
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Target column
TARGET_COLUMN = "label"
