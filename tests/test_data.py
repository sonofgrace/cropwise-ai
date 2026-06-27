import pandas as pd

from src.config import PROCESSED_DATA_PATH, TARGET_COLUMN

REQUIRED_COLUMNS = [
    "n",
    "p",
    "k",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "label",
]


def test_processed_data_file_exists():
    assert PROCESSED_DATA_PATH.exists(), "Processed dataset does not exist."


def test_processed_data_loads_successfully():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_required_columns_exist():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    for column in REQUIRED_COLUMNS:
        assert column in df.columns, f"Missing required column: {column}"


def test_no_missing_values():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    assert df.isna().sum().sum() == 0


def test_target_column_exists():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    assert TARGET_COLUMN in df.columns


def test_expected_number_of_crop_classes():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    assert df[TARGET_COLUMN].nunique() == 22


def test_each_crop_has_records():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    class_counts = df[TARGET_COLUMN].value_counts()

    assert (class_counts > 0).all()