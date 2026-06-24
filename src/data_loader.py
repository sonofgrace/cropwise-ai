import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw crop recommendation dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset containing soil, climate, and crop label columns.
    """
    return pd.read_csv(RAW_DATA_PATH)


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Crop recommendation dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    """
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    return X, y
