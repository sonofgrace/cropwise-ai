import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def clean_crop_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the crop recommendation dataset.

    Current cleaning steps:
    - Standardize column names.
    - Remove duplicated rows.
    -Reset index.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned crop dataset

    """
    cleaned_df = df.copy()
    cleaned_df.columns = (cleaned_df.columns.str.strip()
                          .str.lower()
                          .str.replace(" ", "_")
                          )
    cleaned_df = cleaned_df.drop_duplicates()
    cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)
    cleaned_df = clean_crop_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Cleaned dataset saved successfully.")
    print(f"Path: {PROCESSED_DATA_PATH}")
    print(f"Shape: {cleaned_df.shape}")