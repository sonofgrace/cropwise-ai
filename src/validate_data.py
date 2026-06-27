import pandas as pd

from src.config import RAW_DATA_PATH, TARGET_COLUMN


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the crop recommendation dataset.
    Parameters
    ----------
    df : pd.dataFrame
         Input dataframe


    Returns
    -------
    dict
        Dictionary containing validation results

    """
    validation_report = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "target_column": TARGET_COLUMN,
        "number_of_classes": int(df[TARGET_COLUMN].nunique()),
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "numeric_summary": df.describe().to_dict()
    }
    return validation_report


if __name__ == "__main__":
    data = pd.read_csv(RAW_DATA_PATH)
    report = validate_dataset(data)

    print("Dataset Validation Report")
    print("=" * 30)

    print(f"Shape: {report['shape']}")
    print(f"Columns: {report['columns']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Number of classes: {report['number_of_classes']}")

    print("\nMissing values:")
    for column, count in report["missing_values"].items():
        print(f"{column}: {count}")

    print("\nClass distribution:")
    for crop, count in report["class_distribution"].items():
        print(f"{crop}: {count}")