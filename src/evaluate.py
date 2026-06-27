import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    FIGURES_DIR,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)


def evaluate_saved_model() -> None:
    """
    Evaluate the saved model on the test split.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print("Saved Model Evaluation")
    print("=" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print()
    print(classification_report(y_test, y_pred))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("Confusion Matrix - Saved Model")
    plt.xlabel("Predicted Crop")
    plt.ylabel("Actual Crop")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_saved_model.png", dpi=300)
    plt.close()

    with open(REPORTS_DIR / "saved_model_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_saved_model()