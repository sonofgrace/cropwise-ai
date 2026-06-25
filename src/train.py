import json
from pathlib import Path

import sklearn

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import (
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    REPORTS_DIR,
)


def get_models() -> dict:
    """
    Define candidate machine learning models.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "SVM": CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False)
    }


def build_pipeline(model):
    """
    Build a modeling pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def evaluate_models(X_train, y_train) -> pd.DataFrame:
    """
    Evaluate candidate models using cross-validation.
    """
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    results = []

    for name, model in get_models().items():
        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({
            "model": name,
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
            "cv_macro_f1_std": scores["test_macro_f1"].std(),
            "cv_weighted_f1_mean": scores["test_weighted_f1"].mean(),
            "cv_weighted_f1_std": scores["test_weighted_f1"].std(),
        })

    return (
        pd.DataFrame(results)
        .sort_values("cv_macro_f1_mean", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    cv_results = evaluate_models(X_train, y_train)

    best_model_name = cv_results.loc[0, "model"]
    best_model = get_models()[best_model_name]
    best_pipeline = build_pipeline(best_model)

    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average="macro")
    test_weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    metadata = {
        "best_model": best_model_name,
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "features": X.columns.tolist(),
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "scikit_learn_version": sklearn.__version__,
    }

    joblib.dump(best_pipeline, MODEL_PATH)

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    cv_results.to_csv(
        REPORTS_DIR / "model_comparison_cv_results.csv",
        index=False
    )

    report = classification_report(y_test, y_pred)

    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    print("Model training completed successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro-F1: {test_macro_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()