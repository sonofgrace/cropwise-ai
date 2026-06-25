import json

import sklearn

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import (
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    MODEL_DIR,
    REPORTS_DIR,
)


def build_pipeline(model):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def get_search_spaces():
    rf_pipeline = build_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE)
    )

    et_pipeline = build_pipeline(
        ExtraTreesClassifier(random_state=RANDOM_STATE)
    )

    param_dist = {
        "model__n_estimators": [100, 200, 300, 500, 700],
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
        "model__bootstrap": [True, False],
    }

    return {
        "Random Forest": {
            "pipeline": rf_pipeline,
            "params": param_dist,
        },
        "Extra Trees": {
            "pipeline": et_pipeline,
            "params": param_dist,
        },
    }


def tune_models(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    tuning_rows = []
    best_name = None
    best_search = None
    best_score = -1

    for model_name, config in get_search_spaces().items():
        search = RandomizedSearchCV(
            estimator=config["pipeline"],
            param_distributions=config["params"],
            n_iter=40,
            scoring="f1_macro",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X_train, y_train)

        tuning_rows.append({
            "model": model_name,
            "best_cv_macro_f1": search.best_score_,
            "best_params": search.best_params_,
        })

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = model_name
            best_search = search

    tuning_results = (
        pd.DataFrame(tuning_rows)
        .sort_values("best_cv_macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    return best_name, best_search, tuning_results


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
        random_state=RANDOM_STATE,
    )

    best_model_name, best_search, tuning_results = tune_models(
        X_train,
        y_train
    )

    best_model = best_search.best_estimator_
    y_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average="macro")
    test_weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    tuned_model_path = MODEL_DIR / "crop_model_tuned.joblib"
    tuned_metadata_path = MODEL_DIR / "tuned_model_metadata.json"

    joblib.dump(best_model, tuned_model_path)

    metadata = {
        "model_name": best_model_name,
        "best_cv_macro_f1": float(best_search.best_score_),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "best_params": best_search.best_params_,
        "features": X.columns.tolist(),
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "scikit_learn_version": sklearn.__version__
    }

    with open(tuned_metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    tuning_results.to_csv(
        REPORTS_DIR / "hyperparameter_tuning_results.csv",
        index=False
    )

    with open(REPORTS_DIR / "tuned_model_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    print("Hyperparameter tuning completed successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Best CV macro-F1: {best_search.best_score_:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro-F1: {test_macro_f1:.4f}")
    print(f"Tuned model saved to: {tuned_model_path}")


if __name__ == "__main__":
    main()