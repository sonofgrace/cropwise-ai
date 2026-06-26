import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError:
    shap = None

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from src.config import (
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    TUNED_MODEL_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    REPORTS_DIR,
    FIGURES_DIR,
)


FEATURE_ORDER = [
    "n",
    "p",
    "k",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]


def load_explanation_model():
    """
    Load tuned model if available, otherwise load baseline saved model.
    """
    model_path = TUNED_MODEL_PATH if TUNED_MODEL_PATH.exists() else MODEL_PATH
    return joblib.load(model_path)


def get_crop_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return average feature values for each crop.
    """
    return df.groupby(TARGET_COLUMN)[FEATURE_ORDER].mean()


def get_global_feature_importance(model) -> pd.DataFrame:
    """
    Extract global feature importance from a tree-based model pipeline.
    """
    final_model = model.named_steps["model"]

    if not hasattr(final_model, "feature_importances_"):
        raise AttributeError(
            "The final estimator does not expose feature_importances_."
        )

    importance_df = (
        pd.DataFrame({
            "feature": FEATURE_ORDER,
            "importance": final_model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


def get_permutation_importance(model, X_test, y_test) -> pd.DataFrame:
    """
    Compute permutation importance using macro-F1.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = (
        pd.DataFrame({
            "feature": FEATURE_ORDER,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


def explain_prediction(input_data: dict, top_n: int = 3) -> dict:
    """
    Explain a single crop prediction using probabilities and crop-profile matching.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    model = load_explanation_model()

    crop_profiles = get_crop_profiles(df)

    missing_features = [
        feature for feature in FEATURE_ORDER
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    input_df = pd.DataFrame([input_data])[FEATURE_ORDER]

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    probability_df = (
        pd.DataFrame({
            "crop": model.classes_,
            "probability": probabilities,
        })
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    predicted_profile = crop_profiles.loc[prediction]

    comparison_df = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "input_value": [input_data[f] for f in FEATURE_ORDER],
        "predicted_crop_average": predicted_profile.values,
    })

    comparison_df["difference"] = (
        comparison_df["input_value"] -
        comparison_df["predicted_crop_average"]
    )

    comparison_df["absolute_difference"] = comparison_df["difference"].abs()

    closest_features = (
        comparison_df
        .sort_values("absolute_difference")
        .head(top_n)
    )

    return {
        "prediction": prediction,
        "confidence": float(probability_df.loc[0, "probability"]),
        "top_recommendations": probability_df.head(top_n).to_dict(orient="records"),
        "closest_matching_features": closest_features.to_dict(orient="records"),
    }


def format_explanation(explanation: dict) -> str:
    """
    Convert explanation dictionary into user-friendly text.
    """
    prediction = explanation["prediction"]
    confidence = explanation["confidence"]

    text = (
        f"The recommended crop is {prediction} "
        f"with a confidence of {confidence:.1%}.\n\n"
    )

    text += "The input values most closely match this crop profile for:\n"

    for item in explanation["closest_matching_features"]:
        feature = item["feature"]
        input_value = item["input_value"]
        crop_average = item["predicted_crop_average"]

        text += (f"- {feature}: input value {input_value:.2f}, "
            f"typical {prediction} average {crop_average:.2f}\n"
        )

    return text


def save_importance_plot(
    importance_df: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str
) -> None:
    """
    Save feature importance plot.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x=value_col,
        y="feature"
    )
    plt.title(title)
    plt.xlabel(value_col)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def save_shap_summary_for_class(
    model,
    X_test: pd.DataFrame,
    class_name: str,
    filename: str | None = None
) -> None:
    """
    Save a SHAP summary plot for a selected crop class.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained model pipeline containing scaler and tree-based estimator.
    X_test : pd.DataFrame
        Test feature data before scaling.
    class_name : str
        Crop class to explain.
    filename : str | None
        Output filename. If None, a default filename is used.
    """

    if shap is None:
        print("SHAP is not installed. Skipping SHAP summary plots.")
        return

    if class_name not in model.classes_:
        raise ValueError(
            f"{class_name} is not a valid class. "
            f"Available classes: {list(model.classes_)}"
        )

    scaler = model.named_steps["scaler"]
    final_model = model.named_steps["model"]

    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=FEATURE_ORDER
    )

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_scaled_df)

    class_index = list(model.classes_).index(class_name)

    if isinstance(shap_values, list):
        shap_values_for_class = shap_values[class_index]
    else:
        shap_values_for_class = shap_values[:, :, class_index]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"shap_summary_{class_name}.png"

    shap.summary_plot(
        shap_values_for_class,
        X_test_scaled_df,
        feature_names=FEATURE_ORDER,
        show=False
    )

    plt.title(f"SHAP Summary Plot for {class_name}")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df[FEATURE_ORDER]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = load_explanation_model()

    global_importance = get_global_feature_importance(model)
    permutation_importance_df = get_permutation_importance(
        model,
        X_test,
        y_test
    )

    selected_shap_crops = ["rice", "maize", "cotton", "banana", "coffee"]

    for crop in selected_shap_crops:
        if crop in model.classes_:
            save_shap_summary_for_class(
                model=model,
                X_test=X_test,
                class_name=crop
            )

    global_importance.to_csv(
        REPORTS_DIR / "global_feature_importance.csv",
        index=False
    )

    permutation_importance_df.to_csv(
        REPORTS_DIR / "permutation_importance.csv",
        index=False
    )

    save_importance_plot(
        global_importance,
        "importance",
        "Global Feature Importance",
        "global_feature_importance.png"
    )

    save_importance_plot(
        permutation_importance_df,
        "importance_mean",
        "Permutation Importance Based on Macro-F1",
        "permutation_importance.png"
    )

    sample_input = {
        "n": 90,
        "p": 42,
        "k": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }

    explanation = explain_prediction(sample_input)

    print(format_explanation(explanation))
    print("\nGlobal feature importance:")
    print(global_importance)

    print("\nPermutation importance:")
    print(permutation_importance_df)


if __name__ == "__main__":
    main()