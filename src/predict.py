import json

import joblib
import pandas as pd
import sklearn

from src.config import (
    MODEL_METADATA_PATH,
    MODEL_PATH,
    TUNED_MODEL_METADATA_PATH,
    TUNED_MODEL_PATH,
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


def check_model_version(prefer_tuned: bool = True) -> None:
    """
    Warn if the current scikit-learn version differs from the version
     used during training.
    """
    metadata_path = (
        TUNED_MODEL_METADATA_PATH
        if prefer_tuned and TUNED_MODEL_METADATA_PATH.exists()
        else MODEL_METADATA_PATH
    )

    if not metadata_path.exists():
        return

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    trained_version = metadata.get("scikit_learn_version")
    current_version = sklearn.__version__

    if trained_version and trained_version != current_version:
        print(
            f"Warning: model was trained with scikit-learn {trained_version}, "
            f"but current version is {current_version}. "
            "Retrain the model to avoid compatibility issues."
        )


def load_model(prefer_tuned: bool = True):
    """
    Load the trained crop recommendation model.

    Parameters
    ----------
    prefer_tuned : bool
        If True, load the tuned model. Otherwise, load the baseline best model.
    """
    check_model_version(prefer_tuned=prefer_tuned)

    model_path = (
        TUNED_MODEL_PATH
        if prefer_tuned and TUNED_MODEL_PATH.exists()
        else MODEL_PATH
    )

    return joblib.load(model_path)


def prepare_input(input_data: dict) -> pd.DataFrame:
    """
    Prepare input data for prediction.
    """
    missing_features = [
        feature for feature in FEATURE_ORDER
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_ORDER]

    return input_df


def predict_crop(input_data: dict) -> str:
    """
    Predict the best crop for a single input observation.
    """
    model = load_model()
    input_df = prepare_input(input_data)

    prediction = model.predict(input_df)[0]

    return prediction


def predict_crop_with_probabilities(input_data: dict) -> dict:
    """Predict crop and return class probabilities."""
    model = load_model()
    input_df = prepare_input(input_data)

    prediction = model.predict(input_df)[0]

    probability_array = model.predict_proba(input_df)[0]

    probabilities = {
        str(crop): float(probability)
        for crop, probability in zip(model.classes_,
                                     probability_array, strict=False)
    }

    sorted_probabilities = dict(
        sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    confidence = probabilities.get(
        str(prediction),
        max(probabilities.values()) if probabilities else 0.0,
    )

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "probabilities": sorted_probabilities,
    }

def get_top_n_recommendations(input_data: dict, n: int = 3) -> list[dict]:
    """Return top-N crop recommendations with probabilities."""
    prediction_result = predict_crop_with_probabilities(input_data)
    probabilities = prediction_result["probabilities"]

    return [
        {"crop": crop, "probability": probability}
        for crop, probability in list(probabilities.items())[:n]
    ]


if __name__ == "__main__":
    sample_input = {
        "n": 90,
        "p": 42,
        "k": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9,
    }

    result = predict_crop_with_probabilities(sample_input)

    print(result)

    print("\nTop 3 recommendations:")
    print(get_top_n_recommendations(sample_input, n=3))