import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model():
    """
    Load the trained crop recommendation model.
    """
    return joblib.load(MODEL_PATH)


def predict_crop(input_data: dict) -> str:
    """
    Predict the best crop for a single input observation.

    Parameters
    ----------
    input_data : dict
        Dictionary containing values for:
        n, p, k, temperature, humidity, ph, rainfall

    Returns
    -------
    str
        Predicted crop label.
    """
    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    return prediction


def predict_crop_with_probabilities(input_data: dict) -> dict:
    """
    Predict crop and return class probabilities.
    """
    model = load_model()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    sorted_probabilities = dict(
        sorted(
            class_probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "prediction": prediction,
        "probabilities": sorted_probabilities
    }


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