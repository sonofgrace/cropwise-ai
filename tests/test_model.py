import joblib

from src.config import MODEL_PATH, TUNED_MODEL_PATH


def test_model_file_exists():
    assert TUNED_MODEL_PATH.exists() or MODEL_PATH.exists(),\
        "No trained model file found."


def test_model_loads_successfully():
    model_path = TUNED_MODEL_PATH if TUNED_MODEL_PATH.exists() else MODEL_PATH

    model = joblib.load(model_path)

    assert model is not None


def test_model_has_predict_method():
    model_path = TUNED_MODEL_PATH if TUNED_MODEL_PATH.exists() else MODEL_PATH

    model = joblib.load(model_path)

    assert hasattr(model, "predict")


def test_model_has_predict_proba_method():
    model_path = TUNED_MODEL_PATH if TUNED_MODEL_PATH.exists() else MODEL_PATH

    model = joblib.load(model_path)

    assert hasattr(model, "predict_proba")