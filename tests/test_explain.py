from src.explain import explain_prediction, format_explanation

SAMPLE_INPUT = {
    "n": 90,
    "p": 42,
    "k": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9,
}


def test_explain_prediction_returns_expected_keys():
    explanation = explain_prediction(SAMPLE_INPUT, top_n=3)

    assert isinstance(explanation, dict)
    assert "prediction" in explanation
    assert "confidence" in explanation
    assert "top_recommendations" in explanation
    assert "closest_matching_features" in explanation


def test_format_explanation_returns_string():
    explanation = explain_prediction(SAMPLE_INPUT, top_n=3)
    text = format_explanation(explanation)

    assert isinstance(text, str)
    assert len(text) > 0
    assert "recommended crop" in text.lower()