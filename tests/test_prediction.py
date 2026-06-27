import pytest

from src.predict import (
    get_top_n_recommendations,
    predict_crop,
    predict_crop_with_probabilities,
    prepare_input,
)

SAMPLE_INPUT = {
    "n": 90,
    "p": 42,
    "k": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9,
}


def test_prepare_input_returns_dataframe_with_correct_columns():
    input_df = prepare_input(SAMPLE_INPUT)

    assert list(input_df.columns) == [
        "n",
        "p",
        "k",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]

    assert input_df.shape == (1, 7)


def test_prepare_input_raises_error_for_missing_feature():
    incomplete_input = SAMPLE_INPUT.copy()
    incomplete_input.pop("rainfall")

    with pytest.raises(ValueError):
        prepare_input(incomplete_input)


def test_predict_crop_returns_string():
    prediction = predict_crop(SAMPLE_INPUT)

    assert isinstance(prediction, str)
    assert len(prediction) > 0


def test_predict_crop_with_probabilities_returns_expected_structure():
    result = predict_crop_with_probabilities(SAMPLE_INPUT)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)


def test_probabilities_sum_close_to_one():
    result = predict_crop_with_probabilities(SAMPLE_INPUT)

    probability_sum = sum(result["probabilities"].values())

    assert abs(probability_sum - 1.0) < 1e-6


def test_get_top_n_recommendations_returns_three_items():
    recommendations = get_top_n_recommendations(SAMPLE_INPUT, n=3)

    assert isinstance(recommendations, list)
    assert len(recommendations) == 3

    for item in recommendations:
        assert "crop" in item
        assert "probability" in item