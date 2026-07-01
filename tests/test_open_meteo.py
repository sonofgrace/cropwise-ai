from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.external.open_meteo import fetch_historical_weather


def test_fetch_historical_weather_returns_dataframe():
    fake_response_data = {
        "daily": {
            "time": ["2024-01-01", "2024-01-02"],
            "temperature_2m_mean": [27.0, 28.0],
            "temperature_2m_max": [32.0, 33.0],
            "temperature_2m_min": [22.0, 23.0],
            "precipitation_sum": [0.0, 5.0],
            "rain_sum": [0.0, 5.0],
        }
    }

    mock_response = Mock()
    mock_response.json.return_value = fake_response_data
    mock_response.raise_for_status.return_value = None

    with patch("src.external.open_meteo.requests.get",
               return_value=mock_response):
        result = fetch_historical_weather(
            latitude=6.5244,
            longitude=3.3792,
            start_date="2024-01-01",
            end_date="2024-01-02",
        )

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 2
    assert "temperature_2m_mean" in result.columns
    assert "precipitation_sum" in result.columns


def test_fetch_historical_weather_raises_error_when_daily_missing():
    fake_response_data = {
        "error": "No data found",
    }

    mock_response = Mock()
    mock_response.json.return_value = fake_response_data
    mock_response.raise_for_status.return_value = None

    with patch("src.external.open_meteo.requests.get",
               return_value=mock_response):
        with pytest.raises(ValueError, match="No daily weather data"):
            fetch_historical_weather(
                latitude=6.5244,
                longitude=3.3792,
                start_date="2024-01-01",
                end_date="2024-01-02",
            )


def test_fetch_historical_weather_sends_expected_params():
    fake_response_data = {
        "daily": {
            "time": ["2024-01-01"],
            "temperature_2m_mean": [27.0],
            "temperature_2m_max": [32.0],
            "temperature_2m_min": [22.0],
            "precipitation_sum": [0.0],
            "rain_sum": [0.0],
        }
    }

    mock_response = Mock()
    mock_response.json.return_value = fake_response_data
    mock_response.raise_for_status.return_value = None

    with patch("src.external.open_meteo.requests.get",
               return_value=mock_response) as mock_get:
        fetch_historical_weather(
            latitude=6.5244,
            longitude=3.3792,
            start_date="2024-01-01",
            end_date="2024-01-02",
        )

    _, kwargs = mock_get.call_args

    assert kwargs["params"]["latitude"] == 6.5244
    assert kwargs["params"]["longitude"] == 3.3792
    assert kwargs["params"]["start_date"] == "2024-01-01"
    assert kwargs["params"]["end_date"] == "2024-01-02"
    assert "temperature_2m_mean" in kwargs["params"]["daily"]
    assert kwargs["timeout"] == 30