import pandas as pd
import pytest

from src.features.climate_features import (
    create_climate_summary,
    create_cropwise_input_from_climate,
)


def make_sample_weather_df():
    return pd.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "temperature_2m_mean": [27.0, 28.0, 29.0],
            "temperature_2m_max": [32.0, 33.0, 34.0],
            "temperature_2m_min": [22.0, 23.0, 24.0],
            "precipitation_sum": [0.0, 5.0, 25.0],
            "rain_sum": [0.0, 5.0, 25.0],
        }
    )


def test_create_climate_summary_returns_expected_keys():
    weather_df = make_sample_weather_df()

    summary = create_climate_summary(weather_df)

    expected_keys = {
        "mean_temperature",
        "max_temperature",
        "min_temperature",
        "total_rainfall",
        "mean_daily_rainfall",
        "rainfall_std",
        "dry_days",
        "wet_days",
        "heavy_rain_days",
    }

    assert expected_keys.issubset(summary.keys())


def test_create_climate_summary_values_are_correct():
    weather_df = make_sample_weather_df()

    summary = create_climate_summary(weather_df)

    assert summary["mean_temperature"] == pytest.approx(28.0)
    assert summary["max_temperature"] == pytest.approx(34.0)
    assert summary["min_temperature"] == pytest.approx(22.0)
    assert summary["total_rainfall"] == pytest.approx(30.0)
    assert summary["dry_days"] == 1
    assert summary["wet_days"] == 2
    assert summary["heavy_rain_days"] == 1


    def test_create_climate_summary_raises_error_for_missing_columns():
        weather_df = pd.DataFrame(
            {
               "temperature_2m_mean": [27.0, 28.0],
                "precipitation_sum": [0.0, 5.0],
            }
        )

        with pytest.raises(ValueError, match="Missing required weather columns"):
            create_climate_summary(weather_df)


    def test_create_cropwise_input_from_climate_returns_model_ready_input():
        soil_input = {
            "n": 90,
            "p": 42,
            "k": 43,
            "ph": 6.5,
            "humidity": 82.0,
        }

        climate_summary = {
            "mean_temperature": 28.0,
            "total_rainfall": 1200.0,
        }

        model_input = create_cropwise_input_from_climate(
            soil_input = soil_input,
            climate_summary = climate_summary,
        )

        expected_input = {
            "n": 90,
            "p": 42,
            "k": 43,
            "temperature": 28.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 1200.0,
        }

        assert model_input == expected_input


def test_create_cropwise_input_from_climate_uses_default_humidity():
    soil_input = {
        "n": 90,
        "p": 42,
        "k": 43,
        "ph": 6.5,
    }

    climate_summary = {
        "mean_temperature": 28.0,
        "total_rainfall": 1200.0,
    }

    model_input = create_cropwise_input_from_climate(
        soil_input = soil_input,
        climate_summary = climate_summary,
    )

    assert model_input["humidity"] == 70.0


def test_create_cropwise_input_from_climate_raises_error_for_missing_soil_keys():
    soil_input = {
        "n": 90,
        "p": 42,
        "ph": 6.5,
    }

    climate_summary = {
        "mean_temperature": 28.0,
        "total_rainfall": 1200.0,
    }

    with pytest.raises(ValueError, match="Missing required soil inputs"):
        create_cropwise_input_from_climate(
            soil_input = soil_input,
            climate_summary = climate_summary,
        )