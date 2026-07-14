from __future__ import annotations

import pandas as pd


def create_climate_summary(weather_df: pd.DataFrame) -> dict:
    """
    Create crop-relevant climate summary features
     from daily weather data.
    Parameters
    ----------
    weather_df:
        Daily weather dataframe from Open-Meteo.


    Returns
    -------
    dict
        Climate summary features.
    """
    required_columns = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in weather_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required weather columns:"
                         f" {missing_columns}")
    rainfall = weather_df["precipitation_sum"]

    summary = {
        "mean_temperature": float(weather_df["temperature_2m_mean"].mean()),
        "max_temperature": float(weather_df["temperature_2m_max"].max()),
        "min_temperature": float(weather_df["temperature_2m_min"].min()),
        "total_rainfall": float(rainfall.sum()),
        "mean_daily_rainfall": float(rainfall.mean()),
        "rainfall_std": float(rainfall.std()),
        "dry_days": int((rainfall < 1.0).sum()),
        "wet_days": int((rainfall >= 1.0).sum()),
        "heavy_rain_days": int((rainfall >= 20.0).sum()),
    }

    return summary


def create_cropwise_input_from_climate(
        soil_input: dict,
        climate_summary: dict,
) -> dict:
    """
    Combine user soil values with climate summary
     into model-ready input.

     The current trained model expects:
     n, p, k, temperature, humidity, ph, rainfall.

     Humidity remains user-provided for now because
     the first Open-Meteo prototype does not fetch humidity.
    Parameters
    ----------
    soil_input
    climate_summary

    Returns
    -------

    """
    required_soil_keys = ["n", "p", "k", "ph"]

    missing_keys = [key for key in required_soil_keys
                    if key not in soil_input]

    if missing_keys:
        raise ValueError(f"Missing required soil inputs: {missing_keys}")

    return {
        "n": soil_input["n"],
        "p": soil_input["p"],
        "k": soil_input["k"],
        "temperature": climate_summary["mean_temperature"],
        "humidity": soil_input.get("humidity", 70.0),
        "ph": soil_input["ph"],
        "rainfall": climate_summary["total_rainfall"],

    }