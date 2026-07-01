from __future__ import annotations

import pandas as pd
import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_historical_weather(
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
) -> pd.DataFrame:
    """
    Fetch historical daily weather data from open-meteo.
    Parameters
    ----------
    latitude:
        Location latitude.
    longitude:
        Location longitude.
    start_date:
        Start date in YYYY-MM-DD format.
    end_date:
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Daily historical weather dataframe

    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(
            [
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
            ]
        ),
        "timezone": "auto",
    }

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if "daily" not in data:
        raise ValueError("No daily weather data found in open-meteo response.")

    weather_df = pd.DataFrame(data["daily"])

    if weather_df.empty:
        raise ValueError("Open-Meteo returned an empty weather dataframe.")

    return weather_df


if __name__ == "__main__":
    # Lagos, Nigeria approximate coordinates
    df = fetch_historical_weather(
        latitude=6.5244,
        longitude=3.3792,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    print(df.head())
    print(df.describe())