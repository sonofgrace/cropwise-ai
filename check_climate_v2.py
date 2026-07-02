from src.external.open_meteo import fetch_historical_weather
from src.features.climate_features import (
    create_climate_summary,
    create_cropwise_input_from_climate,
)
from src.predict import predict_crop_with_probabilities
from src.risk.yield_risk import calculate_climate_risk_score

weather_df = fetch_historical_weather(
    latitude=6.5244,
    longitude=3.3792,
    start_date="2024-01-01",
    end_date="2024-12-31",
)

climate_summary = create_climate_summary(weather_df)

soil_input = {
    "n": 90,
    "p": 42,
    "k": 43,
    "ph": 6.5,
    "humidity": 82.0,
}

model_input = create_cropwise_input_from_climate(
    soil_input=soil_input,
    climate_summary=climate_summary,
)

prediction = predict_crop_with_probabilities(model_input)
risk = calculate_climate_risk_score(climate_summary)

print("Climate summary:")
print(climate_summary)

print("\nModel input:")
print(model_input)

print("\nPrediction:")
print(prediction["prediction"])

print("\nRisk:")
print(risk)