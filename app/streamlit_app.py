import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# Allow Streamlit to import from src when app is run from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.explain import explain_prediction, format_explanation  # noqa: E402
from src.predict import (  # noqa: E402
    get_top_n_recommendations,
    load_model,
    predict_crop_with_probabilities,
)

from src.external.open_meteo import fetch_historical_weather
from src.features.climate_features import (
create_climate_summary,
create_cropwise_input_from_climate,
)
from src.risk.yield_risk import calculate_climate_risk_score


@st.cache_resource
def get_cached_model():
    return load_model()


@st.cache_data(show_spinner=False)
def get_cached_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
):
    """Fetch and cache historical weather data."""
    return fetch_historical_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )


st.set_page_config(
    page_title="CropWise AI",
    page_icon="🌱",
    layout="wide",
)


def build_input_data() -> dict:
    """
    Build user input dictionary from sidebar controls.
    """

    preset = st.sidebar.selectbox(
        "Choose a sample preset",
        options=[
            "Custom",
            "Rice-like conditions",
            "Maize-like conditions",
            "Coffee-like conditions",
        ],
        key="manual_crop_preset_selectbox",
    )

    presets = {
        "Custom": {
            "n": 90,
            "p": 42,
            "k": 43,
            "temperature": 20.8,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.9,
        },
        "Rice-like conditions": {
            "n": 90,
            "p": 42,
            "k": 43,
            "temperature": 20.8,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.9,
        },
        "Maize-like conditions": {
            "n": 70,
            "p": 45,
            "k": 20,
            "temperature": 26.0,
            "humidity": 65.0,
            "ph": 6.8,
            "rainfall": 85.0,
        },
        "Coffee-like conditions": {
            "n": 100,
            "p": 30,
            "k": 30,
            "temperature": 25.5,
            "humidity": 58.0,
            "ph": 6.3,
            "rainfall": 150.0,
        },
    }

    default_values = presets[preset]


    st.sidebar.header("Soil and Climate Inputs")

    n = st.sidebar.slider(
        "Nitrogen (N)",
        min_value=0,
        max_value=150,
        value=default_values["n"],
        step=1,
        help="Nitrogen content in soil"
    )

    p = st.sidebar.slider(
        "Phosphorus (P)",
        min_value=0,
        max_value=150,
        value=default_values["p"],
        step=1,
        help="Phosphorus content in soil"
    )

    k = st.sidebar.slider(
        "Potassium (K)",
        min_value=0,
        max_value=220,
        value=default_values["k"],
        step=1,
        help="Potassium content in soil"
    )

    temperature = st.sidebar.slider(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=default_values["temperature"],
        step=0.1,
        help="Average temperature"
    )

    humidity = st.sidebar.slider(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=default_values["humidity"],
        step=0.1,
        help="Relative humidity"
    )

    ph = st.sidebar.slider(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=default_values["ph"],
        step=0.1,
        help="Soil acidity/alkalinity"
    )

    rainfall = st.sidebar.slider(
        "Rainfall",
        min_value=0.0,
        max_value=350.0,
        value=default_values["rainfall"],
        step=0.1,
        help="Rainfall level"
    )

    return {
        "n": n,
        "p": p,
        "k": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }


def create_probability_dataframe(probabilities: dict) -> pd.DataFrame:
    """
    Convert probability dictionary to dataframe.
    Parameters
    """

    probability_df = pd.DataFrame(
        {
            "crop": list(probabilities.keys()),
            "probability": list(probabilities.values())
        }
    )

    probability_df["probability_percent"] = probability_df["probability"] * 100

    return probability_df


def generate_report(
        input_data: dict,
        prediction_result: dict,
        explanation_text: str) -> str:
    """
    Generate downloadable text report.
    Parameters
    ----------
    input_data
    prediction_result
    explanation_text

    Returns
    -------

    """

    top_recommendations = list(prediction_result["probabilities"].items())[:3]

    report = f"""
    CropWise AI Recommendation Report
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Recommended Crop
    ----------------
    {prediction_result["prediction"]}
    
    Top 3 Recommendations
    ---------------------
"""

    for crop, probability in top_recommendations:
        report += f"{crop}: {probability:.2%}\n"

    report += """
    Input Values
    ------------
    """

    for feature, value in input_data.items():
        report += f"{feature}:n{value}\n"

    report += f"""
    
    Explanation
    -----------
    {explanation_text}
    
    Important Note
    --------------
    This recommendation is generated by a machine learning model trained
     on a structured crop recommendation dataset. 
     It should be used as a decision-support tool,
      not as a substitute for professional agronomic advice,
       local soil testing, seasonal planning, or field validation.

"""

    return report

def render_location_aware_tab():
    """Render the location-aware climate-smart recommendation tab."""
    st.header("Location-Aware Climate-Smart Recommendation")

    st.info(
        "This experimental feature fetches historical climate data for a selected "
        "location and combines it with soil inputs to generate a climate-aware "
        "crop recommendation."
    )

    st.subheader("1. Location and Climate Period")

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=6.5244,
            step=0.0001,
            format="%.4f",
            help="Example: Lagos, Nigeria ≈ 6.5244",
        )

        start_date = st.date_input(
            "Start date",
            value=pd.to_datetime("2024-01-01"),
        )

    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=3.3792,
            step=0.0001,
            format="%.4f",
            help="Example: Lagos, Nigeria ≈ 3.3792",
        )

        end_date = st.date_input(
            "End date",
            value=pd.to_datetime("2024-12-31"),
        )

    st.subheader("2. Soil Inputs")

    soil_col1, soil_col2, soil_col3 = st.columns(3)

    with soil_col1:
        n_value = st.slider("Nitrogen (N)", 0, 150, 90)
        ph_value = st.slider("Soil pH", 3.5, 10.0, 6.5, 0.1)

    with soil_col2:
        p_value = st.slider("Phosphorus (P)", 0, 150, 42)
        humidity_value = st.slider("Humidity (%)", 0.0, 100.0, 82.0, 0.1)

    with soil_col3:
        k_value = st.slider("Potassium (K)", 0, 250, 43)

    soil_input = {
        "n": n_value,
        "p": p_value,
        "k": k_value,
        "ph": ph_value,
        "humidity": humidity_value,
    }

    st.subheader("3. Climate-Aware Prediction")

    fetch_button = st.button(
        "Fetch Climate Data and Recommend Crop",
        type="primary",
        key="location_aware_predict_button",
    )

    if fetch_button:
        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
            return

        try:
            with st.spinner("Fetching historical climate data..."):
                weather_df = get_cached_weather(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=str(start_date),
                    end_date=str(end_date),
                )

            climate_summary = create_climate_summary(weather_df)

            model_input = create_cropwise_input_from_climate(
                soil_input=soil_input,
                climate_summary=climate_summary,
            )

            prediction_result = predict_crop_with_probabilities(model_input)
            top_recommendations = get_top_n_recommendations(model_input, n=3)
            climate_risk = calculate_climate_risk_score(climate_summary)

            st.success("Climate data fetched and recommendation generated.")

            st.subheader("Climate Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric(
                    "Mean Temperature",
                    f"{climate_summary['mean_temperature']:.1f} °C",
                )
                st.metric(
                    "Total Rainfall",
                    f"{climate_summary['total_rainfall']:.1f} mm",
                )

            with summary_col2:
                st.metric(
                    "Dry Days",
                    climate_summary["dry_days"],
                )
                st.metric(
                    "Wet Days",
                    climate_summary["wet_days"],
                )

            with summary_col3:
                st.metric(
                    "Heavy Rain Days",
                    climate_summary["heavy_rain_days"],
                )
                st.metric(
                    "Climate Risk",
                    climate_risk["risk_level"],
                )

            st.subheader("Recommended Crop")

            crop_name = prediction_result["prediction"]
            probabilities = prediction_result.get("probabilities", {})

            confidence = prediction_result.get("confidence")

            if confidence is None:
                if probabilities:
                    confidence = probabilities.get(crop_name, max(probabilities.values()))
                else:
                    confidence = 0

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Recommended Crop", crop_name.title())

            with result_col2:
                st.metric("Model Confidence", f"{confidence:.2%}")

            st.subheader("Top 3 Recommendations")

            top_df = pd.DataFrame(top_recommendations)
            top_df["crop"] = top_df["crop"].str.title()
            top_df["probability"] = top_df["probability"].map(lambda x: f"{x:.2%}")

            st.dataframe(top_df, width="stretch")

            st.subheader("Climate Risk Warnings")

            if climate_risk["warnings"]:
                for warning in climate_risk["warnings"]:
                    st.warning(warning)
            else:
                st.success("No major climate risk warning detected by the rule-based layer.")

            st.subheader("Model Input Used")

            model_input_df = pd.DataFrame([model_input])
            st.dataframe(model_input_df, width="stretch")

            st.subheader("Historical Weather Preview")

            st.dataframe(weather_df.head(10), width="stretch")

            report_text = f"""
CropWise AI v2 Climate-Aware Recommendation Report

Location
Latitude: {latitude}
Longitude: {longitude}

Climate Period
Start date: {start_date}
End date: {end_date}

Climate Summary
Mean temperature: {climate_summary['mean_temperature']:.2f} °C
Maximum temperature: {climate_summary['max_temperature']:.2f} °C
Minimum temperature: {climate_summary['min_temperature']:.2f} °C
Total rainfall: {climate_summary['total_rainfall']:.2f} mm
Mean daily rainfall: {climate_summary['mean_daily_rainfall']:.2f} mm
Dry days: {climate_summary['dry_days']}
Wet days: {climate_summary['wet_days']}
Heavy rain days: {climate_summary['heavy_rain_days']}

Soil Inputs
Nitrogen: {n_value}
Phosphorus: {p_value}
Potassium: {k_value}
pH: {ph_value}
Humidity: {humidity_value}

Recommendation
Recommended crop: {crop_name.title()}
Model confidence: {confidence:.2%}

Climate Risk
Risk score: {climate_risk.get('risk_score', climate_risk.get('score', 'N/A'))}
Risk level: {climate_risk.get('risk_level', 'N/A')}

Warnings
{chr(10).join(climate_risk.get('warnings', [])) if climate_risk.get('warnings') else "No major warnings detected."}

Important Note
This location-aware feature is experimental. The original model was trained on a structured benchmark dataset, while external climate API data may not perfectly match the original training distribution. Interpret this result as decision support, not final agronomic advice.
"""

            st.download_button(
                label="Download Climate-Aware Report",
                data=report_text,
                file_name="cropwise_climate_aware_report.txt",
                mime="text/plain",
            )

            st.caption(
                "Experimental note: This feature enriches the original model with "
                "external historical weather data. Results should be interpreted with caution."
            )

        except Exception as error:
            st.error("An error occurred while generating the climate-aware recommendation.")
            st.exception(error)


def main() -> None:
    st.title("🌱 CropWise AI")
    st.subheader("Explainable Crop Recommendation System")

    tab1, tab2 = st.tabs(
        [
            "Manual Recommendation",
            "Location-Aware Recommendation",
        ]
    )

    with tab1:
        st.markdown(
            "Enter soil and climate values manually to generate"
            "a crop recommendation."
        )

        input_data = build_input_data()

        st.write(
            """

             CropWise AI recommends suitable crops using soil nutrient and
              climate variables.
            The model uses nitrogen, phosphorus, potassium, temperature,
             humidity, pH, and rainfall
            to recommend one of 22 crop classes.
            """
        )

        with st.expander("Project context"):
            st.write(
                """

                This application is part of an end-to-end machine learning
                 portfolio project. It includes data validation,
                  exploratory data analysis, baseline modeling,
                model comparison, hyperparameter tuning, explainability,
                 and deployment.
                """
            )


            col_input, col_result = st.columns([1, 1])

            with col_input:
                st.markdown("### Current Input Values")
                input_df = pd.DataFrame([input_data])
                st.dataframe(input_df, use_container_width=True)

            predict_button = st.sidebar.button("Recommend Crop", type="primary")

            if predict_button:
                prediction_result = predict_crop_with_probabilities(input_data)
                top_recommendations = get_top_n_recommendations(input_data, n=3)
                explanation = explain_prediction(input_data, top_n=3)
                explanation_text = format_explanation(explanation)

                probability_df = create_probability_dataframe(
                    prediction_result["probabilities"]
                )

                top_probability = probability_df.iloc[0]["probability"]

                with col_result:
                    st.markdown("### Recommendation Result")
                    st.success(
                        f"Recommended crop: **{prediction_result['prediction'].title()}**")
                    st.metric(
                        label="Model confidence",
                        value=f"{top_probability:.1%}"
                    )

                    st.markdown("### Top 3 Recommendations")
                    top_df = pd.DataFrame(top_recommendations)
                    top_df["probability"] = top_df["probability"].map(lambda x: f"{x:.1%}")
                    st.dataframe(top_df, use_container_width=True)

                    st.markdown("---")

                    st.markdown("### Prediction Probability Chart")

                    top_10_probability_df = probability_df.head(10)

                    fig = px.bar(
                        top_10_probability_df,
                        x="probability_percent",
                        y="crop",
                        orientation="h",
                        title="Top Crop Probabilities",
                        labels={
                            "probability_percent": "Probability (%)",
                            "crop": "Crop",
                        },
                    )

                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        height=500,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### Explanation")
                    st.info(explanation_text)

                    st.markdown("### Important caution")
                    st.warning(
                        """
                This model should be used as a decision-support tool only.
                Real-world crop selection should also consider local soil testing,
                pest and disease pressure, season, seed variety, irrigation access,
                market demand, and expert agronomic advice.
                """
                    )

                    report_text = generate_report(
                        input_data=input_data,
                        prediction_result=prediction_result,
                        explanation_text=explanation_text,
                    )

                    st.download_button(
                        label="Download Recommended Report",
                        data=report_text,
                        file_name="cropwise_recommendation_report.txt",
                        mime="text/plain",
                    )

            else:
                st.info("Adjust the sidebar values and click **Recommend Crop**"
                        " to generate a recommendation.")

            st.markdown("---")

            st.markdown(
                """

                ### Model Inputs

                - **N**: Nitrogen content
                - **P**: Phosphorus content
                - **K**: Potassium content
                - **Temperature**: Average temperature
                - **Humidity**: Relative humidity
                - **pH**: Soil pH
                - **Rainfall**: Rainfall level
                """
            )

            st.markdown(
                """
                ### Portfolio Notes

                This project demonstrates:

                - Multi-class classification
                - Cross-validation
                - Model comparison
                - Hyperparameter tuning
                - Explainability
                - Streamlit deployment
                - ML product thinking
                """
            )


    with tab2:
        render_location_aware_tab()



if __name__ == "__main__":
    main()