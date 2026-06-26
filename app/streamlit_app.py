import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


# Allow Streamlit to import from src when app is run from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predict import predict_crop_with_probabilities, get_top_n_recommendations
from src.explain import explain_prediction, format_explanation
from src.predict import load_model


@st.cache_resource
def get_cached_model():
    return load_model()

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
        ]
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
    This recommendation is generated by a machine learning model trained on a structured crop recommendation dataset. It should be used as a decision-support tool, not as a substitute for professional agronomic advice, local soil testing, seasonal planning, or field validation.

"""

    return report


def main() -> None:
    st.title("🌱 CropWise AI")
    st.subheader("Explainable Crop Recommendation System")

    st.write(
        """
        
         CropWise AI recommends suitable crops using soil nutrient and climate variables.
        The model uses nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall
        to recommend one of 22 crop classes.
        """
    )

    with st.expander("Project context"):
        st.write(
            """
            
            This application is part of an end-to-end machine learning portfolio project.
            It includes data validation, exploratory data analysis, baseline modeling,
            model comparison, hyperparameter tuning, explainability, and deployment.
            """
        )

        input_data = build_input_data()

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
                st.success(f"Recommended crop: **{prediction_result['prediction'].title()}")
                st.metric(
                    label = "Model confidence",
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
            st.info("Adjust the sidebar values and click **Recommend Crop** to generate a recommendation.")

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


if __name__ == "__main__":
    main()