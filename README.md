# # CropWise AI: Explainable and Climate-Aware Crop Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Quick Links

- Live app: https://cropwise-ai.streamlit.app/
- GitHub repository: https://github.com/sonofgrace/cropwise-ai
- Main app file: `app/streamlit_app.py`
- Training script: `src/train.py`
- Prediction module: `src/predict.py`
- Explainability module: `src/explain.py`

## Live Demo

Streamlit app: https://cropwise-ai.streamlit.app/

## Project Summary

CropWise AI is an end-to-end machine learning crop recommendation system that recommends suitable crops using soil nutrient and climate variables.

The system predicts the most suitable crop from nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall values. It also provides model confidence, top crop alternatives, human-readable explanations, and downloadable recommendation reports.

The upgraded version includes a location-aware climate-smart recommendation tab. Users can enter latitude, longitude, soil values, and a climate period. The app then fetches historical weather data, summarizes rainfall and temperature patterns, estimates climate risk, and generates a climate-aware crop recommendation.
## App Preview

![CropWise AI App](reports/figures/streamlit_app_screenshot.png)

## Project Highlights

* Built a complete machine learning workflow from data validation to deployment.
* Performed exploratory data analysis on soil and climate variables.
* Identified the strongest single predictive feature for crop classification.
* Compared multiple supervised machine learning models.
* Tuned ensemble models using cross-validation and macro-F1 optimization.
* Added explainability through feature importance, permutation importance, SHAP analysis, and crop-profile matching.
* Deployed an interactive Streamlit web application.
* Added automated tests for data, model loading, prediction, and explanation functions.
* Designed the project as a portfolio-ready ML decision-support system.

## Climate-Aware v2 Features

The location-aware version extends the original crop recommendation system by adding external climate enrichment.

New v2 features include:

- Latitude and longitude-based climate lookup
- Historical weather retrieval using Open-Meteo
- Rainfall and temperature trend charts
- Climate summary metrics
- Dry-day, wet-day, and heavy-rain-day counts
- Rule-based climate risk scoring
- Climate risk warnings
- Climate-aware crop recommendation
- Downloadable climate-aware recommendation report

This upgrade makes the project more realistic by connecting the original machine learning model with external climate data.

## Problem Statement

Farmers, agricultural extension workers, and agritech platforms often need quick, data-informed support when choosing suitable crops for specific soil and climate conditions.

CropWise AI addresses this by building a machine learning-based crop recommendation system that uses soil nutrient and climate inputs to recommend appropriate crops. The project is designed as a decision-support tool, not a replacement for professional agronomic advice or local field validation.

## Dataset

The dataset contains 2,200 observations and 8 columns.

## External Climate Data

The location-aware version uses historical weather data from Open-Meteo.

The app fetches daily weather variables such as:

- Mean temperature
- Maximum temperature
- Minimum temperature
- Precipitation
- Rainfall

These values are summarized into crop-relevant climate features, including total rainfall, mean temperature, dry days, wet days, heavy rain days, and rainfall variability.

The external climate layer is used for decision-support enrichment. It does not replace agronomic field validation.

### Features

| Feature       | Description                    |
| ------------- | ------------------------------ |
| `N` / `n`     | Nitrogen content in soil       |
| `P` / `p`     | Phosphorus content in soil     |
| `K` / `k`     | Potassium content in soil      |
| `temperature` | Temperature in degrees Celsius |
| `humidity`    | Relative humidity              |
| `ph`          | Soil pH                        |
| `rainfall`    | Rainfall measurement           |

### Target

| Target  | Description      |
| ------- | ---------------- |
| `label` | Recommended crop |

The dataset contains 22 crop classes, with 100 observations per crop. This makes the target distribution perfectly balanced.

## Initial Data Validation

Initial data validation showed that:

* The dataset contains 2,200 rows and 8 columns.
* There are 7 input features and 1 target variable.
* All input features are numeric.
* The target variable contains 22 crop classes.
* Each crop class has exactly 100 observations.
* There are no missing values.
* There are no duplicated rows.

Because the classes are balanced, accuracy is useful, but macro-F1 was selected as the primary evaluation metric because it gives equal importance to every crop class.

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the structure, distribution, and separability of the crop recommendation dataset.

Key findings:

* The dataset is perfectly balanced across all 22 crop classes.
* All predictor variables are numeric.
* Feature ranges differ considerably, making scaling important for models such as Logistic Regression, KNN, and SVM.
* Crop-wise feature summaries show that different crops occupy distinct soil and climate profiles.
* Rainfall, humidity, potassium, phosphorus, and temperature provide useful signals for distinguishing several crop classes.
* PCA visualization shows partial crop-class separation, suggesting that the full feature space is useful for classification.

Generated EDA visuals include:

* Crop class distribution chart
* Feature distribution plots
* Crop-wise boxplots
* Correlation heatmap
* Crop feature profile heatmap
* PCA class-separation plot

## Single-Feature Baseline

To establish a simple baseline, each input feature was evaluated individually using a Logistic Regression classifier and 5-fold Stratified Cross-Validation.

The primary metric was macro-F1.

The best single predictive feature was:

```python
best_predictive_feature = {"rainfall": 0.2582}
```

Rainfall had the strongest individual predictive signal. However, its macro-F1 score was still low compared with full-feature models, showing that crop recommendation requires multiple soil and climate variables rather than a single feature alone.

## Model Training and Comparison

Several supervised machine learning models were trained using all soil and climate features:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* Support Vector Machine

Models were evaluated using 5-fold Stratified Cross-Validation on the training data.

The primary metric was:

```text
macro-F1
```

Tree-based ensemble models performed best, especially Random Forest and Extra Trees. These models are well-suited to this dataset because they can capture non-linear relationships and interactions between soil nutrients and climate conditions.

The trained baseline model was saved as:

```text
models/crop_model.joblib
```

## Hyperparameter Tuning

The strongest baseline models from the model comparison stage were tuned using `RandomizedSearchCV`.

The tuned models included:

* Random Forest
* Extra Trees

The hyperparameter search optimized macro-F1 using 5-fold Stratified Cross-Validation.

The final tuned model was saved as:

```text
models/crop_model_tuned.joblib
```

The tuned model metadata was saved as:

```text
models/tuned_model_metadata.json
```

Hyperparameter tuning helped validate the robustness of the final model, even though the untuned ensemble models already performed strongly.

## Explainability

Explainability was added to make the crop recommendation system more transparent and interpretable.

The project includes:

* Global feature importance from the trained tree-based model
* Permutation importance using macro-F1
* Crop-profile matching for local prediction explanation
* Top-N crop recommendations with probabilities
* SHAP class-level explanations

The explanation layer helps answer:

```text
What crop was recommended?
```

It also helps answer:

```text
Why was this crop recommended?
Which input values were most consistent with the predicted crop profile?
What were the next-best crop alternatives?
```

## SHAP Explainability

SHAP was added for class-level model interpretation. Since crop recommendation is a multi-class classification problem, SHAP explanations are generated separately for each crop class.

For each selected crop, SHAP summary plots show which features push predictions toward or away from that crop. This provides a deeper explanation layer beyond standard feature importance and helps make the recommendation system more transparent.

SHAP is used in the analysis workflow, while the deployed Streamlit application uses a lighter explanation system based on probabilities and crop-profile matching.

## Streamlit Application

A Streamlit web application was built to make the model interactive and user-friendly.

The app contains two main tabs:

### 1. Manual Recommendation

Users manually enter:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- Soil pH
- Rainfall

The app returns:

- Recommended crop
- Model confidence
- Top 3 crop recommendations
- Probability chart
- Human-readable explanation
- Downloadable recommendation report

### 2. Location-Aware Recommendation

Users enter:

- Latitude
- Longitude
- Climate start date
- Climate end date
- Nitrogen
- Phosphorus
- Potassium
- Soil pH
- Humidity

The app fetches historical weather data and returns:

- Historical climate summary
- Rainfall trend chart
- Temperature trend chart
- Recommended crop
- Top 3 crop recommendations
- Climate risk score
- Climate risk level
- Climate warnings
- Downloadable climate-aware report

To run the app locally:

```bash
streamlit run app/streamlit_app.py
```

## Project Architecture

```text
cropwise-ai/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── Crop_recommendation.csv
│   └── processed/
│       └── crop_recommendation_clean.csv
│
├── models/
│   ├── crop_model.joblib
│   ├── crop_model_tuned.joblib
│   ├── model_metadata.json
│   └── tuned_model_metadata.json
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_single_feature_baseline.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_explainability.ipynb
│
├── reports/
│   ├── figures/
│   ├── classification_report.txt
│   ├── model_comparison_cv_results.csv
│   ├── hyperparameter_tuning_results.csv
│   ├── global_feature_importance.csv
│   └── permutation_importance.csv
│
├── src/
│   ├── external/
│   │   ├── __init__.py
│   │   └── open_meteo.py
│   │
│   ├── features/
│   │    ├── __init__.py
│   │    └── climate_features.py
│   │
│   ├──risks/
│   │   ├── __init__.py
│   │   └── yield_risk.py
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── validate_data.py
│   ├── train.py
│   ├── tune_model.py
│   ├── evaluate.py
│   ├── predict.py
│   └── explain.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_prediction.py
│   ├── test_explain.py
│   ├── test_climate_features.py
│   ├── test_open_meteo.py
│   └── test_yield_risk.py
│
├── .github/
│   └── workflows/
│       └── python-ci.yml
│
├── check_climate_v2.py
│
├── requirements.txt
├── runtime.txt
├── pyproject.toml
├── pytest.ini
├── README.md
└── .gitignore
```

## Testing and Code Quality

The project includes automated tests using `pytest`.

Test coverage includes:

* Data file availability
* Required column validation
* Missing value checks
* Target class validation
* Model loading
* Prediction output structure
* Probability consistency
* Explainability output structure

To run tests locally:

```bash
pytest
```

Code quality tools used in the project include:

```bash
ruff check .
black .
```

A GitHub Actions workflow is included to run the project pipeline and tests automatically on push or pull request.

## Results

The single-feature baseline showed that rainfall had the strongest individual predictive performance among all input features. However, its score was much lower than full-feature models, confirming that crop suitability depends on a combination of soil and climate variables.

Using all seven features, tree-based ensemble models achieved the strongest performance. Random Forest and Extra Trees performed especially well because they can capture non-linear interactions among soil nutrients, rainfall, humidity, temperature, and pH.

| Stage                   | Method                                           | Key Finding                                                |
| ----------------------- | ------------------------------------------------ | ---------------------------------------------------------- |
| Single-feature baseline | Logistic Regression                              | Rainfall was the strongest individual predictor            |
| Full model comparison   | Multiple classifiers                             | Ensemble tree-based models performed best                  |
| Hyperparameter tuning   | RandomizedSearchCV                               | Final tuned model selected using macro-F1                  |
| Explainability          | Feature importance, permutation importance, SHAP | Multiple soil and climate variables influenced predictions |
| Deployment              | Streamlit                                        | Interactive crop recommendation app deployed online        |

## Limitations

Although the model performs strongly on the available dataset, the dataset is small, clean, balanced, and highly structured. Real-world agricultural decisions are more complex.

The current model does not include:

* Geographic location
* Soil type
* Planting season
* Seed variety
* Pest and disease pressure
* Irrigation access
* Market demand
* Actual crop yield
* Farmer management practices

Therefore, CropWise AI should be interpreted as a crop suitability decision-support tool, not as a replacement for professional agronomic advice, local soil testing, seasonal planning, or field validation.

## Climate-Enriched Recommendation Note

The location-aware feature enriches user inputs with historical weather data from an external API. Because the original crop recommendation model was trained on a structured benchmark dataset, external climate values may not perfectly match the original training distribution.

Therefore, the climate-aware recommendation should be interpreted as an experimental decision-support extension. It should not be used as final agronomic advice without local soil testing, seasonal context, seed variety information, irrigation assessment, pest and disease evaluation, and field validation.

## Future Improvements

Planned improvements include:

- Add location search by city or region name.
- Add interactive map-based location selection.
- Integrate NASA POWER as a second climate data source.
- Add seasonal crop calendars.
- Add soil type and soil texture information.
- Add crop-specific climate suitability thresholds.
- Add FAOSTAT or national crop production data.
- Build a trained yield-risk model using real yield outcomes.
- Add model monitoring for input drift.
- Add database support for storing user recommendations.
- Add user authentication for saved recommendation histories.
- Extend the app into a climate-smart agriculture decision-support platform.
## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/sonofgrace/cropwise-ai.git
cd cropwise-ai
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the processed dataset:

```bash
python -m src.preprocessing
```

Train the model:

```bash
python -m src.train
```

Optional: tune the model:

```bash
python -m src.tune_model
```

Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Deployment Notes

For Streamlit deployment, the project uses a lightweight `requirements.txt` containing only the dependencies needed by the deployed app.

SHAP is used for analysis and notebook explainability, but it is not required for the deployed Streamlit app.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- Requests
- Open-Meteo API
- SHAP
- Pytest
- Ruff
- Black
- GitHub Actions

## Project Status

The project is complete as an end-to-end portfolio version and is currently deployed as a Streamlit application.

## Author

Dr. Benedict U. Nweke
GitHub: https://github.com/sonofgrace
