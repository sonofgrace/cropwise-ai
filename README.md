\# CropWise AI: Crop Recommendation System



\## Overview



CropWise AI is a machine learning project that recommends suitable crops based on soil nutrients and climate conditions. The system uses nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall values to predict the most suitable crop among 22 possible crop classes.



\## Problem Statement



Farmers, agricultural extension workers, and agritech platforms often need quick, data-informed crop suitability recommendations. This project builds a crop recommendation model using soil and climate variables, with the long-term goal of turning it into an explainable and deployable decision-support tool.



\## Dataset



The dataset contains 2,200 records and 8 columns:



\- N: Nitrogen content

\- P: Phosphorus content

\- K: Potassium content

\- temperature: Temperature in degrees Celsius

\- humidity: Relative humidity

\- ph: Soil pH

\- rainfall: Rainfall measurement

\- label: Recommended crop



There are 22 crop classes, with 100 records per crop.



\## Project Goals



\- Perform exploratory data analysis.

\- Identify the most predictive individual feature.

\- Build baseline and advanced classification models.

\- Evaluate models using appropriate multi-class classification metrics.

\- Add explainability using feature importance and SHAP.

\- Deploy the model as an interactive Streamlit app.

\- Extend the project toward a high-end agriculture decision-support portfolio project.



\## Current Stage



Project setup completed.

## Initial Data Validation

The dataset contains 2,200 observations and 8 columns. Seven columns are input features, while the target variable is the crop label.

The input features are:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

The target variable contains 22 crop classes, with 100 observations per crop. The dataset has no missing values and no duplicated rows.

Because the classes are perfectly balanced, accuracy can be reported, but macro-F1 will be used as the primary evaluation metric to give equal importance to every crop class.

## Exploratory Data Analysis

Exploratory data analysis was performed to understand the structure, distribution, and separability of the crop recommendation dataset.

Key findings:

- The dataset is perfectly balanced across 22 crop classes.
- All input features are numeric.
- There are no missing values or duplicate records.
- Feature ranges differ considerably, making scaling important for some machine learning models.
- Crop-wise feature summaries show that different crops occupy distinct soil and climate profiles.
- PCA visualization shows partial class separation, suggesting that the full feature space is useful for crop classification.

Generated EDA visuals include:

- Crop class distribution chart
- Feature distribution plots
- Crop-wise boxplots
- Correlation heatmap
- Crop feature profile heatmap
- PCA class-separation plot


## Single-Feature Baseline

To establish a simple baseline, each input feature was evaluated individually using a Logistic Regression classifier and 5-fold Stratified Cross-Validation.

The primary metric was macro-F1, which gives equal importance to all 22 crop classes.

The best single predictive feature was:

```python
best_predictive_feature = {"rainfall": 0.2582}


## Model Training and Comparison

Several supervised machine learning models were trained using all soil and climate features:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- Support Vector Machine

Models were evaluated using 5-fold Stratified Cross-Validation on the training data. The primary evaluation metric was macro-F1.

Tree-based ensemble models performed best, especially Random Forest and Extra Trees. These models are well-suited to this dataset because they can capture non-linear relationships and interactions between soil nutrients and climate conditions.

The best model was saved as:

```text
models/crop_model.joblib


## Hyperparameter Tuning

The strongest baseline models from the model comparison stage were tuned using `RandomizedSearchCV`.

The tuned models included:

- Random Forest
- Extra Trees

The search optimized macro-F1 using 5-fold Stratified Cross-Validation.

The final tuned model was saved as:

```text
models/crop_model_tuned.joblib


## Explainability

Explainability was added to make the crop recommendation system more transparent and interpretable.

The project includes:

- Global feature importance from the trained tree-based model
- Permutation importance using macro-F1
- Crop-profile matching for local prediction explanation
- Top-N crop recommendations with probabilities

The explanation layer helps answer not only:

```text
What crop was recommended?

but also:

Why was this crop recommended?
Which input values were most consistent with the predicted crop profile?
What were the next-best crop alternatives?