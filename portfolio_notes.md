CropWise AI — Explainable Crop Recommendation System
Python | Pandas | Scikit-learn | Streamlit | Plotly | SHAP | Pytest | GitHub Actions

Built and deployed an end-to-end machine learning crop recommendation system that predicts suitable crops from soil nutrient and climate variables, including nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

Key contributions:
- Performed data validation and exploratory data analysis on a balanced 2,200-record crop recommendation dataset with 22 crop classes.
- Built a single-feature baseline model and identified rainfall as the strongest individual predictor using macro-F1.
- Compared multiple classification models, including Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and SVM.
- Tuned ensemble models using RandomizedSearchCV and 5-fold Stratified Cross-Validation.
- Added explainability using feature importance, permutation importance, SHAP analysis, and local crop-profile matching.
- Developed a Streamlit web app that provides crop recommendations, confidence scores, top-3 alternatives, probability charts, human-readable explanations, and downloadable reports.
- Added automated tests for data validation, model loading, predictions, probability consistency, and explanation outputs.

Live app: https://cropwise-ai.streamlit.app/
GitHub: https://github.com/sonofgrace/cropwise-ai

CropWise AI — Built and deployed an explainable crop recommendation ML app using Python, scikit-learn, and Streamlit. Implemented data validation, EDA, single-feature baseline modeling, multi-model comparison, hyperparameter tuning, explainability, automated tests, and a downloadable recommendation report. Live app: https://cropwise-ai.streamlit.app/

- Built CropWise AI, an end-to-end ML crop recommendation system using soil nutrient and climate variables across 22 crop classes.
- Compared Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and SVM models using macro-F1 and cross-validation.
- Added explainability through feature importance, permutation importance, SHAP analysis, and crop-profile matching.
- Deployed an interactive Streamlit app with top-3 recommendations, confidence scores, probability charts, explanations, and downloadable reports.

I just completed CropWise AI, an end-to-end machine learning crop recommendation system.

CropWise AI recommends suitable crops using soil nutrient and climate variables such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

What I built:
- Data validation workflow
- Exploratory data analysis
- Single-feature baseline modeling
- Multi-model comparison
- Hyperparameter tuning
- Explainability with feature importance, permutation importance, and SHAP
- Streamlit web application
- Top-3 crop recommendations
- Confidence scores and probability charts
- Downloadable recommendation reports
- Automated tests and GitHub Actions workflow

This project helped me strengthen my practical understanding of classification workflows, model evaluation, explainable AI, and ML deployment.

Live app: https://cropwise-ai.streamlit.app/
GitHub repo: https://github.com/sonofgrace/cropwise-ai

CropWise AI

An end-to-end machine learning application for crop recommendation using soil nutrient and climate variables. The project includes EDA, single-feature baseline modeling, model comparison, hyperparameter tuning, explainability, automated tests, and Streamlit deployment.

Tech stack: Python, Pandas, Scikit-learn, Streamlit, Plotly, SHAP, Pytest, GitHub Actions

Live Demo: https://cropwise-ai.streamlit.app/
GitHub: https://github.com/sonofgrace/cropwise-ai

## Interview explanation

# Tell me about CropWise AI
CropWise AI is an end-to-end machine learning crop recommendation project. The goal was to recommend suitable crops based on soil nutrient and climate features such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall.

I started by validating the dataset, checking missing values, duplicates, class balance, and feature ranges. Then I performed EDA to understand how different crops relate to nutrient and climate profiles.

I first built a single-feature baseline and found that rainfall was the strongest individual predictor, but the score was not strong enough for reliable prediction. That justified moving to full-feature modeling.

I compared several classifiers, including Logistic Regression, KNN, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and SVM. Ensemble tree-based models performed best, so I tuned Random Forest and Extra Trees using RandomizedSearchCV with macro-F1 as the main metric.

To make the project more useful, I added explainability through global feature importance, permutation importance, SHAP analysis, and local crop-profile matching. Finally, I deployed the model as a Streamlit app that returns a recommended crop, confidence score, top-3 alternatives, probability chart, explanation, and downloadable report.

The main limitation is that the dataset is clean and structured, so real-world deployment would require location-specific soil data, seasonality, weather history, crop yield outcomes, and agronomic validation.

# Why macro-F1?
I used macro-F1 because the task is a multi-class classification problem with 22 crop classes. Although the dataset is balanced, macro-F1 is still useful because it gives equal importance to each crop class. This prevents the evaluation from being dominated by overall accuracy alone and helps confirm that the model performs well across all crop categories.

# Why was a single-feature baseline built?
The single-feature baseline helped me understand whether any individual feature had strong predictive power on its own. Rainfall was the best single predictor, but its macro-F1 was much lower than the full-feature model. This showed that crop recommendation depends on a combination of soil and climate variables rather than one feature alone.

# Why did tree-based models perform well?
Tree-based ensemble models like Random Forest and Extra Trees performed well because they can capture non-linear relationships and interactions between variables. Crop suitability is unlikely to depend on one variable in isolation; it depends on combinations of nutrients, pH, rainfall, humidity, and temperature. Ensemble tree models are strong at learning those interactions.

# What would be improved next?
The next improvement would be to make the model location-aware. I would add latitude, longitude, planting season, historical rainfall, temperature trends, soil type, and possibly yield data. I would also integrate external sources such as NASA POWER, Open-Meteo, or FAOSTAT to move from a simple recommendation model to a climate-smart crop suitability system.