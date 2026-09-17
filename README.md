# Employee Promotion Prediction

This project demonstrates an end-to-end machine learning workflow for predicting employee promotions.

## Project Workflow

- Exploratory Data Analysis
- YData Profiling
- Missing Value Handling
- Outlier Detection
- Categorical Encoding
- Ordinal Encoding
- Feature Scaling
- Train-Test Split
- Multicollinearity Check using VIF
- Logistic Regression
- Random Forest
- ROC-AUC Evaluation
- ColumnTransformer
- Machine Learning Pipeline
- Streamlit Web App

## Dataset

The dataset contains employee-related features such as:

- Age
- Gender
- City
- Education Level
- Employment Type
- Experience Years
- Monthly Income
- Work Hours
- Satisfaction Level
- Certification Status
- Performance Score

Target:

- Promoted

## Final Model

Random Forest with automated preprocessing using ColumnTransformer and Pipeline.

Final pipeline performance:

- Accuracy: 91.7%
- ROC-AUC: 0.958
- Promoted Class Recall: 0.83
- Promoted Class F1-score: 0.80

## Files

- `app.py`
- `requirements.txt`
- `promotion_pipeline.pkl`
- `label_encoder.pkl`
- `eda_feature_scaling_column_transformer_dataset.csv`
- `employee_promotion_profile_report.html`
- Jupyter/Colab notebook

## Web App

A Streamlit app was created to allow users to enter employee details and receive a promotion prediction.
