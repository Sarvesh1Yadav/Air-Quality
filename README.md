# Air Quality Index (AQI) Prediction Project

This project aims to predict the Air Quality Index (AQI) range and numerical AQI values based on various air pollutant measurements. It utilizes machine learning models for both classification (predicting AQI range) and regression (predicting AQI value).

## Project Overview

The project involves the following steps:

1.  **Data Loading and Preprocessing:**
    * Loading air quality data from a CSV file.
    * Handling missing values and duplicate entries.
    * Data cleaning and transformation.
    * Feature engineering, including calculating individual pollutant indices (SOi, NOi, Rpi, SPMi).
    * Encoding categorical variables.
2.  **Exploratory Data Analysis (EDA):**
    * Visualizing pollutant distributions and trends.
    * Analyzing pollutant variations by state and year.
    * Exploring relationships between different features.
3.  **Model Training and Evaluation:**
    * Splitting data into training and testing sets.
    * Training classification models (Random Forest, XGBoost) to predict AQI range.
    * Training regression models (Linear Regression, Random Forest, XGBoost, LSTM) to predict numerical AQI values.
    * Hyperparameter tuning using RandomizedSearchCV and GridSearchCV.
    * Evaluating model performance using appropriate metrics (accuracy, classification report, confusion matrix for classification; MAE, MSE, R² for regression).
4.  **Feature Importance:**
    * Determining the most important features using Random Forest.
    * Selecting the top features for model training.

## Files

* `data.csv`: The dataset containing air quality measurements.
* `train_ai (1).ipynb`: Jupyter Notebook containing the code for data preprocessing, EDA, model training, and evaluation.
* `requirements.txt`: List of Python libraries required to run the code.
* `README.md`: This file, providing project documentation.

## Requirements

To run this project, you need the following Python libraries:

```text
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
tensorflow

