
# Model Training and Logging with MLflow 🚀

This project demonstrates how to train multiple machine learning models and log their results using MLflow. The models include Random Forest Regressor, Gradient Boosting Regressor, Linear Regression, and K-Nearest Neighbors Regressor.

---

## Table of Contents 📑
- [Introduction](#introduction)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [MLflow Integration](#mlflow-integration)
- [Setup Instructions](#setup-instructions)
- [Conclusion](#conclusion)

---

## Introduction 🎯
This script processes training data, applies mappings to categorical features, and trains machine learning models to predict reactivity. The results, including metrics like MAE, MSE, and R², are logged using MLflow.

---

## Preprocessing 🛠️
1. **Data Preparation**: The `train.csv` dataset is read, and irrelevant columns are dropped.
2. **Feature Engineering**:
   - A `position` feature is extracted from the `id_seqpos` column.
   - New features like `sequence_update`, `predicted_loop_type_update`, and `structure_update` are created by combining sequences.
3. **Mappings**: Categorical variables are mapped to numerical values.
4. **Target and Features**: The `reactivity` column is used as the target variable.

---

## Model Training 📈
### Models Used:
1. **Random Forest Regressor 🌲**
   - Hyperparameters:
     - `n_estimators`: 100
     - `max_depth`: 10
   - Metrics:
     - MAE, MSE, R²

2. **Gradient Boosting Regressor 🔥**
   - Hyperparameters:
     - `learning_rate`: 0.1 and 0.9
     - `max_depth`: 3 and 6

3. **Linear Regression ➗**
   - Simple linear model with intercept fitting.

4. **K-Nearest Neighbors Regressor 🤝**
   - Hyperparameters:
     - `n_neighbors`: 5
     - `weights`: Uniform

---

## MLflow Integration 📊
- Models and their metrics are logged to MLflow.
- Metrics tracked:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
- Each model run is saved as an experiment with unique parameters.

---

## Setup Instructions ⚙️
1. **Install Required Libraries**:
   ```bash
   pip install sklearn mlflow pandas numpy
   ```
2. **Run MLflow Server**:
   ```bash
   mlflow ui
   ```
   The UI is accessible at `http://127.0.0.1:5000`.

3. **Execute the Script**:
   Run the Python script to train models and log results.

---

## Conclusion ✅
This project illustrates the integration of machine learning and MLflow to streamline model evaluation and tracking. 🚀
