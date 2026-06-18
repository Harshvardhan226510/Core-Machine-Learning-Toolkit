# Core Machine Learning Toolkit

Welcome to the Core Machine Learning Toolkit. This repository serves as an engineering showcase of fundamental machine learning implementations, robust preprocessing automation, and optimized model pipelines. It tracks production-grade workflows spanning high-dimensional classification, imbalanced financial data engineering, spatial feature generation, and real-time interactive user serving.

---

## 📁 Repository Directory Structure

The repository is explicitly modularized across four core predictive domains:

```text
├── california-housing/
│   └── housing_regressor.ipynb      # Spatial transformation & XGBoost regression pipeline
├── crop-prediction/
│   └── crop_recommendation.ipynb    # IoT sensor optimization & stratified ensemble classifier
├── fraud-detection/
│   └── financial_fraud_detector.ipynb # Imbalanced transaction modeling & Gradio web application
├── hotel-cancellation/
│   └── cancellation_predictor.ipynb  # Complex feature encoding, outlier management & classification
└── README.md                        # Central portfolio documentation matrix
```
## 🌾 1. Smart Crop Recommendation System
```
An industrial-grade agro-analytics pipeline designed to optimize agricultural yields by matching soil chemistry profiles with macro-climatic conditions.

Exploratory Data Analysis: Profiled soil Nitrogen (N), Phosphorus (P), and Potassium (K) distributions using multi-variate box plots alongside soil pH bounds to validate physical IoT sensor utility.

Pipeline Engineering: Wrapped automated feature scaling (StandardScaler) and multi-class tracking configurations inside an encapsulated Scikit-Learn pipeline.

Core Intelligence: Leveraged an XGBClassifier to process 2,200 rows across 22 distinct crop varieties, scoring a 98.86% base accuracy validated by a highly stable 5-Fold cross-validation benchmark (99.09% to 99.77%).
```
## 🏠 2. California Housing Price Regressor
```
An advanced geographic and spatial regression ecosystem engineered to evaluate multi-variate regional real estate valuation targets.

Feature Engineering & Imputation: Addressed missing data points using a localized median SimpleImputer. Utilized a custom vectorized FunctionTransformer to synthesize real-time structural ratio indicators:

Rooms per Household = Total Rooms / Households

Bedrooms per Household = Total Bedrooms / Households

Population per Household = Population / Households

Encoding Matrix: Implemented an explicit OrdinalEncoder to transform string-based coastline proximity values (ocean_proximity) based on real-world categorical indexing boundaries.

Hyperparameter Optimization: Conducted high-throughput hyperparameter optimization utilizing GridSearchCV over an estimators, depth, and learning rate grid, matching a strong operational R² accuracy score of 0.85.
```
## 💳 3.Credit Card Fraud Detection System
```
An enterprise financial intelligence system built to analyze multi-million row transaction logs and flag fraudulent behavior inside highly imbalanced environments.

Class Imbalance Mitigation: Handled extreme target imbalance (over 6.3M clean transactions vs. just 5,742 fraud data records) by calculating the strict imbalance factor matrix (770) and mapping it straight into XGBoost's structural weight constraints (scale_pos_weight=500).

Performance Matrix: Optimized model parameters heavily focusing on defensive recall thresholds, resulting in an exceptional 98% raw fraud recall rate to guarantee comprehensive risk interception.

Gradio Application Serving: Serialized the complete extraction pipeline wrapper (Fraud_prediction_model.pkl) and bound it to an interactive web UI layout using the Gradio framework for live transaction inference tracking.
```
## 🏨 4. Hotel Booking Cancellation Predictor
```
A behavioral analysis model engineered to predict operational room churn and reservation cancellations across 119,390 transaction rows.

Data Cleaning & Outlier Squashing: Stripped systemic leakage factors and uninformative columns. Addressed high-variance financial skewing by utilizing median bounding masks on the Average Daily Rate (adr) column to neutralize erratic anomalies (< 0 or > 900).

Mixed-Type Column Transformer: Constructed a comprehensive data transformation network routing features down three explicit avenues:

Numerical Avenue: Median imputation paired with standard normalization scaling.

One-Hot Avenue: Most-frequent imputation string strategy paired with highly sparse structural encoding matrices.

Ordinal Avenue: Explicit layout maps translating booking room types and refundable token properties.

Ensemble Optimization: Fine-tuned an intensive XGBClassifier loop via GridSearchCV over parameters to achieve an optimal production testing score of 88.02% accuracy.
```
