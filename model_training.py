import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('credit_risk_dataset.csv')

# Handle missing values
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# Feature engineering
df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']

# Define features and target
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Build Random Forest and XGBoost models
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])

# Fit models
print("Training Random Forest...")
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
print(classification_report(y_test, rf_preds))

print("Training XGBoost...")
xgb_pipeline.fit(X_train, y_train)
xgb_preds = xgb_pipeline.predict(X_test)
print(classification_report(y_test, xgb_preds))

# Save predictions
X_test_predicted = X_test.copy()
X_test_predicted['actual_status'] = y_test
X_test_predicted['predicted_status'] = xgb_preds
X_test_predicted['default_probability'] = xgb_pipeline.predict_proba(X_test)[:, 1]
X_test_predicted.to_csv('new_credit_risk_dataset.csv', index=False)
print("Saved predictions to new_credit_risk_dataset.csv")
