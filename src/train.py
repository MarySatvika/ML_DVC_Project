import pandas as pd
import joblib
import yaml
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

rf_params = params["random_forest"]
data_params = params["data"]

# Load dataset
data_path = "data/kidney_disease.csv"
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip()

# Target column
target = "classification"

# Remove ID column
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Clean target
y = y.astype(str).str.strip().str.lower()

y = y.replace({
    "ckd": 1,
    "notckd": 0,
    "not ckd": 0
})

y = pd.to_numeric(y, errors="coerce")

# Remove invalid rows
valid = y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)

# Identify column types
numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Random Forest using params.yaml
model = RandomForestClassifier(
    n_estimators=rf_params["n_estimators"],
    max_depth=rf_params["max_depth"],
    random_state=rf_params["random_state"]
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=data_params["test_size"],
    random_state=data_params["random_state"],
    stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Random Forest Experiment")
print("========================")
print("n_estimators:", rf_params["n_estimators"])
print("max_depth:", rf_params["max_depth"])
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = "models/random_forest.pkl"
joblib.dump(pipeline, model_path)

# Save metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nModel saved to:", model_path)
print("Metrics saved to: metrics.json")