import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data_path = "data/kidney_disease.csv"
df = pd.read_csv(data_path)

# Clean column names
df.columns = df.columns.str.strip()

# Target column
target = "classification"

# Remove ID column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Clean target values
y = y.astype(str).str.strip().str.lower()

# Encode common target labels
y = y.replace({
    "ckd": 1,
    "notckd": 0,
    "not ckd": 0
})

# Convert unknown target values to numeric and remove invalid rows
y = pd.to_numeric(y, errors="coerce")

valid = y.notna()
X = X.loc[valid]
y = y.loc[valid].astype(int)

# Identify column types
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = "models/random_forest_v2.pkl"
joblib.dump(pipeline, model_path)

print(f"\nModel saved to: {model_path}")