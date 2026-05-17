import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

BASE_DIR = Path(__file__).parent

# Load data 
xls_path = BASE_DIR / "Car_sales.xls"
if not xls_path.exists():
    raise FileNotFoundError(
        f"File Car_sales.xls tidak ditemukan di: {BASE_DIR}\n"
        "Pastikan Car_sales.xls ada di folder backend/"
    )

df = pd.read_excel(xls_path, engine="openpyxl")
print(f"Data loaded: {len(df)} baris, {len(df.columns)} kolom")

# Prep 
TARGET = "Price_in_thousands"
NUMERIC_FEATURES = [
    "Engine_size", "Horsepower", "Wheelbase",
    "Width", "Length", "Curb_weight",
    "Fuel_capacity", "Fuel_efficiency",
]
CATEGORICAL_FEATURES = ["Vehicle_type"]

df = df.drop_duplicates().dropna(subset=[TARGET])
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

vehicle_types = sorted(df["Vehicle_type"].dropna().unique().tolist())
print(f"Vehicle types found: {vehicle_types}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Pipeline 
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
])
preprocessor = ColumnTransformer([
    ("numeric", num_pipe, NUMERIC_FEATURES),
    ("categorical", cat_pipe, CATEGORICAL_FEATURES),
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])

# Train 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))

print(f"\n=== Hasil Evaluasi ===")
print(f"RMSE : {rmse:.4f} (ribu USD)")
print(f"R2   : {r2:.4f}")

# Save
model_path = BASE_DIR / "car_price_model.joblib"
config_path = BASE_DIR / "feature_config.json"

joblib.dump(model, model_path)
print(f"\nModel disimpan  → {model_path}")

config = {
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "target": TARGET,
    "rmse": rmse,
    "r2_score": r2,
    "vehicle_types": vehicle_types,
}
json.dump(config, open(config_path, "w"), indent=2)
print(f"Config disimpan → {config_path}")

print("\n✅ Selesai! Sekarang jalankan: python main.py")