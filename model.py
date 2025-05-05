# requirements:
# pip install pandas scikit-learn joblib skl2onnx

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

# Load the dataset
df = pd.read_csv("Realistic_Phone_Usage_Dataset__2000_Rows_.csv", parse_dates=["datetime"])

# Feature engineering
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek

# Compute per-minute rates (avoid division by zero)
df["unlock_rate"] = df["unlocks_15min"] / df["recent_15min_usage"].replace(0, 0.1)
df["switch_rate"] = df["app_switches_15min"] / df["recent_15min_usage"].replace(0, 0.1)
df["scroll_rate"] = df["scroll_length"] / df["recent_15min_usage"].replace(0, 0.1)

# Split features/target
X = df.drop(columns=["datetime", "intervene"])
y = df["intervene"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = ["top_app_category"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Model definition: GradientBoosting + calibration
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
calibrated_gb = CalibratedClassifierCV(gb, method="sigmoid", cv=5)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", calibrated_gb)
])

# Train
pipe.fit(X_train, y_train)

# (Optional) Quick evaluation
print("Test accuracy:", pipe.score(X_test, y_test))

# Save pipeline to pickle
joblib.dump(pipe, "phone_usage_model.pkl")
print("Saved trained pipeline to phone_usage_model.pkl")

# Export to ONNX for mobile deployment
#    - We list every numeric feature as FloatTensorType
#    - And the one categorical as StringTensorType
initial_types = [
    (name, FloatTensorType([None, 1])) for name in numeric_features
] + [
    ("top_app_category", StringTensorType([None, 1]))
]

onnx_model = convert_sklearn(
    pipe,
    initial_types=initial_types,
    target_opset=15,
    options={"zipmap": False}   # produce raw probability array
)

with open("phone_usage_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Exported ONNX model to phone_usage_model.onnx")