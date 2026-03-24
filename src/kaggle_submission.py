import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer

# -----------------------
# Load data
# -----------------------
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

# Save test IDs
test_ids = test["Id"]

# -----------------------
# Log transform target
# -----------------------
train["SalePrice"] = np.log1p(train["SalePrice"])

# -----------------------
# Split features
# -----------------------
X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]

# -----------------------
# Identify feature types
# -----------------------
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

# -----------------------
# Preprocessing
# -----------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=1))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# -----------------------
# Model Pipeline
# -----------------------
model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", Ridge(alpha=10))
])

# -----------------------
# Train on full dataset
# -----------------------
model.fit(X, y)

# -----------------------
# Predict
# -----------------------
preds_log = model.predict(test)
preds = np.expm1(preds_log)

# -----------------------
# Create submission folder (if not exists)
# -----------------------
import os
os.makedirs("submissions", exist_ok=True)

# -----------------------
# Save submission
# -----------------------
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": preds
})

submission.to_csv("submissions/submission.csv", index=False)

print("✅ Submission file created at submissions/submission.csv")