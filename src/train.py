import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessor
from pipeline import build_pipeline

def train_model():
    df = pd.read_csv("data/raw/train.csv")
    df["SalePrice"] = np.log1p(df["SalePrice"])

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor)

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test

if __name__ == "__main__":
    train_model()