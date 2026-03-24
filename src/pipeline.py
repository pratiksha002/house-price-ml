from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def build_pipeline(preprocessor):
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", Ridge(alpha=10))
    ])
    return pipeline