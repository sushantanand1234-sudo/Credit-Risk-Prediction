import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EDUCATION_MAP = {
    "SSC": 1,
    "OTHERS": 1,
    "12TH": 2,
    "GRADUATE": 3,
    "UNDER GRADUATE": 3,
    "PROFESSIONAL": 3,
    "POST-GRADUATE": 4
}

class EducationOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping=EDUCATION_MAP, column="EDUCATION"):
        self.mapping = mapping
        self.column = column

    def fit(self, X, y=None):
        self.fallback_value_ = pd.Series(
            X[self.column].map(self.mapping)
        ).mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = (
            X[self.column]
            .map(self.mapping)
            .fillna(self.fallback_value_)
            .astype(int)
        )
        return X

pipeline = joblib.load("credit_approval_pipeline.pkl")
print("SUCCESS!")