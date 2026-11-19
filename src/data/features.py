import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Tạo các đặc trưng tài chính bổ sung an toàn (row-wise, không gây data leakage)."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if {"loan_amnt", "loan_int_rate"}.issubset(df.columns):
            df["interest_burden"] = df["loan_amnt"] * (df["loan_int_rate"] / 100.0)
        if {"person_emp_length", "person_age"}.issubset(df.columns):
            working_years = np.maximum(df["person_age"] - 18, 1)
            df["emp_to_age_ratio"] = df["person_emp_length"] / working_years
        return df