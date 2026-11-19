import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Xử lý ngoại trị theo chuẩn scikit-learn (chỉ fit ngưỡng trên tập Train).
    Hỗ trợ:
      - 'iqr':           [Q1 - factor*IQR, Q3 + factor*IQR]
      - 'winsorization': [quantile(lower_pct), quantile(upper_pct)]
      - 'zscore':        [mean - z_threshold*std, mean + z_threshold*std]
      - 'none':          Giữ nguyên không cắt
    """
    
    def __init__(
        self,
        method: str = "iqr",
        factor: float = 1.5,
        lower_pct: float = 0.01,
        upper_pct: float = 0.99,
        z_threshold: float = 3.0,
    ):
        self.method = method
        self.factor = factor
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.z_threshold = z_threshold
        
    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        
        if self.method == "iqr":
            q1 = np.nanpercentile(X_arr, 25, axis=0)
            q3 = np.nanpercentile(X_arr, 75, axis=0)
            iqr = q3 - q1
            self.lower_ = q1 - self.factor * iqr
            self.upper_ = q3 + self.factor * iqr
            
        elif self.method == "winsorization":
            self.lower_ = np.nanpercentile(X_arr, self.lower_pct * 100, axis=0)
            self.upper_ = np.nanpercentile(X_arr, self.upper_pct * 100, axis=0)
            
        elif self.method == "zscore":
            mean = np.nanmean(X_arr, axis=0)
            std = np.nanstd(X_arr, axis=0)
            std = np.where(std == 0, 1.0, std)
            self.lower_ = mean - self.z_threshold * std
            self.upper_ = mean + self.z_threshold * std
            
        elif self.method == "none":
            self.lower_ = None
            self.upper_ = None
            
        else:
            raise ValueError(
                f"Phương pháp outlier không hợp lệ: '{self.method}'. "
                "Chọn: 'iqr', 'winsorization', 'zscore', 'none'."
            )
            
        return self
    
    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.method == "none":
            return X_arr
        return np.clip(X_arr, self.lower_, self.upper_)
    
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
    
def detect_outliers_summary(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    """Thống kê nhanh số lượng & tỷ lệ ngoại trị theo IQR phục vụ EDA / báo cáo."""
    rows = []
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        rows.append(
            {
                "column": col,
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outliers_count": n_out,
                "outliers_pct": round(n_out / len(s) * 100, 2) if len(s) else 0.0,
            }
        )
    return pd.DataFrame(rows)