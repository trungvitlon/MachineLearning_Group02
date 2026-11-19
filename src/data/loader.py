from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import PROJECT_ROOT

def load_raw_data(config: dict) -> pd.DataFrame:
    """Đọc file dữ liệu gốc từ đường dẫn trong config.yaml."""
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    return pd.read_csv(raw_path)

def clean_basic_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Xóa dòng trùng lặp và xử lý lỗi nhập liệu phi logic (không gây rò rỉ dữ liệu)."""
    df = df.drop_duplicates().copy()
    
    # Loại bỏ tuổi > 100 (lỗi nhập liệu như 123, 144 tuổi)
    if "person_age" in df.columns:
        df = df[df["person_age"] <= 100]
    
    # Gán NaN cho thâm niên > 60 năm hoặc lớn hơn tuổi để Imputer xử lý trong Pipeline
    if "person_emp_length" in df.columns:
        invalid_emp = (df["person_emp_length"] > 60) | (
            df["person_emp_length"] > df["person_age"]
        )
        df.loc[invalid_emp, "person_emp_length"] = np.nan
    
    return df.reset_index(drop=True)

def check_imbalance(y: pd.Series, threshold: float = 1.5) -> dict:
    """Kiểm tra mức độ mất cân bằng nhãn và gợi ý phương pháp xử lý (ý tưởng từ data2)."""
    counts = y.value_counts()
    total = len(y)
    ratio = float(counts.max() / counts.min()) if len(counts) > 1 else 1.0
    
    if ratio < threshold:
        recommendation = "Không cần cân bằng (dữ liệu tương đối cân bằng)"
    elif ratio < 3.0:
        recommendation = "SMOTE hoặc class_weight (mất cân bằng nhẹ)"
    elif ratio < 10.0:
        recommendation = "Khuyên dùng SMOTE trong Pipeline CV (mất cân bằng trung bình)"
    else:
        recommendation = "SMOTE kết hợp Under-sampling hoặc class_weight (mất cân bằng nặng)"
    
    return {
        "counts": counts.to_dict(),
        "percentages": {k: round(v / total * 100, 2) for k, v in counts.items()},
        "imbalance_ratio": round(ratio, 2),
        "is_imbalanced": ratio > threshold,
        "recommendation": recommendation,
    }
    
def load_and_split_data(config: dict, save_splits: bool = False):
    """Đọc dữ liệu, làm sạch lỗi cơ bản và chia Stratified Train/Test."""
    df = load_raw_data(config)
    df = clean_basic_errors(df)
    
    target_col = config["data"]["target_col"]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        stratify=y,
        random_state=config["project"]["random_state"],
    )
    
    if save_splits:
        processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        pd.concat([X_train, y_train], axis=1).to_csv(processed_dir / "train.csv", index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(processed_dir / "test.csv", index=False)
    
    return X_train, X_test, y_train, y_test