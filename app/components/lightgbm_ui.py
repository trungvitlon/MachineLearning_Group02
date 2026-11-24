"""Inference validation and artifact reading; no training or test-set access."""

import json

import numpy as np
import pandas as pd


CATEGORIES = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
    "loan_grade": list("ABCDEFG"),
    "cb_person_default_on_file": ["N", "Y"],
}


def prepare_inputs(frame, config):
    """Keep every input row, including duplicates, and the input row order.

    Apply only the training loader's row-wise employment correction; do not fit
    preprocessing or call the loader's drop_duplicates on inference requests.
    """
    numeric = config["data"]["numeric_features"]
    categorical = config["data"]["categorical_features"]
    required = numeric + categorical
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError("Thiếu cột: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Dữ liệu không có hồ sơ nào.")
    data = frame.loc[:, required].copy()
    for col in numeric:
        original = data[col]
        values = pd.to_numeric(original, errors="coerce")
        if (original.notna() & values.isna()).any():
            raise ValueError(f"{col}: phải là số hoặc để trống nếu được phép.")
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"{col}: không chấp nhận giá trị vô hạn.")
        if col not in ("person_emp_length", "loan_int_rate") and values.isna().any():
            raise ValueError(f"{col}: không được để trống.")
        if (values < 0).any():
            raise ValueError(f"{col}: không được âm.")
        data[col] = values.astype(float)
    if not data["person_age"].between(18, 100).all():
        raise ValueError("Tuổi phải nằm trong khoảng 18–100.")
    if (data["person_income"] <= 0).any() or (data["loan_amnt"] <= 0).any():
        raise ValueError("Thu nhập và số tiền vay phải lớn hơn 0.")
    if (data["cb_person_cred_hist_length"] > data["person_age"]).any():
        raise ValueError("Lịch sử tín dụng không được dài hơn tuổi.")
    for col in categorical:
        data[col] = data[col].astype("string").str.strip().str.upper()
        if not data[col].isin(CATEGORIES[col]).all():
            raise ValueError(f"{col}: giá trị hợp lệ là {', '.join(CATEGORIES[col])}.")
        data[col] = data[col].astype(object)
    invalid_emp = (data["person_emp_length"] > 60) | (
        data["person_emp_length"] > data["person_age"]
    )
    data.loc[invalid_emp, "person_emp_length"] = np.nan
    return data


def load_result_rows(results_dir, names):
    rows, errors = [], []
    required = ["model", "cv_auc", "train_auc", "test_auc", "gini", "f1", "recall", "threshold"]
    for name in names:
        path = results_dir / f"{name}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = {key: payload[key] for key in required}
            if row["model"] != name or any(
                not isinstance(row[k], (int, float)) or not np.isfinite(row[k])
                for k in required if k != "model"
            ):
                raise ValueError("Sai tên mô hình hoặc chỉ số không hợp lệ")
            rows.append(row)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path.name}: {exc}")
    return pd.DataFrame(rows), errors
