"""Run: python -m streamlit run app/main.py (from the repository root)."""

import sys
from pathlib import Path

# Streamlit executes this file directly; resolve the repository without a hard-coded path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.lightgbm_ui import CATEGORIES, load_result_rows, prepare_inputs
from src.utils.config import PROJECT_ROOT, load_config


NAMES = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]


@st.cache_resource
def load_pipeline(path, modified_ns):
    # Only load trusted, locally generated artifacts; no model upload endpoint.
    return joblib.load(path)


def predict(pipe, inputs):
    classes = list(pipe.classes_)
    if 1 not in classes:
        raise ValueError("Mô hình không có lớp nợ xấu = 1.")
    values = pipe.predict_proba(inputs)[:, classes.index(1)]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Mô hình trả về xác suất không hợp lệ.")
    return values


def main():
    st.set_page_config(page_title="Dự đoán rủi ro tín dụng", layout="wide")
    st.title("Dự đoán rủi ro tín dụng")
    st.caption("Demo đồ án IT3190 · Ngưỡng phân loại cố định 0,5 · Nhãn vỡ nợ = 1")
    config = load_config()
    model_dir = PROJECT_ROOT / config["paths"]["models_dir"]
    results_dir = PROJECT_ROOT / config["paths"]["reports_dir"] / "results"
    available = [name for name in NAMES if (model_dir / f"{name}.joblib").exists()]
    tabs = st.tabs(["Một hồ sơ", "Hàng loạt CSV", "So sánh mô hình"])
    pipe = None
    if available:
        name = st.sidebar.selectbox("Mô hình đã huấn luyện", available,
                                    index=available.index("lightgbm") if "lightgbm" in available else 0)
        path = model_dir / f"{name}.joblib"
        try:
            pipe = load_pipeline(str(path), path.stat().st_mtime_ns)
        except Exception as exc:
            st.error(f"Không nạp được mô hình. Kiểm tra môi trường thư viện: {exc}")
        if not (results_dir / f"{name}.json").exists():
            st.sidebar.info("Ứng viên đã fit; chưa có báo cáo đánh giá test cuối cùng.")
    else:
        st.info("Chưa có pipeline đã lưu. Chạy `python train.py --model lightgbm` trước.")
    st.sidebar.caption("Hiện dùng xác suất gốc của mô hình; chưa tích hợp calibration, scorecard và SHAP.")
    st.sidebar.caption("SMOTE và clipping theo pipeline chung của nhóm. Kết quả là minh họa học thuật.")

    with tabs[0]:
        with st.form("single_application"):
            left, right = st.columns(2)
            with left:
                age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
                income = st.number_input("Thu nhập hằng năm (đơn vị tiền như dữ liệu)", min_value=1.0, value=60000.0)
                amount = st.number_input("Số tiền vay", min_value=1.0, value=10000.0)
                employment = st.number_input("Số năm làm việc", min_value=0.0, max_value=60.0, value=5.0)
                employment_missing = st.checkbox("Không có thông tin số năm làm việc")
                history = st.number_input("Số năm lịch sử tín dụng", min_value=0, max_value=100, value=5)
            with right:
                rate = st.number_input("Lãi suất (%)", min_value=0.0, max_value=100.0, value=10.0)
                rate_missing = st.checkbox("Không có thông tin lãi suất")
                home = st.selectbox("Tình trạng nhà ở", CATEGORIES["person_home_ownership"])
                intent = st.selectbox("Mục đích vay", CATEGORIES["loan_intent"])
                grade = st.selectbox("Hạng khoản vay đã biết", CATEGORIES["loan_grade"])
                default = st.selectbox("Lịch sử từng vỡ nợ", CATEGORIES["cb_person_default_on_file"])
            st.caption("Tỷ lệ khoản vay/thu nhập được tính từ hai giá trị nhập. Hạng và lãi suất cần có sẵn trong kịch bản sử dụng.")
            submitted = st.form_submit_button("Dự đoán", disabled=pipe is None)
        if submitted:
            row = {
                "person_age": age, "person_income": income,
                "person_emp_length": np.nan if employment_missing else employment,
                "loan_amnt": amount, "loan_int_rate": np.nan if rate_missing else rate,
                "loan_percent_income": round(amount / income, 2),
                "cb_person_cred_hist_length": history, "person_home_ownership": home,
                "loan_intent": intent, "loan_grade": grade, "cb_person_default_on_file": default,
            }
            try:
                probability = predict(pipe, prepare_inputs(pd.DataFrame([row]), config))[0]
                st.metric("Xác suất thuộc lớp vỡ nợ (chưa hiệu chuẩn)", f"{probability:.1%}")
                st.write("Nhãn dự đoán: **1 — vỡ nợ**" if probability >= 0.5 else "Nhãn dự đoán: **0 — không vỡ nợ**")
            except (ValueError, TypeError) as exc:
                st.error(str(exc))

    with tabs[1]:
        columns = config["data"]["numeric_features"] + config["data"]["categorical_features"]
        st.write("Tải CSV có đủ 11 cột đầu vào. Các cột khác được bỏ qua; thứ tự hồ sơ được giữ nguyên.")
        st.code(", ".join(columns))
        uploaded = st.file_uploader("CSV hồ sơ vay", type=["csv"])
        if st.button("Dự đoán hàng loạt", disabled=pipe is None or uploaded is None):
            try:
                frame = pd.read_csv(uploaded)
                if len(frame) > 10000:
                    raise ValueError("Mỗi lần tải tối đa 10.000 hồ sơ.")
                inputs = prepare_inputs(frame, config)
                output = inputs.copy()
                output["y_proba"] = predict(pipe, inputs)
                output["y_pred"] = (output["y_proba"] >= 0.5).astype(int)
                st.dataframe(output, hide_index=True)
                st.download_button("Tải kết quả", output.to_csv(index=False).encode("utf-8-sig"),
                                   "credit_predictions.csv", "text/csv")
            except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(str(exc))

    with tabs[2]:
        table, errors = load_result_rows(results_dir, NAMES)
        for error in errors:
            st.warning(error)
        if table.empty:
            st.info("Chưa có kết quả test đã chốt. Bảng này chỉ đọc JSON, không chạy lại tập test.")
        else:
            table = table.sort_values("cv_auc", ascending=False)
            st.caption("Sắp theo CV AUC; không tự chọn mô hình bằng test AUC.")
            st.dataframe(table, hide_index=True)
            st.plotly_chart(px.bar(table, x="model", y=["cv_auc", "test_auc"], barmode="group"))
            for name in table["model"]:
                with st.expander(name):
                    for suffix in ("roc", "confusion_matrix", "feature_importance"):
                        figure = PROJECT_ROOT / config["paths"]["figures_dir"] / f"{name}_{suffix}.png"
                        if figure.exists():
                            st.image(str(figure))


if __name__ == "__main__":
    main()
