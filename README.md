# Credit Scoring & Default Risk Prediction

Hệ thống Học máy dự đoán **rủi ro nợ xấu (Probability of Default - PD)**, ước tính **Điểm tín dụng (Credit Score 300–850)** và giải thích minh bạch bằng **SHAP**, kèm giao diện thẩm định trực quan bằng **Streamlit + Plotly**.

---

## 1. Giới thiệu Đề tài

Dự án giải quyết 3 bài toán cốt lõi trong thẩm định tín dụng:
1. **Dự đoán rủi ro nợ xấu ($PD$)**: Tiền xử lý dữ liệu (xử lý khuyết, cắt ngoại trị, chuẩn hóa thang đo, mã hóa) và áp dụng **SMOTE bên trong `imblearn.pipeline.Pipeline`** để xử lý mất cân bằng dữ liệu mà **không gây rò rỉ dữ liệu (Data Leakage)**.
2. **So sánh 5 mô hình phân loại**: Logistic Regression, Decision Tree, Random Forest, XGBoost và LightGBM; đánh giá qua **ROC-AUC, Hệ số Gini, F1-Score, Recall** và hiệu chuẩn xác suất sau SMOTE.
3. **Chấm điểm & Giải thích minh bạch**: Quy đổi xác suất nợ xấu thành **Điểm tín dụng (`300–850`)** theo công thức **PDO** (hạng `A → E`) và kết hợp **SHAP** (Global & Local) để giải thích từng hồ sơ vay.

---

## 2. Tập dữ liệu

- **Nguồn**: [Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Đường dẫn lưu**: `data/raw/credit_risk_dataset.csv`
- **Quy mô**: 32,581 bản ghi × 12 thuộc tính (7 biến số, 4 biến phân loại, 1 biến mục tiêu `loan_status`).
- **Tỷ lệ nhãn**: ~78.18% khách hàng tốt (`0`) vs. ~21.82% khách hàng nợ xấu (`1`).

---

## 3. Cấu trúc Thư mục

```text
CreditScoring/
├── configs/
│   └── config.yaml                  # Cấu hình dữ liệu, tiền xử lý, mô hình & scorecard
├── data/
│   ├── raw/                         # Dữ liệu gốc (credit_risk_dataset.csv)
│   └── processed/                   # Tập train.csv & test.csv sạch cơ bản
├── notebooks/
│   ├── 01_eda.ipynb                 # Khám phá dữ liệu (EDA)
│   ├── 01b_preprocessing_check.ipynb# Kiểm tra trực quan trước/sau tiền xử lý & SMOTE
│   ├── 02_model_experiments.ipynb   # Thử nghiệm & so sánh 5 mô hình
│   └── 03_shap_and_scorecard.ipynb  # Phân tích SHAP & kiểm chứng thang điểm A-E
├── src/
│   ├── data/                        # loader.py, features.py, outliers.py, preprocessing.py
│   ├── modeling/                    # trainer.py, calibration.py, evaluator.py
│   ├── scoring/                     # scorecard.py, explainability.py
│   └── utils/                       # config.py, io.py
├── app/
│   ├── main.py                      # Giao diện Streamlit chính
│   ├── pages/                       # 1_single_scoring, 2_batch_scoring, 3_model_comparison, 4_shap_insights
│   └── components/                  # charts.py (Plotly), styles.py
├── models/                          # Lưu pipeline tốt nhất & artifacts (.joblib)
├── reports/
│   ├── project_report.md            # Báo cáo tiến độ & kết quả dự án
│   └── figures/                     # Biểu đồ chất lượng cao phục vụ báo cáo
├── tests/                           # Kiểm thử tự động (pytest)
├── train.py                         # Chạy toàn bộ pipeline huấn luyện end-to-end
├── requirements.txt
└── README.md
```

---

## 4. Hướng dẫn Cài đặt & Chạy Dự án (Windows PowerShell)

### 4.1. Cài đặt môi trường
```powershell
git clone <link-repo-cua-nhom>
cd CreditScoring

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2. Chuẩn bị dữ liệu
Tải `credit_risk_dataset.csv` từ Kaggle và đặt vào:
```text
data/raw/credit_risk_dataset.csv
```

### 4.3. Chạy kiểm thử tự động (Unit Tests)
```powershell
python -m pytest -v
```

### 4.4. Huấn luyện & Đánh giá mô hình *(Bước 4–7)*
```powershell
python train.py
```

### 4.5. Khởi chạy Giao diện Trực quan Streamlit *(Bước 8)*
```powershell
streamlit run app/main.py
```

---

## 5. Nhóm Thực hiện (5 Thành viên)

| STT | Họ và tên | MSSV | Vai trò & Module phụ trách |
|---:|---|---|---|
| 1 | [Thành viên 1] | [...] | Khám phá dữ liệu (EDA), Tiền xử lý & Pipeline chống rò rỉ (`notebooks/01_eda.ipynb`, `src/data/`) |
| 2 | [Thành viên 2] | [...] | Huấn luyện 5 mô hình, SMOTE & Tối ưu siêu tham số (`src/modeling/trainer.py`, `notebooks/02_model_experiments.ipynb`) |
| 3 | [Thành viên 3] | [...] | Đánh giá (`ROC-AUC`, `Gini`, `F1`, `Recall`), Threshold Tuning & Calibration (`src/modeling/evaluator.py`, `calibration.py`) |
| 4 | [Thành viên 4] | [...] | Quy đổi Điểm tín dụng (`300–850`) & Giải thích mô hình SHAP (`src/scoring/`, `notebooks/03_shap_and_scorecard.ipynb`) |
| 5 | [Thành viên 5] | [...] | Giao diện trực quan Streamlit + Plotly, tích hợp hệ thống & Báo cáo (`app/`, `reports/`, `tests/`) |
