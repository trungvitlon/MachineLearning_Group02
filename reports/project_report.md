# Báo cáo Dự án: Dự đoán Rủi ro Nợ xấu & Ước tính Điểm Tín dụng (Credit Scoring)

- **Nhóm thực hiện**: [Tên nhóm / Thành viên]
- **Cập nhật lần cuối**: 19/11/2026
- **Trạng thái tổng quan**: Hoàn thành Bước 1 (Khởi tạo kiến trúc & môi trường), chuẩn bị sang Bước 2 (Dữ liệu & EDA)

---

## 1. Mô tả Bài toán

### 1.1. Bối cảnh & Mục tiêu
Trong hoạt động cho vay của ngân hàng và tổ chức tài chính, việc đánh giá chính xác khả năng trả nợ của khách hàng là yếu tố sống còn để kiểm soát tỷ lệ nợ xấu (NPL). Đề tài tập trung xây dựng hệ thống Học máy đầu cuối (end-to-end) nhằm giải quyết đồng thời **3 mục tiêu**:

1. **Dự đoán rủi ro nợ xấu (Probability of Default - PD)**: Phân loại nhị phân ước lượng xác suất khách hàng không có khả năng trả nợ dựa trên dữ liệu lịch sử tài chính, nhân khẩu học và hành vi tín dụng.
2. **Ước tính Điểm tín dụng (Credit Scoring)**: Quy đổi xác suất rủi ro (sau hiệu chuẩn) sang thang điểm trực quan **300 – 850** (theo phương pháp PDO — *Points to Double the Odds*) và phân hạng rủi ro (**A → E**) phục vụ ra quyết định duyệt vay.
3. **Giải thích minh bạch (Explainable AI - SHAP)**: Sử dụng phương pháp **SHAP** ở cả mức toàn cục (Global) và cục bộ cho từng hồ sơ (Local) để chỉ rõ tác động của từng đặc trưng, hỗ trợ nhân viên tín dụng ra quyết định minh bạch và chuẩn xác.

### 1.2. Phương pháp Kỹ thuật Cốt lõi
- **Tiền xử lý chống rò rỉ dữ liệu (Data Leakage Prevention)**: Chia tập `Train / Test` (Stratified Split) ngay từ đầu; đóng gói toàn bộ bước điền giá trị khuyết (`Median` / `"Missing"`), xử lý ngoại trị (`IQRClipper`), chuẩn hóa thang đo (`RobustScaler`), mã hóa biến phân loại và cân bằng dữ liệu (**SMOTE**) bên trong `imblearn.pipeline.Pipeline` để chỉ `fit` trên tập Train trong từng fold Cross-Validation.
- **Các mô hình áp dụng & so sánh**:
  1. Logistic Regression *(Baseline)*
  2. Decision Tree
  3. Random Forest
  4. XGBoost
  5. LightGBM
- **Chỉ số đánh giá**: `ROC-AUC`, `Hệ số Gini` ($\text{Gini} = 2 \times \text{ROC-AUC} - 1$), `F1-Score`, `Recall` (ưu tiên kiểm soát lớp nợ xấu), kết hợp hiệu chuẩn xác suất sau SMOTE.
- **Giao diện trực quan**: Xây dựng Dashboard tương tác bằng **Streamlit + Plotly** gồm 4 chức năng (Thẩm định 1 hồ sơ + What-If, Chấm điểm hàng loạt CSV, So sánh 5 mô hình, Phân tích SHAP).

---

## 2. Tập dữ liệu Sử dụng: Credit Risk Dataset

### 2.1. Tổng quan tập dữ liệu đã chọn
Nhóm lựa chọn bộ **Credit Risk Dataset** (nguồn Kaggle, file dự kiến lưu tại `data/raw/credit_risk_dataset.csv`) làm tập dữ liệu chính cho toàn bộ dự án:

- **Quy mô**: ~32,581 bản ghi, 12 thuộc tính (11 biến đầu vào + 1 biến mục tiêu `loan_status`).
- **Tỷ lệ mất cân bằng nhãn**: ~78.2% khách hàng tốt (`0`) và ~21.8% khách hàng nợ xấu (`1`) → phù hợp để kiểm chứng hiệu quả của **SMOTE** và các chỉ số `ROC-AUC`, `Gini`, `F1-Score`, `Recall`.
- **Đặc điểm chất lượng dữ liệu cần xử lý trong EDA & Tiền xử lý**:
  - Có giá trị khuyết tự nhiên ở `person_emp_length` (~2.7%) và `loan_int_rate` (~9.6%).
  - Có giá trị ngoại trị bất hợp lý cần kiểm tra và xử lý (ví dụ: `person_age > 100`, `person_emp_length > 60`, thu nhập cực lớn).
  - Đầy đủ cả 3 nhóm thông tin theo đề bài: **Nhân khẩu học**, **Tài chính**, và **Hành vi / Lịch sử tín dụng**.

### 2.2. Mô tả chi tiết 12 thuộc tính

| Tên cột | Nhóm đặc trưng | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|---|
| `person_age` | Nhân khẩu học | Số nguyên | Tuổi khách hàng |
| `person_home_ownership` | Nhân khẩu học | Phân loại | Tình trạng nhà ở (`RENT`, `OWN`, `MORTGAGE`, `OTHER`) |
| `person_emp_length` | Nhân khẩu học | Số thực | Số năm kinh nghiệm làm việc |
| `person_income` | Tài chính | Số thực | Thu nhập hàng năm |
| `loan_intent` | Tài chính | Phân loại | Mục đích vay (`EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT`) |
| `loan_grade` | Hành vi / Tín dụng | Phân loại | Hạng khoản vay (`A` đến `G`) |
| `loan_amnt` | Tài chính | Số thực | Số tiền vay |
| `loan_int_rate` | Tài chính | Số thực | Lãi suất khoản vay (%) |
| `loan_percent_income` | Tài chính | Số thực | Tỷ lệ khoản vay trên thu nhập (`loan_amnt / person_income`) |
| `cb_person_default_on_file` | Hành vi / Tín dụng | Phân loại | Lịch sử từng vỡ nợ trước đây (`Y` / `N`) |
| `cb_person_cred_hist_length` | Hành vi / Tín dụng | Số nguyên | Độ dài lịch sử tín dụng (năm) |
| **`loan_status`** *(Target)* | **Biến mục tiêu** | Nhị phân (`0`/`1`) | `0` = Không vỡ nợ (Khách hàng tốt), `1` = Vỡ nợ (Khách hàng xấu) |

### 2.3. Kết quả Khám phá Dữ liệu (EDA)
- **Quy mô & Trùng lặp**: 32,581 dòng × 12 cột; phát hiện **165 dòng trùng lặp hoàn toàn** → sẽ loại bỏ trước khi chia Train/Test.
- **Mất cân bằng nhãn**: Lớp `0` (Tốt) = **25,473 (78.18%)**, Lớp `1` (Nợ xấu) = **7,108 (21.82%)**.
- **Giá trị khuyết**: `person_emp_length` (~895 dòng, 2.75%) và `loan_int_rate` (~3,116 dòng, 9.56%).
- **Ngoại trị & Lỗi nhập liệu**: Phát hiện giá trị phi logic (`person_age > 100`, `person_emp_length > 60`) và phân phối lệch phải mạnh ở `person_income`, `loan_amnt`.
- **Biểu đồ EDA đã lưu**:
  - `reports/figures/eda_target_distribution.png`
  - `reports/figures/eda_boxplots.png`
  - `reports/figures/eda_categorical_default_rate.png`
  - `reports/figures/eda_correlation_heatmap.png`

---

## 3. Tiến độ Hiện tại của Dự án

| Giai đoạn | Nội dung công việc | Trạng thái | Kết quả đầu ra (Deliverables) |
|---|---|---|---|
| **Bước 1** | Thiết kế kiến trúc, tạo cấu trúc thư mục & cài môi trường `.venv` | ✅ Hoàn thành | Cây thư mục `src/`, `app/`, `notebooks/`, `requirements.txt`, `.gitignore` |
| **Bước 2** | Chốt tập dữ liệu chính & thực hiện EDA (phân tích khuyết, ngoại trị, tương quan) | ✅ Hoàn thành | `data/raw/`, `notebooks/01_eda.ipynb` |
| **Bước 3** | Xây dựng Pipeline Tiền xử lý + SMOTE chống rò rỉ dữ liệu | 🔄 Đang thực hiện | `src/data/` (`loader.py`, `features.py`, `preprocessing.py`) |
| **Bước 4** | Huấn luyện, tinh chỉnh & so sánh 5 mô hình (có/không SMOTE) | ⏳ Chưa bắt đầu | `src/modeling/trainer.py`, `notebooks/02_model_experiments.ipynb` |
| **Bước 5** | Đánh giá (`ROC-AUC`, `Gini`, `F1`, `Recall`), Threshold Tuning & Calibration | ⏳ Chưa bắt đầu | `src/modeling/evaluator.py`, `calibration.py`, `reports/figures/` |
| **Bước 6** | Quy đổi xác suất sang Điểm tín dụng (300–850) & phân hạng A–E | ⏳ Chưa bắt đầu | `src/scoring/scorecard.py` |
| **Bước 7** | Giải thích mô hình bằng SHAP (Global & Local + Reason Codes) | ⏳ Chưa bắt đầu | `src/scoring/explainability.py`, `notebooks/03_shap_and_scorecard.ipynb` |
| **Bước 8** | Xây dựng giao diện trực quan Streamlit + Plotly & hoàn thiện báo cáo | ⏳ Chưa bắt đầu | `app/main.py`, `app/pages/`, `models/credit_scoring_bundle.joblib` |

---

## 4. Kế hoạch Tiếp theo (Next Steps)
1. Chốt tập dữ liệu chính thức và đưa file CSV gốc vào thư mục `data/raw/`.
2. Viết cấu hình ban đầu tại `configs/config.yaml` (tên cột mục tiêu, biến số, biến phân loại).
3. Triển khai `notebooks/01_eda.ipynb` để thống kê chi tiết tỷ lệ mất cân bằng, giá trị khuyết, ngoại trị và phát hiện các cột có nguy cơ rò rỉ dữ liệu.