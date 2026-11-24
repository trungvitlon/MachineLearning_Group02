# LightGBM và tích hợp hệ thống — thành viên 5

## Phạm vi và nguyên tắc

- Nhánh: `Hoa/LightGBM`.
- Giữ nguyên `data/raw/`, `configs/config.yaml` và `src/data/`.
- Dữ liệu chỉ lấy qua `load_and_split_data(config)`: 80/20, stratify, seed 42.
- Pipeline chỉ tạo bằng `build_full_pipeline(config, model)`. SMOTE chạy trong
  từng fold, không kết hợp với trọng số lớp trong thí nghiệm này.
- Tuning: `RandomizedSearchCV`, 40 cấu hình, ROC-AUC, 5 fold stratified,
  shuffle và seed 42; refit trên toàn bộ train sau khi chọn bằng CV AUC.
- Không dùng test để chọn mô hình, siêu tham số, ngưỡng hoặc hiệu chuẩn.

## Mô hình và không gian tìm kiếm

LightGBM kết hợp tuần tự các cây để tối ưu hàm mất mát phân loại nhị phân.
Mỗi cây bổ sung điều chỉnh những sai số còn lại của tổ hợp trước đó. Mô hình
có thể học quan hệ phi tuyến nhưng cần kiểm soát độ phức tạp và overfitting.

| Tham số | Giá trị thử |
|---|---|
| `n_estimators` | 150, 300, 500 |
| `learning_rate` | 0.03, 0.05, 0.1 |
| `num_leaves` | 7, 15 |
| `max_depth` | 4, 6, 8, -1 |
| `min_child_samples` | 20, 50, 100 |
| `subsample` | 0.8, 1.0 |
| `colsample_bytree` | 0.8, 1.0 |
| `reg_alpha` | 0, 0.1, 1 |
| `reg_lambda` | 0, 1, 5 |

`subsample_freq=1` kích hoạt lấy mẫu dòng. `num_leaves` không vượt quá giới hạn
của độ sâu hữu hạn nhỏ nhất. Mỗi estimator dùng một luồng, tìm kiếm chạy hai
fit song song để hạn chế quá tải CPU/RAM. Chế độ CPU deterministic và seed
được bật; cần lưu phiên bản thư viện để tái lập kết quả.

## Chạy từ thư mục gốc

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lightgbm.txt
python -m pytest -v
python train.py --model lightgbm --n-iter 40
```

Lệnh huấn luyện chỉ dùng train/CV để chọn tham số. Kết quả ứng viên gồm:

- `models/lightgbm.joblib`: toàn bộ pipeline tốt nhất đã fit (không đưa lên Git).
- `reports/results/lightgbm_cv.json`: CV AUC, độ lệch chuẩn, train AUC,
  tham số, phiên bản thư viện và fingerprint.
- `reports/results/lightgbm_cv_results.csv`: điểm từng fold của các cấu hình.
- `reports/results/lightgbm_feature_importance.csv`: importance trên tập train.
- `reports/figures/lightgbm_feature_importance.png`.

`requirements-lightgbm.txt` là môi trường tối thiểu cho nhánh này, không sửa
requirements chung. Khi ghép module khác, cài thêm phụ thuộc của module đó hoặc
dùng requirements chung. Trên Windows nên đặt môi trường ảo ở đường dẫn ngắn
nếu cài Jupyter từ requirements chung để tránh giới hạn độ dài đường dẫn.

Sau khi đã chốt ứng viên bằng CV:

```powershell
python train.py --model lightgbm --finalize
python train.py --summary
python -m streamlit run app/main.py
```

`--finalize` nạp pipeline đã lưu, không tuning lại, xuất:

- `reports/results/lightgbm.json` với đúng các khóa nhóm quy định.
- `reports/results/lightgbm_test_proba.csv`: đúng hai cột `y_true,y_proba`,
  giữ nguyên thứ tự do loader trả về.
- `reports/figures/lightgbm_roc.png` và `lightgbm_confusion_matrix.png`.
- Marker `lightgbm_finalization.lock` để chặn đánh giá test lặp.

Không chạy lại tuning sau khi biết kết quả test. Không tự xóa marker để thử
lại. Nếu bị lỗi giữa bước finalize, cần xem lại artifact cùng nhóm trưởng;
marker được giữ để báo rõ đây là một lần đánh giá chưa hoàn tất.
Nếu cần thử nghiệm kỹ thuật trước lần chạy chính thức, dùng dữ liệu tổng hợp
trong tests thay vì test của đồ án. Không ghi đè ứng viên đã có bằng lệnh train.

## Ứng dụng Streamlit

- Nhập một hồ sơ: kiểm tra kiểu dữ liệu và miền giá trị, suy luận bằng pipeline
  đã lưu. Thông tin việc làm/lãi suất thiếu được pipeline điền như khi train.
- CSV hàng loạt: tối đa 10.000 hồ sơ, giữ thứ tự và hồ sơ trùng lặp, xuất
  xác suất và nhãn ở ngưỡng 0.5. Bắt buộc đủ 11 cột trong config.
- So sánh: đọc JSON đã lưu của các mô hình, sắp theo CV AUC; không train hoặc
  dự đoán lại trên test. Đọc các biểu đồ có sẵn.
- Chưa tích hợp calibration, điểm 300–850 hoặc SHAP. Xác suất gốc chưa được
  kiểm chứng là xác suất vỡ nợ đã hiệu chuẩn.
- Chỉ nạp `.joblib` do nhóm tạo từ thư mục `models/`, không tải model lạ lên app.

## Ghép với các thành viên khác

`train.py` nhận `--model logistic_regression|decision_tree|random_forest|xgboost|lightgbm|all`.
Các module phải có đúng ba hàm nhóm đã thống nhất. Nếu thiếu module, lệnh dừng
trước khi huấn luyện, không tự tạo mô hình thay người phụ trách.

Nhánh này chưa có `evaluator.py`, `calibration.py`, `scorecard.py` và
`explainability.py`. Việc xuất kết quả tạm nằm trong `train.py`; khi nhận được
API của evaluator, nối lại tại điểm xuất kết quả, không sửa module của người 1.
Không fit calibration hoặc chọn ngưỡng trên CSV xác suất test. Người 2 cần dùng
validation/OOF trên train. Điểm tín dụng và SHAP được ghép sau khi có API và
artifact tương ứng; không giả lập kết quả trên giao diện.

## Các hạn chế cần ghi trong báo cáo chung

- SMOTE sau One-Hot và IQR clipping của pipeline chung vẫn giữ nguyên để kết
  quả so sánh được. Đây là hạn chế phương pháp cần báo nhóm trưởng, không tự sửa.
- `loan_grade` và `loan_int_rate` cần có sẵn tại thời điểm dự đoán trong kịch bản.
- Importance là đóng góp vào mô hình, không chứng minh quan hệ nhân quả.
- Chênh lệch train AUC và CV/test AUC là dấu hiệu cần phân tích, không tự nó
  chứng minh mô hình tổng quát hóa ngoài dataset.
- Không khẳng định mô hình tốt nhất nhóm khi chưa có đủ kết quả của năm mô hình.

## Kết quả thực nghiệm

Lượt chạy chính thức: 25.928 hồ sơ train, 6.483 hồ sơ test; 40 cấu hình,
200 lượt fit trong CV. Không có cấu hình nào lỗi hoặc trả CV AUC thiếu.
Tham số được chốt theo CV AUC trước lượt đánh giá test cuối cùng.

| Chỉ số | Kết quả |
|---|---:|
| CV AUC trung bình | 0.945937 |
| Độ lệch chuẩn CV AUC | 0.002605 |
| Train AUC (refit) | 0.985628 |
| Test AUC | 0.946495 |
| Gini test | 0.892991 |
| F1 lớp 1, ngưỡng 0.5 | 0.836262 |
| Recall lớp 1, ngưỡng 0.5 | 0.738364 |

Tham số tốt nhất: `n_estimators=500`, `learning_rate=0.1`, `num_leaves=15`,
`max_depth=8`, `min_child_samples=50`, `subsample=0.8`, `colsample_bytree=0.8`,
`reg_alpha=0.1`, `reg_lambda=0`.

### Nhận xét bàn giao (8 ý)

1. LightGBM đạt CV AUC 0.9459 ± 0.0026 trên năm fold, cho thấy kết quả
   tương đối ổn định trong thiết lập chia dữ liệu hiện tại.
2. Test AUC 0.9465 gần CV AUC; chưa thấy suy giảm lớn trên tập test này.
3. Train AUC 0.9856 cao hơn CV AUC khoảng 0.0397, thể hiện dấu hiệu overfit;
   không nên kết luận mô hình hoàn toàn không overfit.
4. Ở ngưỡng 0.5, mô hình phát hiện 1.047/1.418 hồ sơ vỡ nợ, Recall 73.84%.
5. Còn bỏ sót 371 hồ sơ vỡ nợ và cảnh báo nhầm 39 hồ sơ không vỡ nợ;
   đây là hạn chế quan trọng dù AUC cao.
6. Các đặc trưng gain cao nhất là tỷ lệ khoản vay/thu nhập, thu nhập và lãi suất;
   importance được tính trên mô hình đã fit, không chứng minh nhân quả.
7. Xác suất sau pipeline có SMOTE chưa được hiệu chuẩn; phần calibration
   cần thực hiện trên validation/OOF train, không fit trên CSV test đã xuất.
8. Chưa kết luận LightGBM tốt nhất nhóm khi chưa nhận kết quả bốn mô hình còn lại;
   không tinh chỉnh thêm theo kết quả test này.

### Kiểm tra sản phẩm

- 16/16 kiểm thử qua, gồm các kiểm thử tiền xử lý chung có sẵn.
- Kiểm tra Streamlit bằng AppTest: khởi động, nhập hồ sơ, dự đoán, từ chối
  đầu vào không hợp lệ; không huấn luyện mô hình từ giao diện.
- Kiểm tra định dạng JSON, CSV hai cột, thứ tự nhãn, công thức Gini;
  chặn finalize lặp và từ chối model/config đã thay đổi.
- Đã xem lại ảnh ROC, confusion matrix và importance sau khi xuất.
- `.joblib` và dữ liệu processed không được commit; bàn giao model riêng.

## Tài liệu phương pháp

- [LightGBM LGBMClassifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing)
