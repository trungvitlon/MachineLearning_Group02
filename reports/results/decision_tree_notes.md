# Nhận xét - Decision Tree

- Tham số tốt nhất (CV 5-fold, SMOTE trong pipeline): {'min_samples_split': 30, 'min_samples_leaf': 20, 'max_depth': None, 'criterion': 'entropy', 'ccp_alpha': 0.0005}.
- ROC-AUC: CV = 0.9151, train = 0.9312, test = 0.9166; Gini = 0.8333.
- Chênh lệch AUC train - test = 0.0145: cây ít bị overfit.
- Ngưỡng 0.5: F1 = 0.7757, Recall = 0.7257, Precision = 0.8332.
- Trên tập test, mô hình bắt đúng 1029 và bỏ sót 389 khách nợ xấu; nhầm 206 khách tốt thành nợ xấu (tổng 6483 hồ sơ).
- 5 đặc trưng quan trọng nhất: num__loan_percent_income (0.234), num__loan_int_rate (0.215), num__person_income (0.129), cat__person_home_ownership_RENT (0.117), cat__loan_grade_C (0.039).
- Cây nông (max_depth=3) để giải thích luật đạt test AUC = 0.8415, thấp hơn cây tối ưu 0.0752: đánh đổi giữa dễ giải thích và độ chính xác.
- Lưu ý cho scorecard/calibration: xác suất của cây rời rạc (mỗi lá một giá trị), nên điểm tín dụng quy đổi sẽ có nhiều mức trùng nhau; cần hiệu chuẩn sau SMOTE.
