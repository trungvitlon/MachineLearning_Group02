import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.utils.config import load_config
from src.data.loader import check_imbalance, load_and_split_data
from src.data.outliers import OutlierClipper
from src.data.preprocessing import build_full_pipeline


def test_load_and_split_stratified_and_clean():
    config = load_config()
    X_train, X_test, y_train, y_test = load_and_split_data(config)

    # 1. Kiểm tra không còn lỗi nhập liệu phi logic (tuổi > 100)
    assert X_train["person_age"].max() <= 100
    assert X_test["person_age"].max() <= 100

    # 2. Kiểm tra chia Stratified giữ đều tỷ lệ nợ xấu giữa Train và Test
    assert abs(y_train.mean() - y_test.mean()) < 0.01

    # 3. Kiểm tra hàm đánh giá mất cân bằng hoạt động đúng
    imb = check_imbalance(y_train)
    assert imb["is_imbalanced"] is True
    assert imb["imbalance_ratio"] > 1.5


@pytest.mark.parametrize("method", ["iqr", "winsorization", "zscore", "none"])
def test_outlier_clipper_methods(method):
    X_train_dummy = np.array([[10.0], [12.0], [11.0], [13.0], [12.0], [14.0], [15.0], [11.0]])
    X_test_extreme = np.array([[1000.0], [-500.0]])

    clipper = OutlierClipper(method=method)
    clipper.fit(X_train_dummy)
    X_test_clipped = clipper.transform(X_test_extreme)

    assert X_test_clipped.shape == X_test_extreme.shape
    if method != "none":
        # Ngoại trị cực lớn trên Test phải bị cắt về ngưỡng học từ Train
        assert X_test_clipped[0, 0] < 1000.0
        assert X_test_clipped[1, 0] > -500.0
    else:
        assert X_test_clipped[0, 0] == 1000.0


@pytest.mark.parametrize("balancing_method", ["smote", "none"])
def test_full_pipeline_end_to_end(balancing_method):
    config = load_config()
    X_train, X_test, y_train, y_test = load_and_split_data(config)

    pipe = build_full_pipeline(
        config,
        model=LogisticRegression(max_iter=1000, random_state=42),
        balancing_method=balancing_method,
    )
    pipe.fit(X_train, y_train)

    # 1. Dự đoán xác suất trên tập Test sạch, không NaN, nằm trong [0, 1]
    probs = pipe.predict_proba(X_test)[:, 1]
    assert len(probs) == len(X_test)
    assert not np.isnan(probs).any()
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    # 2. Kiểm tra lấy được tên đặc trưng sau tiền xử lý (phục vụ SHAP ở Bước 7)
    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
    assert len(feature_names) > 0