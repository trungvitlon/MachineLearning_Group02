import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.data.loader import load_and_split_data
from src.data.preprocessing import build_full_pipeline
from src.modeling.evaluator import evaluate_and_save
from src.utils.config import load_config

MODEL_NAME = "logistic_regression"


def get_model(config: dict) -> LogisticRegression:
    """Trả về estimator Logistic Regression chưa fit."""
    return LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=config["project"]["random_state"],
    )
    
    
def get_param_space() -> dict:
    """Không gian siêu tham số tinh chỉnh (tiền tố 'model__' theo ImbPipeline)."""
    return {
        "model__C": np.logspace(-3, 2, 30),
        "model__l1_ratio": [0.0, 1.0],  # 0.0 = L2 (Ridge), 1.0 = L1 (Lasso)
    }
    
    
def tune(config: dict, X_train, y_train, n_iter: int = 40) -> RandomizedSearchCV:
    """Tinh chỉnh siêu tham số bằng Stratified 5-Fold CV trên tập Train (scoring='roc_auc')."""
    random_state = config["project"]["random_state"]
    cv_folds = config["modeling"].get("cv_folds", 5)
    
    pipe = build_full_pipeline(config, get_model(config))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=get_param_space(),
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def get_top_features(best_pipe, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lấy Top đặc trưng đẩy rủi ro tăng (coef > 0) và giảm rủi ro (coef < 0)."""
    feature_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
    coefs = best_pipe.named_steps["model"].coef_[0]
    df_coef = pd.DataFrame({"feature": feature_names, "coef": coefs})
    
    top_pos = df_coef.sort_values("coef", ascending=False).head(top_n)
    top_neg = df_coef.sort_values("coef", ascending=True).head(top_n)
    return top_pos, top_neg


if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test = load_and_split_data(config)
    
    print(f"Đang tinh chỉnh & huấn luyện {MODEL_NAME} (5-Fold CV, n_iter=40)...")
    search = tune(config, X_train, y_train, n_iter=40)
    
    # Đánh giá 1 LẦN DUY NHẤT trên tập Test và lưu đủ các file đầu ra
    metrics = evaluate_and_save(
        search, X_train, y_train, X_test, y_test, model_name=MODEL_NAME, threshold=0.5
    )
    
    print("\n=== KẾT QUẢ LOGISTIC REGRESSION ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
        
    top_pos, top_neg = get_top_features(search.best_estimator_, top_n=5)
    print("\n--- Top 5 đặc trưng TĂNG rủi ro nợ xấu (coef > 0) ---")
    print(top_pos.to_string(index=False))
    print("\n--- Top 5 đặc trưng GIẢM rủi ro nợ xấu (coef < 0) ---")
    print(top_neg.to_string(index=False))