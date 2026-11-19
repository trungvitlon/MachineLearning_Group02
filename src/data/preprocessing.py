from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.features import CreditFeatureEngineer
from src.data.outliers import OutlierClipper

def _get_scaler(name: str):
    scalers = {
        "robust": RobustScaler(),
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
    }
    if name not in scalers:
        raise ValueError(f"Scaler '{name}' không hỗ trợ. Chọn: {list(scalers.keys())}")
    return scalers[name]

def _get_encoder(name: str):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Encoder '{name}' không hỗ trợ. Chọn: 'onehot' hoặc 'ordinal'.")

def _get_balancer(method: str, random_state: int):
    if method == "smote":
        return SMOTE(random_state=random_state)
    if method == "random_over":
        return RandomOverSampler(random_state=random_state)
    if method == "random_under":
        return RandomUnderSampler(random_state=random_state)
    if method in ("none", None, False):
        return None
    raise ValueError(
        f"Phương pháp cân bằng '{method}' không hỗ trợ. "
        "Chọn: 'smote', 'random_over', 'random_under', 'none'."
    )
    
def build_preprocessor(config: dict) -> ColumnTransformer:
    """Xây dựng ColumnTransformer xử lý biến số và biến phân loại không rò rỉ dữ liệu."""
    num_cols = list(config["data"]["numeric_features"]) + [
        "interest_burden",
        "emp_to_age_ratio",
    ]
    cat_cols = list(config["data"]["categorical_features"])
    
    prep_cfg = config.get("preprocessing", {})
    impute_strategy = prep_cfg.get("num_impute_strategy", "median")
    outlier_method = prep_cfg.get("outlier_method", "iqr")
    iqr_factor = prep_cfg.get("iqr_factor", 1.5)
    scaler_name = prep_cfg.get("scaler", "robust")
    encoder_name = prep_cfg.get("encoder", "onehot")
    
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("outlier", OutlierClipper(method=outlier_method, factor=iqr_factor)),
            ("scaler", _get_scaler(scaler_name)),
        ]
    )
    
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", _get_encoder(encoder_name)),
        ]
    )
    
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    
def build_full_pipeline(
    config: dict,
    model,
    balancing_method: str | None = None,
) -> ImbPipeline:
    """
    Đóng gói trọn bộ:
    CreditFeatureEngineer -> ColumnTransformer -> Balancer (chỉ trên Train fold) -> Model
    """
    prep_cfg = config.get("preprocessing", {})
    random_state = config["project"]["random_state"]
    
    if balancing_method is None:
        if "balancing_method" in prep_cfg:
            balancing_method = prep_cfg["balancing_method"]
        else:
            balancing_method = "smote" if prep_cfg.get("use_smote", True) else "none"
            
    steps = [
        ("features", CreditFeatureEngineer()),
        ("preprocessor", build_preprocessor(config)),
    ]
    
    balancer = _get_balancer(balancing_method, random_state)
    if balancer is not None:
        steps.append(("balancer", balancer))
        
    steps.append(("model", model))
    return ImbPipeline(steps=steps)