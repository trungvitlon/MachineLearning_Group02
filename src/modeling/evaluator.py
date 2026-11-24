import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Chạy mượt trên terminal, không bật cửa sổ treo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import PROJECT_ROOT

def _to_json_serializable(obj):
    """Chuyển các kiểu dữ liệu numpy sang kiểu Python thuần để lưu JSON an toàn."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def evaluate_search(
    search,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    threshold: float = 0.5,
):
    """Tính đầy đủ 9 khóa chỉ số theo luật chung evaluator.md (dùng tập Test 1 lần)."""
    best_pipe = search.best_estimator_
    
    # Xác suất lớp nợ xấu (= 1)
    y_train_proba = best_pipe.predict_proba(X_train)[:, 1]
    y_test_proba = best_pipe.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)
    
    train_auc = float(roc_auc_score(y_train, y_train_proba))
    test_auc = float(roc_auc_score(y_test, y_test_proba))
    gini = float(2 * test_auc - 1)
    f1 = float(f1_score(y_test, y_test_pred, pos_label=1))
    recall = float(recall_score(y_test, y_test_pred, pos_label=1))
    
    metrics = {
        "model": model_name,
        "best_params": _to_json_serializable(search.best_params_),
        "cv_auc": round(float(search.best_score_), 4),
        "train_auc": round(train_auc, 4),
        "test_auc": round(test_auc, 4),
        "gini": round(gini, 4),
        "f1": round(f1, 4),
        "recall": round(recall, 4),
        "threshold": float(threshold),
    }
    return metrics, y_test_proba

def save_artifacts(best_pipe, metrics: dict, y_test, y_proba, model_name: str):
    """Lưu model .joblib, kết quả .json và xác suất tập test _test_proba.csv."""
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "reports" / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Lưu pipeline tốt nhất đã fit
    joblib.dump(best_pipe, models_dir / f"{model_name}.joblib")
    
    # 2. Lưu file JSON đúng 9 khóa
    with open(results_dir / f"{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    # 3. Lưu CSV 2 cột y_true, y_proba
    proba_df = pd.DataFrame({
        "y_true": np.asarray(y_test, dtype=int),
        "y_proba": np.asarray(y_proba, dtype=float),
    })
    proba_df.to_csv(results_dir / f"{model_name}_test_proba.csv", index=False)
    
def plot_roc_curve(y_test, y_proba, model_name: str) -> Path:
    """Vẽ và lưu đường cong ROC của 1 mô hình."""
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    gini = 2 * auc - 1
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"{model_name} (AUC = {auc:.4f}, Gini = {gini:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    out_path = figures_dir / f"{model_name}_roc.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_confusion_matrix(y_test, y_proba, model_name: str, threshold: float = 0.5) -> Path:
    """Vẽ và lưu Confusion Matrix tại ngưỡng threshold."""
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Dự đoán: Tốt (0)", "Dự đoán: Xấu (1)"],
        yticklabels=["Thực tế: Tốt (0)", "Thực tế: Xấu (1)"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name} (th={threshold})")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_xlabel("Nhãn dự đoán")
    
    out_path = figures_dir / f"{model_name}_cm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

def evaluate_and_save(
    search,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    threshold: float = 0.5,
) -> dict:
    """Hàm trọn gói cho từng thành viên: chấm điểm, lưu model/JSON/CSV và vẽ ROC + CM."""
    metrics, y_proba = evaluate_search(
        search, X_train, y_train, X_test, y_test, model_name, threshold
    )
    save_artifacts(search.best_estimator_, metrics, y_test, y_proba, model_name)
    plot_roc_curve(y_test, y_proba, model_name)
    plot_confusion_matrix(y_test, y_proba, model_name, threshold)
    return metrics

def compare_models(results_dir: str | Path = "reports/results") -> pd.DataFrame:
    """Tổng hợp toàn bộ file *.json kết quả mô hình trong reports/results/ thành bảng so sánh."""
    rdir = PROJECT_ROOT / results_dir
    required_keys = {"model", "cv_auc", "train_auc", "test_auc", "gini", "f1", "recall", "threshold"}
    rows = []
    for json_file in sorted(rdir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Bỏ qua các file JSON phụ (vd: calibration, scorecard, shap) không phải kết quả model
        if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
            continue
        rows.append({
            "model": data["model"],
            "cv_auc": round(float(data["cv_auc"]), 4),
            "train_auc": round(float(data["train_auc"]), 4),
            "test_auc": round(float(data["test_auc"]), 4),
            "gini": round(float(data["gini"]), 4),
            "f1": round(float(data["f1"]), 4),
            "recall": round(float(data["recall"]), 4),
            "threshold": float(data["threshold"]),
        })
        
    if not rows:
        raise FileNotFoundError(f"Chưa tìm thấy file *.json nào trong {rdir}")
    
    df = pd.DataFrame(rows).sort_values(by="test_auc", ascending=False).reset_index(drop=True)
    out_csv = PROJECT_ROOT / "reports" / "model_comparison.csv"
    df.to_csv(out_csv, index=False)
    return df

def plot_all_roc_curves(results_dir: str | Path = "reports/results") -> Path:
    """Vẽ tất cả đường cong ROC từ *_test_proba.csv lên cùng 1 đồ thị."""
    rdir = PROJECT_ROOT / results_dir
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for csv_file in sorted(rdir.glob("*_test_proba.csv")):
        model_name = csv_file.stem.replace("_test_proba", "")
        df = pd.read_csv(csv_file)
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_proba"])
        auc = roc_auc_score(df["y_true"], df["y_proba"])
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.4f})")
        
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random (0.5)")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("So sánh đường cong ROC - Các mô hình Credit Scoring")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    out_path = figures_dir / "all_models_roc.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    df_cmp = compare_models()
    print("\n=== BẢNG SO SÁNH CÁC MÔ HÌNH ===")
    print(df_cmp.to_string(index=False))
    roc_path = plot_all_roc_curves()
    print(f"\nĐã lưu biểu đồ ROC tổng hợp tại: {roc_path}")