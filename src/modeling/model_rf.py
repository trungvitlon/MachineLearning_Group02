import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    roc_curve,
    confusion_matrix,
)
import joblib

from src.utils.config import PROJECT_ROOT, load_config
from src.data.loader import load_and_split_data
from src.data.preprocessing import build_full_pipeline


def get_model(config: dict) -> RandomForestClassifier:
    """Trả về estimator RandomForestClassifier chưa fit."""
    random_state = config["project"]["random_state"]
    return RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
    )


def get_param_space() -> dict:
    """Trả về không gian tìm kiếm siêu tham số với tiền tố 'model__'."""
    return {
        "model__n_estimators": [100, 150, 200, 250],
        "model__max_depth": [8, 10, 12, 15, None],
        "model__min_samples_split": [5, 10, 15, 20],
        "model__min_samples_leaf": [2, 4, 6, 8],
        "model__max_features": ["sqrt", "log2", None],
        "model__criterion": ["gini", "entropy"],
    }


def tune(config: dict, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 40) -> RandomizedSearchCV:
    """
    Tinh chỉnh tham số bằng StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trên TẬP TRAIN, scoring='roc_auc'.
    """
    random_state = config["project"]["random_state"]
    model = get_model(config)
    pipe = build_full_pipeline(config, model)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_space = get_param_space()

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def plot_and_save_roc_curve(y_true, y_proba, auc_score, gini_score, save_path: Path):
    """Vẽ và lưu ROC curve theo quy định nhóm."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2.5,
        label=f"Random Forest (AUC = {auc_score:.4f}, Gini = {gini_score:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.5, label="Baseline (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve - Random Forest", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_and_save_confusion_matrix(y_true, y_pred, save_path: Path):
    """Vẽ và lưu Confusion Matrix theo quy định nhóm (ngưỡng 0.5)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Default (0)", "Default (1)"],
        yticklabels=["Non-Default (0)", "Default (1)"],
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion Matrix - Random Forest (Threshold = 0.5)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_and_save_feature_importance(pipeline, save_path: Path, top_n: int = 15):
    """Trích xuất và vẽ biểu đồ Top đặc trưng quan trọng nhất."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    rf_model = pipeline.named_steps["model"]
    importances = rf_model.feature_importances_

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(5, int(top_n * 0.35))))
    sns.barplot(
        data=fi_df,
        x="importance",
        y="feature",
        palette="viridis",
        hue="feature",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances - Random Forest", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fi_df


def run():
    print("=" * 60)
    print("NGƯỜI 3: HUẤN LUYỆN VÀ TINH CHỈNH RANDOM FOREST (TUÂN THỦ LUẬT NHÓM)")
    print("=" * 60)

    # 1. Đọc config & nạp dữ liệu bằng load_and_split_data duy nhất
    config = load_config()
    X_train, X_test, y_train, y_test = load_and_split_data(config)
    print(f"Dữ liệu: Train = {X_train.shape}, Test = {X_test.shape}")

    # 2. Tinh chỉnh tham số trên TẬP TRAIN (5-Fold StratifiedKFold, n_iter=40, scoring='roc_auc')
    print("Bắt đầu tinh chỉnh siêu tham số trên tập TRAIN (n_iter=40, cv=5)...")
    search = tune(config, X_train, y_train, n_iter=40)
    best_pipe = search.best_estimator_
    cv_auc = float(search.best_score_)
    print(f"CV ROC-AUC tốt nhất: {cv_auc:.4f}")
    print("Tham số tốt nhất:", search.best_params_)

    # 3. Tính AUC trên tập Train để kiểm tra overfit
    train_proba = best_pipe.predict_proba(X_train)[:, 1]
    train_auc = float(roc_auc_score(y_train, train_proba))
    print(f"Train ROC-AUC (kiểm tra overfit): {train_auc:.4f}")

    # 4. Đánh giá DUY NHẤT 1 LẦN trên tập Test sau khi đã chốt tham số
    print("Đánh giá 1 lần duy nhất trên tập TEST...")
    test_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_proba))
    gini = float(2 * test_auc - 1)

    # Dự đoán tại ngưỡng cố định 0.5 theo đúng quy định luật nhóm
    test_pred_05 = (test_proba >= 0.5).astype(int)
    f1 = float(f1_score(y_test, test_pred_05, pos_label=1))
    recall = float(recall_score(y_test, test_pred_05, pos_label=1))

    print("-" * 50)
    print(f"KẾT QUẢ TEST (Ngưỡng 0.5):")
    print(f"  * ROC-AUC : {test_auc:.4f}")
    print(f"  * Gini    : {gini:.4f}")
    print(f"  * F1      : {f1:.4f}")
    print(f"  * Recall  : {recall:.4f}")
    print(f"  * Train AUC: {train_auc:.4f} (Chênh lệch Train-Test: {train_auc - test_auc:.4f})")
    print("-" * 50)

    # 5. Lưu các file nộp theo đúng quy định Rule 5
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "reports" / "results"
    figures_dir = PROJECT_ROOT / "reports" / "figures"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 5.1 models/random_forest.joblib
    model_path = models_dir / "random_forest.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"1. Đã lưu model: {model_path}")

    # 5.2 reports/results/random_forest.json
    results_payload = {
        "model": "random_forest",
        "best_params": search.best_params_,
        "cv_auc": cv_auc,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "gini": gini,
        "f1": f1,
        "recall": recall,
        "threshold": 0.5,
    }
    json_path = results_dir / "random_forest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4, ensure_ascii=False)
    print(f"2. Đã lưu kết quả JSON: {json_path}")

    # 5.3 reports/results/random_forest_test_proba.csv
    test_proba_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_proba": test_proba,
    })
    proba_path = results_dir / "random_forest_test_proba.csv"
    test_proba_df.to_csv(proba_path, index=False)
    print(f"3. Đã lưu test proba CSV: {proba_path}")

    # 5.4 reports/figures/random_forest_*.png
    roc_fig_path = figures_dir / "random_forest_roc_curve.png"
    cm_fig_path = figures_dir / "random_forest_confusion_matrix.png"
    fi_fig_path = figures_dir / "random_forest_feature_importance.png"

    plot_and_save_roc_curve(y_test, test_proba, test_auc, gini, roc_fig_path)
    plot_and_save_confusion_matrix(y_test, test_pred_05, cm_fig_path)
    top_fi = plot_and_save_feature_importance(best_pipe, fi_fig_path, top_n=12)
    print(f"4. Đã lưu hình ảnh biểu đồ vào {figures_dir}")
    print("Top 5 đặc trưng quan trọng nhất:")
    print(top_fi.head(5).to_string(index=False))

    print("=" * 60)
    print("HOÀN THÀNH HUẤN LUYỆN RANDOM FOREST THEO ĐÚNG LUẬT NHÓM!")
    print("=" * 60)


if __name__ == "__main__":
    run()
