"""Decision Tree
Chạy từ thư mục gốc dự án:  python -m src.modeling.model_dt
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)

from src.utils.config import load_config, PROJECT_ROOT
from src.data.loader import load_and_split_data
from src.data.preprocessing import build_full_pipeline

MODEL_NAME = "decision_tree"


def get_model(config):
    return DecisionTreeClassifier(random_state=config["project"]["random_state"])


def get_param_space():
    return {
        "model__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
        "model__min_samples_leaf": [5, 10, 20, 50, 100],
        "model__min_samples_split": [2, 10, 30, 60],
        "model__criterion": ["gini", "entropy"],
        "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
    }


def tune(config, X_train, y_train, n_iter=40):
    rs = config["project"]["random_state"]
    pipe = build_full_pipeline(config, get_model(config))  # SMOTE nằm trong pipeline
    cv = StratifiedKFold(n_splits=config["modeling"]["cv_folds"], shuffle=True, random_state=rs)
    search = RandomizedSearchCV(pipe, get_param_space(), n_iter=n_iter, cv=cv,
                                scoring=config["modeling"]["primary_metric"],
                                n_jobs=-1, random_state=rs, refit=True)
    search.fit(X_train, y_train)
    return search


def evaluate(best, X_train, y_train, X_test, y_test, threshold=0.5):
    p_tr = best.predict_proba(X_train)[:, 1]
    p_te = best.predict_proba(X_test)[:, 1]
    pred = (p_te >= threshold).astype(int)
    auc = roc_auc_score(y_test, p_te)
    return {
        "train_auc": roc_auc_score(y_train, p_tr),
        "test_auc": auc,
        "gini": 2 * auc - 1,
        "f1": f1_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "threshold": threshold,
    }, p_te


def get_feature_names(best):
    """Tên đặc trưng sau tiền xử lý (num__..., cat__...)."""
    return list(best.named_steps["preprocessor"].get_feature_names_out())


def plot_feature_importance(best, fig_dir, top_n=10):
    names = get_feature_names(best)
    imp = best.named_steps["model"].feature_importances_
    s = pd.Series(imp, index=names).sort_values(ascending=False)
    top = s.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color="#2980b9")
    ax.set_title(f"Top {top_n} đặc trưng quan trọng nhất - Decision Tree")
    ax.set_xlabel("Feature importance")
    fig.savefig(fig_dir / f"{MODEL_NAME}_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return s


def plot_shallow_tree(config, best_params, X_train, y_train, X_test, y_test, fig_dir, depth=3):
    """Cây nông (max_depth=3) cùng pipeline, chỉ để giải thích luật quyết định."""
    p = {k.replace("model__", ""): v for k, v in best_params.items()}
    p.update({"max_depth": depth, "random_state": config["project"]["random_state"]})
    p.pop("ccp_alpha", None)  # cây đã nông, không cần cắt tỉa
    pipe = build_full_pipeline(config, DecisionTreeClassifier(**p))
    pipe.fit(X_train, y_train)
    names = get_feature_names(pipe)
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(pipe.named_steps["model"], feature_names=names,
              class_names=["Tốt", "Nợ xấu"], filled=True, rounded=True,
              fontsize=9, ax=ax, impurity=False, proportion=True)
    ax.set_title(f"Cây quyết định nông (max_depth={depth}) - dùng để giải thích")
    fig.savefig(fig_dir / f"{MODEL_NAME}_tree_depth{depth}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    return auc


def write_notes(result, importance, shallow_auc, cm, path, depth=3):
    """Sinh nhận xét tự động từ số liệu thật (không viết tay số liệu)."""
    tn, fp, fn, tp = cm.ravel()
    gap = result["train_auc"] - result["test_auc"]
    top = importance.head(5)
    top_txt = ", ".join(f"{n} ({v:.3f})" for n, v in top.items())
    bp = {k.replace("model__", ""): v for k, v in result["best_params"].items()}
    lines = [
        "# Nhận xét - Decision Tree",
        "",
        f"- Tham số tốt nhất (CV 5-fold, SMOTE trong pipeline): {bp}.",
        f"- ROC-AUC: CV = {result['cv_auc']:.4f}, train = {result['train_auc']:.4f}, "
        f"test = {result['test_auc']:.4f}; Gini = {result['gini']:.4f}.",
        f"- Chênh lệch AUC train - test = {gap:.4f}: "
        + ("cây ít bị overfit." if gap < 0.03 else "có dấu hiệu overfit, nên giảm max_depth hoặc tăng min_samples_leaf."),
        f"- Ngưỡng 0.5: F1 = {result['f1']:.4f}, Recall = {result['recall']:.4f}, Precision = {result['precision']:.4f}.",
        f"- Trên tập test, mô hình bắt đúng {tp} và bỏ sót {fn} khách nợ xấu; "
        f"nhầm {fp} khách tốt thành nợ xấu (tổng {tn + fp + fn + tp} hồ sơ).",
        f"- 5 đặc trưng quan trọng nhất: {top_txt}.",
        f"- Cây nông (max_depth={depth}) để giải thích luật đạt test AUC = {shallow_auc:.4f}, "
        f"thấp hơn cây tối ưu {result['test_auc'] - shallow_auc:.4f}: đánh đổi giữa dễ giải thích và độ chính xác.",
        "- Lưu ý cho scorecard/calibration: xác suất của cây rời rạc (mỗi lá một giá trị), "
        "nên điểm tín dụng quy đổi sẽ có nhiều mức trùng nhau; cần hiệu chuẩn sau SMOTE.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    config = load_config()
    X_train, X_test, y_train, y_test = load_and_split_data(config)

    search = tune(config, X_train, y_train)
    best = search.best_estimator_
    metrics, p_te = evaluate(best, X_train, y_train, X_test, y_test)

    result = {"model": MODEL_NAME, "best_params": search.best_params_,
              "cv_auc": search.best_score_, **metrics}
    for k, v in result.items():
        print(f"{k}: {v}")

    # lưu kết quả
    res_dir = PROJECT_ROOT / "reports" / "results"
    fig_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    mdl_dir = PROJECT_ROOT / config["paths"]["models_dir"]
    for d in (res_dir, fig_dir, mdl_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(res_dir / f"{MODEL_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    pd.DataFrame({"y_true": y_test.values, "y_proba": p_te}).to_csv(
        res_dir / f"{MODEL_NAME}_test_proba.csv", index=False)
    joblib.dump(best, mdl_dir / f"{MODEL_NAME}.joblib")

    # biểu đồ ROC + confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, p_te, ax=ax, name="Decision Tree")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC - Decision Tree")
    fig.savefig(fig_dir / f"{MODEL_NAME}_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(y_test, (p_te >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Tốt (0)", "Nợ xấu (1)"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix - Decision Tree")
    fig.savefig(fig_dir / f"{MODEL_NAME}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # feature importance + cây nông + nhận xét
    importance = plot_feature_importance(best, fig_dir)
    shallow_auc = plot_shallow_tree(config, search.best_params_, X_train, y_train,
                                    X_test, y_test, fig_dir)
    write_notes(result, importance, shallow_auc, cm, res_dir / f"{MODEL_NAME}_notes.md")
    print("Top 5 feature:", importance.head(5).round(4).to_dict())
    print("Shallow tree (depth=3) test AUC:", round(shallow_auc, 4))


if __name__ == "__main__":
    main()