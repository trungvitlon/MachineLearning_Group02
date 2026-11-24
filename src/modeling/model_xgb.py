"""XGBoost model API and Step 4 training outputs required by the team contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.data.loader import load_and_split_data
from src.data.preprocessing import build_full_pipeline
from src.utils.config import PROJECT_ROOT, load_config


def get_model(config: dict) -> XGBClassifier:
    """Return an unfitted XGBoost estimator using the project's random seed."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config["project"]["random_state"],
        n_jobs=1,
    )


def get_param_space() -> dict[str, list[int | float]]:
    """Search space for RandomizedSearchCV; keys target pipeline's model step."""
    return {
        "model__n_estimators": [200, 300, 400, 500, 600],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__reg_lambda": [1.0, 3.0, 5.0, 10.0],
        "model__gamma": [0.0, 0.1, 0.3, 0.5],
    }


def tune(config: dict, X_train, y_train, n_iter: int = 40) -> RandomizedSearchCV:
    """Tune only on the supplied training split with the mandated 5-fold ROC-AUC."""
    if n_iter < 1:
        raise ValueError("n_iter phải >= 1")

    pipeline = build_full_pipeline(config, get_model(config))
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config["project"]["random_state"],
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=get_param_space(),
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        random_state=config["project"]["random_state"],
        n_jobs=1,
        pre_dispatch="n_jobs",
        return_train_score=False,
        error_score="raise",
    )
    return search.fit(X_train, y_train)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_roc_curve(y_true, y_proba, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.4f})", color="#2457C5", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="XGBoost ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_confusion_matrix(y_true, y_proba, output_path: Path) -> None:
    predictions = (np.asarray(y_proba) >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["Non-default (0)", "Default (1)"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("XGBoost Confusion Matrix (threshold = 0.5)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_shap_outputs(pipeline, X_test, output_dir: Path) -> str:
    """Explain up to 1,000 holdout rows with TreeSHAP; return the top feature."""
    engineered = pipeline.named_steps["features"].transform(X_test)
    transformed = pipeline.named_steps["preprocessor"].transform(engineered)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    sample = transformed[: min(1000, transformed.shape[0])]
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    importance = np.abs(np.asarray(shap_values)).mean(axis=0)
    top_index = int(np.argmax(importance))
    top_feature = str(feature_names[top_index])

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )
    plt.title("XGBoost SHAP Summary")
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return top_feature


def train_and_report(config: dict, n_iter: int = 40) -> dict[str, Any]:
    """Run the required shared split, tune on train, and evaluate test once."""
    X_train, X_test, y_train, y_test = load_and_split_data(config)
    search = tune(config, X_train, y_train, n_iter=n_iter)
    best_pipeline = search.best_estimator_

    # Evaluate train for the requested overfit check, then touch holdout once.
    train_auc = float(roc_auc_score(y_train, best_pipeline.predict_proba(X_train)[:, 1]))
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_proba))
    test_pred = (test_proba >= 0.5).astype(int)
    gini = float(2 * test_auc - 1)
    f1 = float(f1_score(y_test, test_pred, zero_division=0))
    recall = float(recall_score(y_test, test_pred, zero_division=0))

    results_dir = PROJECT_ROOT / "reports" / "results"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    model_dir = PROJECT_ROOT / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": "xgboost",
        "best_params": _json_value(search.best_params_),
        "cv_auc": float(search.best_score_),
        "train_auc": train_auc,
        "test_auc": test_auc,
        "gini": gini,
        "f1": f1,
        "recall": recall,
        "threshold": 0.5,
    }
    with (results_dir / "xgboost.json").open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, indent=2, ensure_ascii=False)
    pd.DataFrame({"y_true": np.asarray(y_test), "y_proba": test_proba}).to_csv(
        results_dir / "xgboost_test_proba.csv", index=False
    )
    joblib.dump(best_pipeline, model_dir / "xgboost.joblib")

    _save_roc_curve(y_test, test_proba, figures_dir / "xgboost_roc_curve.png")
    _save_confusion_matrix(y_test, test_proba, figures_dir / "xgboost_confusion_matrix.png")
    top_feature = _save_shap_outputs(best_pipeline, X_test, figures_dir)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    overfit_gap = train_auc - test_auc
    notes = [
        f"XGBoost được chọn theo ROC-AUC trung bình của Stratified 5-fold CV: {search.best_score_:.4f}.",
        f"ROC-AUC train {train_auc:.4f}; ROC-AUC test {test_auc:.4f}; chênh lệch train-test {overfit_gap:.4f}.",
        "Chênh lệch train-test lớn cần được xem là dấu hiệu overfit và kiểm tra thêm khi so sánh với các mô hình khác.",
        f"Gini trên test là {gini:.4f}; F1 tại threshold 0.5 là {f1:.4f}.",
        f"Recall lớp nợ xấu (1) tại threshold 0.5 là {recall:.4f}.",
        f"Confusion matrix test: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}.",
        f"SHAP mean absolute value xếp {top_feature} là đặc trưng ảnh hưởng mạnh nhất trong mẫu test được giải thích.",
    ]
    (results_dir / "xgboost_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    payload["notes"] = notes
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and report the XGBoost model.")
    parser.add_argument("--iterations", type=int, default=40, help="Random search candidates (default: 40).")
    args = parser.parse_args()
    result = train_and_report(load_config(), n_iter=args.iterations)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
