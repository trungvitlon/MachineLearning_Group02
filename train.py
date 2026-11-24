"""Train on CV, then explicitly finalize once. Run from the repository root.

The evaluator belonging to member 1 is not available on this branch yet.
Artifact writing lives here until that module's public interface is agreed.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.data.loader import load_and_split_data
from src.utils.config import PROJECT_ROOT, load_config


MODEL_MODULES = {
    "logistic_regression": "model_lr",
    "decision_tree": "model_dt",
    "random_forest": "model_rf",
    "xgboost": "model_xgb",
    "lightgbm": "model_lgbm",
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default,
                   allow_nan=False) + "\n", encoding="utf-8"
    )


def artifact_paths(config, name):
    reports = PROJECT_ROOT / config["paths"]["reports_dir"]
    return {
        "model": PROJECT_ROOT / config["paths"]["models_dir"] / f"{name}.joblib",
        "cv": reports / "results" / f"{name}_cv.json",
        "cv_table": reports / "results" / f"{name}_cv_results.csv",
        "result": reports / "results" / f"{name}.json",
        "proba": reports / "results" / f"{name}_test_proba.csv",
        "lock": reports / "results" / f"{name}_finalization.lock",
        "figures": PROJECT_ROOT / config["paths"]["figures_dir"],
    }


def experiment_fingerprint(config):
    """Prevent finalizing with a different dataset, config or preprocessing."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    paths = [PROJECT_ROOT / config["data"]["raw_path"]]
    paths += sorted((PROJECT_ROOT / "src" / "data").glob("*.py"))
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fit_candidate(config, name, module, X_train, y_train, n_iter):
    paths = artifact_paths(config, name)
    if paths["result"].exists() or paths["lock"].exists():
        raise ValueError(f"{name}: already finalized/started test evaluation; refusing retraining.")
    if paths["model"].exists() or paths["cv"].exists():
        raise ValueError(f"{name}: candidate already exists. Review it before starting another run.")
    search = module.tune(config, X_train, y_train, n_iter=n_iter)
    best = search.best_estimator_
    train_auc = roc_auc_score(y_train, best.predict_proba(X_train)[:, 1])
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    paths["cv"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, paths["model"])
    versions = {p: importlib.metadata.version(p) for p in (
        "numpy", "pandas", "scikit-learn", "imbalanced-learn", "lightgbm", "joblib"
    )}
    metadata = {
        "model": name,
        "best_params": search.best_params_,
        "cv_auc": float(search.best_score_),
        "cv_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "train_auc": float(train_auc),
        "n_iter": n_iter,
        "train_rows": len(y_train),
        "fingerprint": experiment_fingerprint(config),
        "model_sha256": hashlib.sha256(paths["model"].read_bytes()).hexdigest(),
        "versions": versions,
    }
    pd.DataFrame(search.cv_results_).to_csv(paths["cv_table"], index=False)
    estimator = best.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        feature_names = best.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.DataFrame({
            "feature": feature_names, "importance": estimator.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(paths["cv"].with_name(f"{name}_feature_importance.csv"), index=False)
        paths["figures"].mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        importance.head(15).iloc[::-1].plot.barh(x="feature", y="importance", ax=ax, legend=False)
        ax.set_title(f"{name}: training feature importance (not causal)")
        fig.tight_layout()
        fig.savefig(paths["figures"] / f"{name}_feature_importance.png", dpi=160)
        plt.close(fig)
    _write_json(paths["cv"], metadata)
    print(json.dumps(metadata, indent=2, default=_json_default))
    print("Candidate saved. Review CV results, then use --finalize once.")


def finalize(config, name, X_test, y_test):
    paths = artifact_paths(config, name)
    if paths["result"].exists() or paths["proba"].exists() or paths["lock"].exists():
        raise ValueError(f"{name}: test evaluation already exists/started. Refusing to repeat it.")
    if not paths["model"].exists() or not paths["cv"].exists():
        raise ValueError(f"{name}: train and review the candidate first.")
    metadata = json.loads(paths["cv"].read_text(encoding="utf-8"))
    if metadata["fingerprint"] != experiment_fingerprint(config):
        raise ValueError("Data/config/preprocessing changed since training.")
    if metadata["model_sha256"] != hashlib.sha256(paths["model"].read_bytes()).hexdigest():
        raise ValueError("Saved pipeline does not match the reviewed CV candidate.")
    best = joblib.load(paths["model"])
    # Exclusive persistent marker: a failed partial evaluation must be reviewed,
    # not silently repeated. Never used to tune parameters or thresholds.
    with paths["lock"].open("x", encoding="utf-8") as marker:
        marker.write("Final test evaluation started.\n")
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba))
    result = {
        "model": name,
        "best_params": metadata["best_params"],
        "cv_auc": metadata["cv_auc"],
        "train_auc": metadata["train_auc"],
        "test_auc": auc,
        "gini": 2 * auc - 1,
        "f1": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
        "threshold": 0.5,
    }
    pd.DataFrame({"y_true": np.asarray(y_test), "y_proba": proba}).to_csv(
        paths["proba"], index=False
    )
    paths["figures"].mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=name)
    fig.tight_layout()
    fig.savefig(paths["figures"] / f"{name}_roc.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=[0, 1], display_labels=["Non-default (0)", "Default (1)"],
        ax=ax, colorbar=False,
    )
    ax.set_title(f"{name} | threshold = 0.5")
    fig.tight_layout()
    fig.savefig(paths["figures"] / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    _write_json(paths["result"], result)
    print(json.dumps(result, indent=2, default=_json_default))
    return result


def summarize(config):
    """Read existing final results without loading models or predicting on test."""
    rows = []
    for name in MODEL_MODULES:
        path = artifact_paths(config, name)["result"]
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            rows.append({k: v for k, v in row.items() if k != "best_params"})
    if not rows:
        print("No finalized results available yet.")
        return pd.DataFrame()
    table = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
    print(table.to_string(index=False))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_MODULES, "all"], default="lightgbm")
    parser.add_argument("--n-iter", type=int, default=40)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--finalize", action="store_true", help="Evaluate saved candidate on test ONCE")
    actions.add_argument("--summary", action="store_true", help="Read saved results only")
    args = parser.parse_args(argv)
    config = load_config()
    if args.summary:
        summarize(config)
        return
    if config["project"]["random_state"] != 42 or config["data"]["test_size"] != 0.2:
        parser.error("Group protocol requires random_state=42 and test_size=0.2.")
    if args.n_iter < 1:
        parser.error("--n-iter must be positive")
    names = list(MODEL_MODULES) if args.model == "all" else [args.model]
    modules = {}
    if not args.finalize:
        for name in names:
            module_name = f"src.modeling.{MODEL_MODULES[name]}"
            if importlib.util.find_spec(module_name) is None:
                parser.error(f"Missing {module_name}. Ask its owner to supply the module before running it.")
            modules[name] = importlib.import_module(module_name)
            if not all(callable(getattr(modules[name], f, None)) for f in (
                "get_model", "get_param_space", "tune"
            )):
                parser.error(f"{module_name} does not implement the three group functions.")
    X_train, X_test, y_train, y_test = load_and_split_data(config)
    for name in names:
        if args.finalize:
            finalize(config, name, X_test, y_test)
        else:
            fit_candidate(config, name, modules[name], X_train, y_train, args.n_iter)


if __name__ == "__main__":
    main()
