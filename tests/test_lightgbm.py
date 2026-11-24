"""Member 5's tests use synthetic data, never the group's held-out test set."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import train
from app.components.lightgbm_ui import load_result_rows, prepare_inputs
from src.modeling import model_lgbm
from src.utils.config import load_config


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    n = 160
    frame = pd.DataFrame({
        "person_age": rng.integers(22, 65, n),
        "person_income": rng.integers(20000, 100000, n),
        "person_emp_length": rng.integers(0, 15, n).astype(float),
        "loan_amnt": rng.integers(1000, 25000, n),
        "loan_int_rate": rng.uniform(5, 20, n),
        "loan_percent_income": rng.uniform(0.05, 0.5, n),
        "cb_person_cred_hist_length": rng.integers(1, 10, n),
        "person_home_ownership": rng.choice(["RENT", "OWN"], n),
        "loan_intent": rng.choice(["MEDICAL", "EDUCATION"], n),
        "loan_grade": rng.choice(["A", "D"], n),
        "cb_person_default_on_file": rng.choice(["N", "Y"], n),
    })
    frame.loc[0, "person_emp_length"] = np.nan
    y = pd.Series((np.arange(n) % 4 == 0).astype(int))
    return frame, y


@pytest.fixture
def fitted_search(monkeypatch, sample):
    monkeypatch.setattr(model_lgbm, "get_param_space", lambda: {
        "model__n_estimators": [8], "model__num_leaves": [7],
        "model__min_child_samples": [5],
    })
    X, y = sample
    return model_lgbm.tune(load_config(), X, y, n_iter=1)


def test_model_contract_and_search_space():
    config = load_config()
    estimator = model_lgbm.get_model(config)
    assert not hasattr(estimator, "classes_")
    assert estimator.random_state == 42
    assert estimator.class_weight is None
    assert estimator.subsample_freq > 0
    assert all(k.startswith("model__") for k in model_lgbm.get_param_space())


def test_cv_pipeline_and_probability_contract(fitted_search, sample):
    search = fitted_search
    X, y = sample
    assert search.cv.n_splits == 5
    assert search.cv.shuffle and search.cv.random_state == 42
    assert search.scoring == "roc_auc" and search.random_state == 42
    pipe = search.best_estimator_
    assert "balancer" in pipe.named_steps
    assert pipe.named_steps["balancer"].random_state == 42
    assert pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"].n_features_in_ == 9
    proba = pipe.predict_proba(X)[:, 1]
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.isfinite(roc_auc_score(y, proba))


def test_export_and_single_finalization(tmp_path, monkeypatch, fitted_search, sample):
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    config = load_config()
    raw = tmp_path / config["data"]["raw_path"]
    raw.parent.mkdir(parents=True)
    raw.write_text("synthetic-test-data", encoding="utf-8")
    X, y = sample
    module = SimpleNamespace(tune=lambda *a, **kw: fitted_search)
    train.fit_candidate(config, "lightgbm", module, X, y, n_iter=1)
    paths = train.artifact_paths(config, "lightgbm")
    assert not paths["result"].exists() and not paths["proba"].exists()
    with pytest.raises(ValueError, match="candidate already exists"):
        train.fit_candidate(config, "lightgbm", module, X, y, n_iter=1)
    result = train.finalize(config, "lightgbm", X.iloc[:40], y.iloc[:40])
    assert set(result) == {"model", "best_params", "cv_auc", "train_auc", "test_auc", "gini", "f1", "recall", "threshold"}
    assert result["gini"] == pytest.approx(2 * result["test_auc"] - 1)
    output = pd.read_csv(paths["proba"])
    assert output.columns.tolist() == ["y_true", "y_proba"]
    assert output.y_true.tolist() == y.iloc[:40].tolist()
    assert paths["figures"].joinpath("lightgbm_roc.png").exists()
    with pytest.raises(ValueError, match="Refusing to repeat"):
        train.finalize(config, "lightgbm", X, y)
    with pytest.raises(ValueError, match="refusing retraining"):
        train.fit_candidate(config, "lightgbm", module, X, y, n_iter=1)
    assert len(train.summarize(config)) == 1


def test_input_validation_keeps_rows_and_missing_values(sample):
    X, _ = sample
    frame = pd.concat([X.iloc[[0]], X.iloc[[0]]], ignore_index=True)
    frame["person_emp_length"] = 200.0
    clean = prepare_inputs(frame, load_config())
    assert len(clean) == 2 and clean.person_emp_length.isna().all()
    with pytest.raises(ValueError, match="Thiếu cột"):
        prepare_inputs(frame.drop(columns="person_age"), load_config())
    frame["person_income"] = "not-a-number"
    with pytest.raises(ValueError, match="phải là số"):
        prepare_inputs(frame, load_config())


def test_result_reader_reports_corruption(tmp_path):
    (tmp_path / "lightgbm.json").write_text('{"model": "lightgbm"}', encoding="utf-8")
    table, errors = load_result_rows(tmp_path, ["lightgbm", "random_forest"])
    assert table.empty and len(errors) == 1


def test_streamlit_initial_render():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(train.PROJECT_ROOT / "app" / "main.py"))
    app.run(timeout=30)
    assert not app.exception
    assert len(app.tabs) == 3
