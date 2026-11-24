import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.utils.config import load_config
from src.data.loader import load_and_split_data
from src.modeling.model_rf import get_model, get_param_space, tune


def test_get_model():
    config = load_config()
    model = get_model(config)
    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == config["project"]["random_state"]


def test_get_param_space():
    space = get_param_space()
    assert isinstance(space, dict)
    assert len(space) > 0
    for k in space.keys():
        assert k.startswith("model__"), f"Key {k} phải có tiền tố 'model__'"


def test_tune_quick_run():
    config = load_config()
    X_train, X_test, y_train, y_test = load_and_split_data(config)

    # Chạy mẫu nhỏ (small subset & n_iter=2) để test nhanh trong pytest
    X_sub = X_train.head(100)
    y_sub = y_train.head(100)

    search = tune(config, X_sub, y_sub, n_iter=2)
    assert isinstance(search, RandomizedSearchCV)
    assert hasattr(search, "best_estimator_")

    best_pipe = search.best_estimator_
    probs = best_pipe.predict_proba(X_test.head(10))[:, 1]
    assert len(probs) == 10
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
