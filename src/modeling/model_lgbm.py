"""LightGBM experiments using the group's unchanged preprocessing pipeline."""

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.data.preprocessing import build_full_pipeline


def get_model(config):
    """Return an unfitted, deterministic CPU estimator (no class weighting)."""
    return LGBMClassifier(
        objective="binary",
        random_state=config["project"]["random_state"],
        n_jobs=1,  # Parallelize CV fits instead of nesting thread pools.
        deterministic=True,
        force_col_wise=True,
        importance_type="gain",
        verbosity=-1,
        subsample_freq=1,  # Required for subsample < 1 to have an effect.
    )


def get_param_space():
    """Finite search space; cap leaves at 2**min(max_depth)=16."""
    return {
        "model__n_estimators": [150, 300, 500],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__num_leaves": [7, 15],
        "model__max_depth": [4, 6, 8, -1],
        "model__min_child_samples": [20, 50, 100],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [0.0, 1.0, 5.0],
    }


def tune(config, X_train, y_train, n_iter=40):
    """Tune only on train. Imputation/encoding/SMOTE are fitted in each fold."""
    if not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    seed = config["project"]["random_state"]
    search = RandomizedSearchCV(
        estimator=build_full_pipeline(config, get_model(config)),
        param_distributions=get_param_space(),
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=2,
        pre_dispatch=2,
        refit=True,
        return_train_score=True,
        error_score="raise",
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search
