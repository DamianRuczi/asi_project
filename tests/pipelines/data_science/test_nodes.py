"""Testy węzłów porównania i wyboru modeli."""

import pandas as pd
from sklearn.datasets import make_classification

from asi_project.pipelines.data_science.nodes import (
    compare_models,
    train_best_model,
    tune_best_model,
)


def _comparison_params() -> dict:
    return {
        "cv_folds": 2,
        "selection_metric": "roc_auc",
        "logistic_regression": {"C": 1.0, "max_iter": 100, "solver": "lbfgs"},
        "random_forest": {
            "n_estimators": 5,
            "max_depth": 3,
            "min_samples_split": 2,
        },
        "xgboost": {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
        },
    }


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_compare_models_ocenia_wszystkich_kandydatow(monkeypatch):
    X, y = _training_data()
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes._log_comparison_run",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.mlflow.set_experiment",
        lambda *args, **kwargs: None,
    )

    result = compare_models(X, y, _comparison_params(), random_state=42)

    assert set(result["models"]) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert result["best_model"] in result["models"]
    assert result["selection_metric"] == "roc_auc"


def test_train_best_model_trenuje_model_wybrany_przez_porownanie():
    X, y = _training_data()
    optuna_results = {
        "model_name": "random_forest",
        "best_params": {
            "n_estimators": 5,
            "max_depth": 3,
            "min_samples_split": 2,
        },
    }

    model = train_best_model(X, y, optuna_results, random_state=42)

    assert len(model.predict(X)) == len(y)


def test_tune_best_model_zwraca_najlepsze_parametry(monkeypatch):
    X, y = _training_data()
    comparison = {"best_model": "random_forest"}
    optuna_params = {
        "n_trials": 1,
        "timeout_seconds": 30,
        "cv_folds": 2,
        "selection_metric": "roc_auc",
    }
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.mlflow.set_experiment",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.mlflow.start_run",
        lambda *args, **kwargs: _NullContext(),
    )
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.mlflow.log_param",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.mlflow.log_params",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "asi_project.pipelines.data_science.nodes.mlflow.log_metric",
        lambda *args, **kwargs: None,
    )

    result = tune_best_model(
        X,
        y,
        comparison,
        {
            "model_comparison": _comparison_params(),
            "optuna": optuna_params,
            "random_state": 42,
        },
    )

    assert result["model_name"] == "random_forest"
    assert result["completed_trials"] == 1
    assert result["best_params"]


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None
