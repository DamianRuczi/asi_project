"""Potok 'data_science' — porównanie modeli, trening, ewaluacja + MLflow.

Każde uruchomienie `kedro run` tworzy jeden run w MLflow z parametrami,
metrykami i artefaktem najlepszego modelu (system śledzenia eksperymentów).
"""

import logging

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "hotel-cancellation"
# Baza śledzenia eksperymentów (plik w katalogu projektu).
# Podgląd: mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
SCORING = ("accuracy", "f1", "roc_auc")


def split_data(
    model_input: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
):
    """Podział na zbiór treningowy i testowy (ze stratyfikacją po klasie)."""
    X = model_input.drop(columns=[target_column])
    y = model_input[target_column]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_models(
    comparison_params: dict, random_state: int
) -> dict[str, ClassifierMixin]:
    """Buduje kandydatów używanych w porównaniu na tych samych danych."""
    return {
        name: build_model(name, params, random_state)
        for name, params in comparison_params.items()
        if name not in {"cv_folds", "selection_metric"}
    }


def build_model(
    model_name: str, model_params: dict, random_state: int
) -> ClassifierMixin:
    """Buduje pojedynczy model na bazie nazwy i parametrów."""
    if model_name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(**model_params),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            **model_params,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            **model_params,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Nieznany model: {model_name}")


def _log_comparison_run(
    model_name: str,
    model_params: dict,
    cv_folds: int,
    metrics: dict,
) -> None:
    """Zapisuje wynik jednego kandydata jako osobny run MLflow."""
    with mlflow.start_run(run_name=f"comparison-{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_params({f"model__{key}": value for key, value in model_params.items()})
        mlflow.log_metrics(metrics)


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    comparison_params: dict,
    random_state: int,
) -> dict:
    """Porównuje kandydatów przez stratyfikowaną walidację krzyżową."""
    cv_folds = comparison_params["cv_folds"]
    selection_metric = comparison_params["selection_metric"]
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = {}
    for model_name, model in build_models(comparison_params, random_state).items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
        )
        metrics = {
            metric: round(float(np.mean(scores[f"test_{metric}"])), 4)
            for metric in SCORING
        }
        metrics["fit_time_seconds"] = round(float(np.mean(scores["fit_time"])), 2)
        results[model_name] = metrics
        _log_comparison_run(
            model_name,
            comparison_params[model_name],
            cv_folds,
            metrics,
        )
        logger.info("compare_models: %s -> %s", model_name, metrics)

    best_model = max(results, key=lambda name: results[name][selection_metric])
    return {
        "selection_metric": selection_metric,
        "best_model": best_model,
        "cv_folds": cv_folds,
        "models": results,
    }


def _suggest_params(
    trial: optuna.Trial, model_name: str, base_params: dict
) -> dict:
    """Definiuje przestrzeń strojenia dla zwycięskiego typu modelu."""
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 10, 20, 30, 40]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.7]
            ),
        }
    if model_name == "xgboost":
        return {
            **base_params,
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_name == "logistic_regression":
        return {
            **base_params,
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
        }
    raise ValueError(f"Nieznany model: {model_name}")


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    comparison: dict,
    parameters: dict,
) -> dict:
    """Stroi zwycięski model przez CV bez dotykania zbioru testowego."""
    comparison_params = parameters["model_comparison"]
    optuna_params = parameters["optuna"]
    random_state = parameters["random_state"]
    model_name = comparison["best_model"]
    base_params = comparison_params[model_name]
    cv = StratifiedKFold(
        n_splits=optuna_params["cv_folds"],
        shuffle=True,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        candidate_params = _suggest_params(trial, model_name, base_params)
        model = build_model(model_name, candidate_params, random_state)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=optuna_params["selection_metric"],
            n_jobs=1,
        )
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=f"tune-{model_name}",
    )
    study.optimize(
        objective,
        n_trials=optuna_params["n_trials"],
        timeout=optuna_params["timeout_seconds"],
    )

    result = {
        "model_name": model_name,
        "selection_metric": optuna_params["selection_metric"],
        "best_cv_score": round(study.best_value, 4),
        "best_params": study.best_params,
        "completed_trials": len(study.trials),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"optuna-{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("completed_trials", len(study.trials))
        mlflow.log_params({f"best__{key}": value for key, value in study.best_params.items()})
        mlflow.log_metric("best_cv_score", study.best_value)

    logger.info("tune_best_model: %s", result)
    return result


def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    optuna_results: dict,
    random_state: int,
) -> ClassifierMixin:
    """Trenuje na pełnym zbiorze treningowym model dostrojony Optuną."""
    model_name = optuna_results["model_name"]
    model = build_model(model_name, optuna_results["best_params"], random_state)
    model.fit(X_train, y_train)
    logger.info("train_best_model: wytrenowano %s", model_name)
    return model


def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optuna_results: dict,
    parameters: dict,
) -> dict:
    """Końcowa ewaluacja najlepszego modelu na nietkniętym zbiorze testowym."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"final-{optuna_results['model_name']}"):
        mlflow.log_param("model_name", optuna_results["model_name"])
        mlflow.log_params(optuna_results["best_params"])
        mlflow.log_param("test_size", parameters["test_size"])
        mlflow.log_param("random_state", parameters["random_state"])
        mlflow.log_param("top_n_countries", parameters["top_n_countries"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

    logger.info("evaluate_model: %s", metrics)
    return metrics
