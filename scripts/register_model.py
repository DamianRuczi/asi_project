"""Rejestracja najlepszego modelu w MLflow Model Registry.

Ładuje wytrenowany artefakt (data/06_models/model.pkl), loguje go jako model
MLflow w dedykowanym przebiegu "model-registration" i rejestruje pod nazwą
hotel-cancellation-rf, ustawiając alias `production` na najnowszą wersję.
Dzięki temu wdrożenie (API) ma jednoznaczne źródło prawdy: "model produkcyjny
to ten z aliasem production w rejestrze".

Użycie:
    python scripts/register_model.py
Podgląd:
    mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  zakładka Models
"""
import pickle
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "data" / "06_models" / "model.pkl"
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "hotel-cancellation"
MODEL_NAME = "hotel-cancellation-rf"
ALIAS = "production"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run(run_name="model-registration"):
        info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=MODEL_NAME
        )

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print(f"Zarejestrowano: {MODEL_NAME}, wersja {version}, alias: {ALIAS}")
    print("Podglad: mlflow ui --backend-store-uri sqlite:///mlflow.db -> Models")


if __name__ == "__main__":
    main()
