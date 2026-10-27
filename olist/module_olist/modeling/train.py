import json

import joblib
from loguru import logger
import pandas as pd

from module_olist.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
from module_olist.modeling.cross_validation import cross_validate_models
from module_olist.modeling.evaluate import evaluate_models
from module_olist.modeling.pipeline import create_model_pipelines
from module_olist.modeling.split import split_data


def train_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
):
    models = create_model_pipelines()
    if model_name not in models:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    model = models[model_name]
    logger.info(f"Treinando modelo selecionado: {model_name}")
    model.fit(x_train, y_train)
    return model


def run_training(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    X_train, X_test, y_train, y_test = split_data(data)
    model_name, threshold = cross_validate_models(X_train, y_train)
    model = train_model(model_name, X_train, y_train)
    save_best_model(model, model_name, threshold)
    return evaluate_models(
        {model_name: model},
        X_test,
        y_test,
        {model_name: threshold},
    )


def save_best_model(
    model,
    model_name: str,
    threshold: float,
) -> None:
    """
    Salva o melhor modelo e os metadados
    necessários para a inferência.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Modelo
    # ---------------------------------------------------------

    model_path = MODELS_DIR / "best_model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "model_name": model_name,
        "threshold": float(threshold),
    }

    metadata_path = MODELS_DIR / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    logger.success(f"Modelo salvo em: {model_path}")

    logger.success(f"Metadata salvo em: {metadata_path}")


def main() -> None:
    dataset_path = INTERIM_DATA_DIR / "orders_dataset_refined.csv"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset preparado não encontrado: {dataset_path}. "
            "Execute `uv run python -m module_olist.main` primeiro."
        )

    data = pd.read_csv(dataset_path)
    results = run_training(data)
    logger.success(f"Avaliação no conjunto de teste: {results}")


if __name__ == "__main__":
    main()
