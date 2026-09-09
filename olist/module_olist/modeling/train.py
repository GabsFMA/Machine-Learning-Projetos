import json

import joblib
import pandas as pd

from loguru import logger

from module_olist.config import MODELS_DIR

from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
):

    models = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }

    trained_models = {}

    for name, model in models.items():

        logger.info(
            f"Treinando modelo: {name}"
        )

        model.fit(
            x_train,
            y_train,
        )

        trained_models[name] = model

    return trained_models


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

    model_path = (
        MODELS_DIR
        / "best_model.joblib"
    )

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

    metadata_path = (
        MODELS_DIR
        / "metadata.json"
    )

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

    logger.success(
        f"Modelo salvo em: {model_path}"
    )

    logger.success(
        f"Metadata salvo em: {metadata_path}"
    )