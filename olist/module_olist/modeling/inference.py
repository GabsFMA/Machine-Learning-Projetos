import pandas as pd

from loguru import logger

from module_olist.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

from module_olist.modeling.predict import (
    load_model,
    predict,
)


def main():

    # -------------------------------------------------
    # 1. Carregamento do dataset preparado
    # -------------------------------------------------

    data = pd.read_csv(
        INTERIM_DATA_DIR
        / "orders_dataset_refined.csv"
    )

    # -------------------------------------------------
    # 2. Seleção das mesmas features usadas no treino
    # -------------------------------------------------

    X = data[
        [
            "promised_days",
            "item_count",
            "seller_count",
            "total_price",
            "total_freight",
            "purchase_month",
            "purchase_weekday",
            "purchase_hour",
            "customer_state",
        ]
    ]

    # -------------------------------------------------
    # 3. Seleção de algumas amostras para demonstração
    # -------------------------------------------------

    X_sample = X.sample(
        n=5,
        random_state=42,
    )

    # -------------------------------------------------
    # 4. Carregamento do modelo e threshold
    # -------------------------------------------------

    model, model_name, threshold = load_model(
        model_path=(
            MODELS_DIR
            / "best_model.joblib"
        ),
        metadata_path=(
            MODELS_DIR
            / "metadata.json"
        ),
    )

    # -------------------------------------------------
    # 5. Inferência
    # -------------------------------------------------

    predictions = predict(
        model=model,
        X=X_sample,
        threshold=threshold,
    )

    # -------------------------------------------------
    # 6. Resultado
    # -------------------------------------------------

    logger.info(
        f"Modelo utilizado: {model_name}"
    )

    logger.info(
        f"Amostras selecionadas para inferência:\n"
        f"{X_sample}\n"
    )

    logger.success(
        f"Predições realizadas:\n"
        f"{predictions}"
    )


if __name__ == "__main__":
    main()