from pathlib import Path

from loguru import logger

from module_olist.config import (
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from module_olist.dataset import (
    create_dataset,
    load_data,
    save_dataset,
)

from module_olist.features import create_features

def main(
    orders_path: Path = (
        RAW_DATA_DIR / "olist_orders_dataset.csv"
    ),
    items_path: Path = (
        RAW_DATA_DIR / "olist_order_items_dataset.csv"
    ),
    customers_path: Path = (
        RAW_DATA_DIR / "olist_customers_dataset.csv"
    ),
    output_path: Path = (
        INTERIM_DATA_DIR / "orders_dataset_refined.csv"
)
) -> None:
    """
    Executa o pipeline de construção do dataset intermediário.

    Etapas:
        1. Carrega os dados brutos.
        2. Seleciona os pedidos válidos para o problema.
        3. Cria a variável-alvo `is_late`.
        4. Agrega os itens por pedido.
        5. Realiza as junções entre pedidos, itens e clientes.
        6. Salva o resultado em data/interim.
    """

    logger.info(
        "Iniciando pipeline de construção do dataset..."
    )

    # ---------------------------------------------------------
    # 1. Carregamento dos dados brutos
    # ---------------------------------------------------------
    logger.info("Carregando dados brutos...")

    orders, items, customers = load_data(
        orders_path=orders_path,
        items_path=items_path,
        customers_path=customers_path,
    )

    logger.info(
        f"Pedidos carregados: {len(orders):,}"
    )

    logger.info(
        f"Itens carregados: {len(items):,}"
    )

    logger.info(
        f"Clientes carregados: {len(customers):,}"
    )

    # ---------------------------------------------------------
    # 2. Construção do dataset intermediário
    # ---------------------------------------------------------
    logger.info(
        "Construindo dataset intermediário..."
    )

    dataset = create_dataset(
        orders=orders,
        items=items,
        customers=customers,
    )

    # ---------------------------------------------------------
    # 3. Engenharia de features
    # ---------------------------------------------------------

    logger.info(
        "Criando features..."
    )

    dataset = create_features(
        dataset
    )

    # ---------------------------------------------------------
    # 4. Salvamento em data/interim
    # ---------------------------------------------------------
    logger.info(
        "Salvando dataset intermediário..."
    )

    save_dataset(
        dataset=dataset,
        output_path=output_path,
    )

    logger.success(
        "Pipeline concluído com sucesso."
    )


if __name__ == "__main__":
    main()