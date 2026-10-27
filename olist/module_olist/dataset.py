from pathlib import Path

import kagglehub
from loguru import logger
import pandas as pd

from module_olist.config import RAW_DATA_DIR

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"
REQUIRED_RAW_FILES = (
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
)


def download_data(output_dir: Path = RAW_DATA_DIR) -> None:
    """Download the public Olist dataset if required source files are missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_files = [
        filename for filename in REQUIRED_RAW_FILES if not (output_dir / filename).is_file()
    ]

    if not missing_files:
        logger.info(f"Dados brutos já disponíveis em: {output_dir}")
        return

    logger.info(f"Baixando o dataset Kaggle {KAGGLE_DATASET}...")
    kagglehub.dataset_download(KAGGLE_DATASET, output_dir=str(output_dir))

    missing_files = [
        filename for filename in REQUIRED_RAW_FILES if not (output_dir / filename).is_file()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"O download do dataset não disponibilizou os arquivos necessários em "
            f"{output_dir}: {missing}"
        )

    logger.success(f"Dados brutos disponíveis em: {output_dir}")


def load_data(
    orders_path: Path,
    items_path: Path,
    customers_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados de pedidos, itens e clientes a partir de arquivos CSV.

    Args:
        orders_path: Caminho para o arquivo CSV de pedidos.
        items_path: Caminho para o arquivo CSV de itens.
        customers_path: Caminho para o arquivo CSV de clientes.

    Returns:
        Tupla contendo:
        - DataFrame de pedidos;
        - DataFrame de itens;
        - DataFrame de clientes.
    """

    orders = pd.read_csv(
        orders_path,
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )

    items = pd.read_csv(items_path)

    customers = pd.read_csv(customers_path)

    return orders, items, customers


def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Seleciona pedidos adequados para o problema de previsão de atraso
    e cria a variável-alvo `is_late`.

    A variável assume:
        0 -> pedido entregue no prazo ou antes;
        1 -> pedido entregue depois da data estimada.

    Args:
        orders: DataFrame contendo os pedidos.

    Returns:
        DataFrame contendo somente os pedidos utilizados na análise,
        incluindo a variável-alvo `is_late`.
    """

    # ---------------------------------------------------------
    # Seleção dos pedidos que podem compor o histórico
    # ---------------------------------------------------------
    delivered_orders = orders.loc[
        # Mantém somente pedidos efetivamente entregues.
        orders["order_status"].eq("delivered")
        # A data real de entrega é necessária para determinar
        # se o pedido atrasou.
        & orders["order_delivered_customer_date"].notna()
        # A data estimada é necessária para comparar
        # o prazo previsto com a entrega real.
        & orders["order_estimated_delivery_date"].notna()
        # A aprovação representa o momento no qual será
        # realizada a previsão.
        & orders["order_approved_at"].notna()
    ].copy()

    # ---------------------------------------------------------
    # Criação da variável-alvo
    # ---------------------------------------------------------
    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"]
        > delivered_orders["order_estimated_delivery_date"]
    ).astype("int8")

    # ---------------------------------------------------------
    # Logs para acompanhamento do processamento
    # ---------------------------------------------------------
    logger.info(f"Pedidos originais: {len(orders):,}")

    logger.info(f"Pedidos no recorte histórico: {len(delivered_orders):,}")

    logger.info(
        f"Distribuição da variável-alvo:\n{delivered_orders['is_late'].value_counts(dropna=False)}"
    )

    return delivered_orders


def aggregate_items(items: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega a tabela de itens para alterar sua granularidade de
    uma linha por item para uma linha por pedido.

    Args:
        items: DataFrame contendo os itens dos pedidos.

    Returns:
        DataFrame agregado por `order_id`.
    """

    items_agg = items.groupby(
        "order_id",
        as_index=False,
    ).agg(
        # Quantidade de itens existentes no pedido.
        item_count=("order_item_id", "count"),
        # Quantidade de vendedores distintos envolvidos.
        seller_count=("seller_id", "nunique"),
        # Valor total dos produtos.
        total_price=("price", "sum"),
        # Valor total de frete.
        total_freight=("freight_value", "sum"),
    )

    # Após a agregação, cada pedido deve aparecer apenas uma vez.
    assert items_agg["order_id"].is_unique, "order_id não é único após a agregação dos itens."

    logger.info(f"Pedidos após agregação dos itens: {len(items_agg):,}")

    return items_agg


def create_dataset(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói o dataset intermediário consolidando pedidos,
    informações agregadas dos itens e localização dos clientes.

    Args:
        orders: DataFrame de pedidos.
        items: DataFrame de itens.
        customers: DataFrame de clientes.

    Returns:
        Dataset consolidado com uma linha por pedido.
    """

    # ---------------------------------------------------------
    # Criação da variável-alvo e seleção do histórico
    # ---------------------------------------------------------
    orders = create_target(orders)

    # ---------------------------------------------------------
    # Agregação dos itens para uma linha por pedido
    # ---------------------------------------------------------
    items_agg = aggregate_items(items)

    # ---------------------------------------------------------
    # Junção entre pedidos e informações dos itens
    # ---------------------------------------------------------
    data = orders.merge(
        items_agg,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Junção com informações geográficas dos clientes
    # ---------------------------------------------------------
    data = data.merge(
        customers[
            [
                "customer_id",
                "customer_city",
                "customer_state",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # Mantém a garantia de uma linha por pedido.
    assert data["order_id"].is_unique, "order_id deixou de ser único após as junções."

    logger.info(f"Dataset consolidado: {data.shape[0]:,} linhas x {data.shape[1]} colunas")

    return data


def save_dataset(
    dataset: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Salva o dataset em formato CSV.

    Args:
        dataset: DataFrame que será salvo.
        output_path: Caminho de destino do arquivo.
    """

    # Garante que a pasta de destino exista.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    logger.success(f"Dataset salvo em: {output_path}")


def main() -> None:
    download_data()


if __name__ == "__main__":
    main()
