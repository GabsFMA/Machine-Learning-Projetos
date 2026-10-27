import pandas as pd

from module_olist.dataset import aggregate_items, create_target, save_dataset


def test_create_target_keeps_delivered_orders_with_complete_dates():
    orders = pd.DataFrame(
        {
            "order_id": ["late", "on_time", "canceled", "missing_date"],
            "order_status": ["delivered", "delivered", "canceled", "delivered"],
            "order_approved_at": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2024-01-12", "2024-01-08", None, "2024-01-09"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2024-01-10", "2024-01-10", "2024-01-10", None]
            ),
        }
    )

    result = create_target(orders)

    assert result["order_id"].tolist() == ["late", "on_time"]
    assert result["is_late"].tolist() == [1, 0]


def test_aggregate_items_summarizes_each_order():
    items = pd.DataFrame(
        {
            "order_id": ["order-1", "order-1", "order-2"],
            "order_item_id": [1, 2, 1],
            "seller_id": ["seller-a", "seller-b", "seller-a"],
            "price": [10.0, 20.0, 5.0],
            "freight_value": [2.0, 3.0, 1.0],
        }
    )

    result = aggregate_items(items).set_index("order_id")

    assert result.loc["order-1", "item_count"] == 2
    assert result.loc["order-1", "seller_count"] == 2
    assert result.loc["order-1", "total_price"] == 30.0
    assert result.loc["order-1", "total_freight"] == 5.0
    assert result.index.is_unique


def test_save_dataset_creates_parent_directory_and_round_trips(tmp_path):
    dataset = pd.DataFrame({"order_id": ["order-1"], "is_late": [1]})
    output_path = tmp_path / "interim" / "dataset.csv"

    save_dataset(dataset, output_path)

    pd.testing.assert_frame_equal(pd.read_csv(output_path), dataset)
