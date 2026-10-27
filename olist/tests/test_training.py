import numpy as np
import pandas as pd

from module_olist.modeling import train


def test_run_training_uses_cross_validation_threshold_for_test_evaluation(monkeypatch):
    data = pd.DataFrame(
        {
            "purchase_hour": range(20),
            "purchase_weekday": [index % 7 for index in range(20)],
            "purchase_month": [index % 12 + 1 for index in range(20)],
            "promised_days": np.arange(20, dtype=float),
            "item_count": [1] * 20,
            "seller_count": [1] * 20,
            "total_price": np.arange(20, dtype=float),
            "total_freight": np.arange(20, dtype=float),
            "customer_state": ["SP"] * 20,
            "is_late": [index % 2 for index in range(20)],
        }
    )
    calls = {}
    model = object()

    monkeypatch.setattr(
        train,
        "cross_validate_models",
        lambda X, y: ("LightGBM", 0.37),
    )
    monkeypatch.setattr(train, "train_model", lambda name, X, y: model)
    monkeypatch.setattr(
        train,
        "save_best_model",
        lambda fitted_model, name, threshold: calls.update(saved=(fitted_model, name, threshold)),
    )
    monkeypatch.setattr(
        train,
        "evaluate_models",
        lambda models, X, y, thresholds: (
            calls.update(evaluated=(models, X, y, thresholds)) or {"LightGBM": {"f1": 0.5}}
        ),
    )

    results = train.run_training(data)

    assert results == {"LightGBM": {"f1": 0.5}}
    assert calls["saved"] == (model, "LightGBM", 0.37)
    assert calls["evaluated"][0] == {"LightGBM": model}
    assert calls["evaluated"][3] == {"LightGBM": 0.37}
