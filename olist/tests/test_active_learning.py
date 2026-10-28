import numpy as np
import pandas as pd

from module_olist.modeling.active_learning import (
    create_label_spreading_pipeline,
    run_active_learning,
    select_queries,
)


def synthetic_data(size=40):
    X = pd.DataFrame(
        {
            "purchase_hour": np.arange(size) % 24,
            "purchase_weekday": np.arange(size) % 7,
            "purchase_month": np.arange(size) % 12 + 1,
            "promised_days": np.linspace(1, 30, size),
            "item_count": np.ones(size),
            "seller_count": np.ones(size),
            "total_price": np.linspace(10, 500, size),
            "total_freight": np.linspace(1, 50, size),
            "customer_state": ["SP" if i % 2 else "RJ" for i in range(size)],
        }
    )
    y = pd.Series([0] * (size // 2) + [1] * (size - size // 2), name="is_late")
    return X, y


def test_pipeline_fits_with_unlabeled_samples():
    X, y = synthetic_data()
    y_train = np.where(np.arange(len(y)) % 2 == 0, y.to_numpy(), -1)
    model = create_label_spreading_pipeline()
    model.fit(X, y_train)

    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()


def test_select_queries_supports_random_and_uncertainty():
    X, y = synthetic_data()
    y_train = np.where(np.arange(len(y)) % 2 == 0, y.to_numpy(), -1)
    model = create_label_spreading_pipeline()
    model.fit(X, y_train)
    unlabeled = np.flatnonzero(y_train == -1)

    random_queries = select_queries(
        model,
        unlabeled,
        3,
        "random",
        np.random.default_rng(42),
    )
    uncertainty_queries = select_queries(
        model, unlabeled, 3, "uncertainty", np.random.default_rng(42)
    )

    assert random_queries.shape == (3,)
    assert uncertainty_queries.shape == (3,)
    assert len(set(random_queries)) == 3


def test_run_active_learning_records_initial_and_query_round():
    X_pool, y_pool = synthetic_data(40)
    X_test, y_test = synthetic_data(20)

    history = run_active_learning(
        X_pool,
        y_pool,
        X_test,
        y_test,
        strategy="uncertainty",
        n_initial_labeled=8,
        n_queries=4,
        max_iterations=1,
    )

    assert list(history["n_labeled"]) == [8, 12]
    assert history["strategy"].eq("uncertainty").all()
    assert history["pr_auc"].between(0, 1).all()
    assert history["roc_auc"].between(0, 1).all()
