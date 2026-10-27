import numpy as np
import pandas as pd
import pytest

from module_olist.modeling import cross_validation


def test_model_selection_uses_cv_pr_auc_before_threshold_tuning(monkeypatch):
    models = {
        "highest-f1-if-tuned": object(),
        "highest-pr-auc": object(),
        "lowest-pr-auc": object(),
    }
    pr_auc_by_model = {
        "highest-f1-if-tuned": 0.7,
        "highest-pr-auc": 0.9,
        "lowest-pr-auc": 0.6,
    }
    threshold_models = []

    def fake_cross_validate(estimator, **kwargs):
        name = next(key for key, model in models.items() if model is estimator)
        return {
            f"test_{metric}": np.array([pr_auc_by_model[name], pr_auc_by_model[name]])
            if metric == "pr_auc"
            else np.array([0.5, 0.5])
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
        }

    def fake_cross_val_predict(estimator, **kwargs):
        threshold_models.append(estimator)
        return np.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])

    monkeypatch.setattr(cross_validation, "create_model_pipelines", lambda: models)
    monkeypatch.setattr(cross_validation, "cross_validate", fake_cross_validate)
    monkeypatch.setattr(cross_validation, "cross_val_predict", fake_cross_val_predict)

    selected_name, threshold = cross_validation.cross_validate_models(
        pd.DataFrame({"feature": [0, 1, 2, 3]}),
        pd.Series([0, 1, 1, 0]),
    )

    assert selected_name == "highest-pr-auc"
    assert threshold == pytest.approx(0.21)
    assert threshold_models == [models["highest-pr-auc"]]
