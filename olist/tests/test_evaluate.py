import numpy as np
import pytest

from module_olist.modeling.evaluate import evaluate_models


class FixedProbabilityModel:
    def predict_proba(self, features):
        return np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.8, 0.2],
            ]
        )


def test_evaluate_models_returns_metrics_for_each_model():
    features = np.zeros((4, 1))
    labels = np.array([0, 1, 1, 0])

    results = evaluate_models(
        {"fixed": FixedProbabilityModel()},
        features,
        labels,
        {"fixed": 0.5},
    )

    assert results["fixed"]["threshold"] == pytest.approx(0.5)
    assert results["fixed"]["precision"] == pytest.approx(1.0)
    assert results["fixed"]["recall"] == pytest.approx(1.0)
    assert results["fixed"]["f1"] == pytest.approx(1.0)
    assert results["fixed"]["roc_auc"] == pytest.approx(1.0)
