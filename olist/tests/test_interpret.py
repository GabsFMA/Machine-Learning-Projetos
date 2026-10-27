import numpy as np
import pandas as pd
from scipy import sparse
import shap

from module_olist.modeling import interpret


class FakePreprocessor:
    def transform(self, features):
        return sparse.csr_matrix([[1, 0], [0, 1]])

    def get_feature_names_out(self):
        return np.array(["numeric__price", "category__state_SP"])


class FakePipeline:
    def __init__(self):
        self.named_steps = {
            "preprocessor": FakePreprocessor(),
            "model": object(),
        }


def test_prepare_data_for_shap_transforms_sparse_data_and_preserves_index():
    features = pd.DataFrame({"price": [10, 20]}, index=[4, 8])

    result = interpret.prepare_data_for_shap(FakePipeline(), features)

    assert result.columns.tolist() == ["numeric__price", "category__state_SP"]
    assert result.index.tolist() == [4, 8]
    np.testing.assert_array_equal(result.to_numpy(), [[1, 0], [0, 1]])


def test_create_explainer_uses_pipeline_model(monkeypatch):
    pipeline = FakePipeline()
    expected_model = pipeline.named_steps["model"]
    calls = []
    monkeypatch.setattr(shap, "TreeExplainer", lambda model: calls.append(model) or "explainer")

    result = interpret.create_explainer(pipeline)

    assert result == "explainer"
    assert calls == [expected_model]


def test_calculate_shap_values_calls_explainer():
    expected = object()

    class FakeExplainer:
        def __call__(self, features):
            return expected

    result = interpret.calculate_shap_values(
        FakeExplainer(),
        pd.DataFrame({"feature": [1]}),
    )

    assert result is expected
