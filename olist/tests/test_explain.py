import matplotlib.pyplot as plt
import pandas as pd
import pytest

from module_olist import explain
from module_olist.modeling.split import FEATURES


def test_run_explanations_saves_summary_and_feature_importance(
    tmp_path,
    monkeypatch,
):
    dataset_path = tmp_path / "dataset.csv"
    output_dir = tmp_path / "figures"
    rows = 8
    data = pd.DataFrame(
        {
            feature: (["SP"] * rows if feature == "customer_state" else range(rows))
            for feature in FEATURES
        }
    )
    data.to_csv(dataset_path, index=False)
    calls = {}

    monkeypatch.setattr(
        explain,
        "load_model",
        lambda model_path, metadata_path: ("model", "LightGBM", 0.14),
    )

    def prepare_data(model, features):
        calls["sample_size"] = len(features)
        calls["sample_index"] = features.index.tolist()
        return features

    monkeypatch.setattr(explain, "prepare_data_for_shap", prepare_data)
    monkeypatch.setattr(explain, "create_explainer", lambda model: "explainer")
    monkeypatch.setattr(explain, "calculate_shap_values", lambda explainer, X: "values")

    def draw_summary(values, features, **kwargs):
        calls.setdefault("plot_types", []).append(kwargs.get("plot_type", "beeswarm"))
        plt.plot([0, 1], [0, 1])

    monkeypatch.setattr(explain.shap, "summary_plot", draw_summary)

    output_paths = explain.run_explanations(
        data_path=dataset_path,
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "metadata.json",
        output_dir=output_dir,
        sample_size=5,
    )

    assert calls["sample_size"] == 5
    assert calls["plot_types"] == ["beeswarm", "bar"]
    assert output_paths == {
        "summary": output_dir / "shap_summary.png",
        "feature_importance": output_dir / "shap_feature_importance.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in output_paths.values())


@pytest.mark.parametrize("sample_size", [0, -1])
def test_run_explanations_rejects_non_positive_sample_size(tmp_path, sample_size):
    with pytest.raises(ValueError, match="sample_size"):
        explain.run_explanations(
            data_path=tmp_path / "missing.csv",
            output_dir=tmp_path / "figures",
            sample_size=sample_size,
        )


def test_run_explanations_rejects_empty_dataset(tmp_path):
    dataset_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=FEATURES).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="não contém linhas"):
        explain.run_explanations(data_path=dataset_path)
