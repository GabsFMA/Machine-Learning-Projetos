from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import shap

from module_olist.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
from module_olist.modeling.interpret import (
    calculate_shap_values,
    create_explainer,
    prepare_data_for_shap,
)
from module_olist.modeling.predict import load_model
from module_olist.modeling.split import FEATURES


def run_explanations(
    data_path: Path = INTERIM_DATA_DIR / "orders_dataset_refined.csv",
    model_path: Path = MODELS_DIR / "best_model.joblib",
    metadata_path: Path = MODELS_DIR / "metadata.json",
    output_dir: Path = FIGURES_DIR,
    sample_size: int = 500,
) -> dict[str, Path]:
    if sample_size <= 0:
        raise ValueError("sample_size deve ser maior que zero.")

    data = pd.read_csv(data_path)
    if data.empty:
        raise ValueError(f"O dataset não contém linhas para explicar: {data_path}")

    X = data[FEATURES].sample(
        n=min(sample_size, len(data)),
        random_state=42,
    )
    model, model_name, _ = load_model(model_path, metadata_path)
    X_transformed = prepare_data_for_shap(model, X)
    explainer = create_explainer(model)
    shap_values = calculate_shap_values(explainer, X_transformed)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary": output_dir / "shap_summary.png",
        "feature_importance": output_dir / "shap_feature_importance.png",
    }

    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.tight_layout()
    plt.savefig(output_paths["summary"], dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        X_transformed,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_paths["feature_importance"], dpi=150, bbox_inches="tight")
    plt.close()

    logger.success(
        f"Explicações SHAP de {model_name} salvas em: "
        f"{output_paths['summary']} e {output_paths['feature_importance']}"
    )
    return output_paths


def main() -> None:
    run_explanations()


if __name__ == "__main__":
    main()
