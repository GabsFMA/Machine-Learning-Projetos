import pandas as pd
from scipy import sparse
import shap


def prepare_data_for_shap(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)

    if sparse.issparse(transformed):
        transformed = transformed.toarray()

    return pd.DataFrame(
        transformed,
        columns=preprocessor.get_feature_names_out(),
        index=X.index,
    )


def create_explainer(pipeline):
    return shap.TreeExplainer(pipeline.named_steps["model"])


def calculate_shap_values(explainer, X_transformed: pd.DataFrame):
    return explainer(X_transformed)
