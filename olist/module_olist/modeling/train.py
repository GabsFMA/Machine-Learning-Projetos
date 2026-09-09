import pandas as pd
from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline, 
    create_xgboost_pipeline, 
    create_lightgbm_pipeline
)

def train_models(
    x_train: pd.DataFrame, 
    y_train: pd.Series
):

    models = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }

    train_models = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        train_models[name] = model

    return train_models 