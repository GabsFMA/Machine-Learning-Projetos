from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier  #Bom com dado tabular
from lightgbm import LGBMClassifier 

NUMERIC_FEATURES = [
    "promise_days", 
    "item_count", 
    "seller_count", 
    "total_price", 
    "total_freight"
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purshace_weekday",
    "purchase_hour",
    "costumer_state"
]

def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # Nas colunas numéricas
            ("numeric", "passthrough", NUMERIC_FEATURES), 
            # Nas colunas categoricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Cria o pipeline de pré-processamento  e treinamento do modelo
    """
    preprocessor = create_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=100, #número de árvores na floresta
        learning_rate=0.1, #Taxa de aprendizado
        max_depth=3, 
        random_state=42, 

    )

    return Pipeline(
        steps = [   ("preprocessor", preprocessor), 
                    ("model", model)
        ]
    )

def create_xgboost_pipeline() -> Pipeline:
    """
    Cria o pipeline de pré-processamento  e treinamento do modelo
    """
    preprocessor = create_preprocessor()

    model = XGBClassifier(
        n_estimators=100, #número de árvores na floresta
        learning_rate=0.1, #Taxa de aprendizado
        max_depth=3, 
        random_state=42, 

    )

    return Pipeline(
        steps = [   ("preprocessor", preprocessor), 
                    ("model", model)
        ]
    )

def create_lightgbm_pipeline() -> Pipeline:
    """
    Cria o pipeline de pré-processamento  e treinamento do modelo
    """
    preprocessor = create_preprocessor()

    model = LGBMClassifier(
        n_estimators=100, #número de árvores na floresta
        learning_rate=0.1, #Taxa de aprendizado
        max_depth=3, 
        random_state=42, 

    )

    return Pipeline(
        steps = [   ("preprocessor", preprocessor), 
                    ("model", model)
        ]
    )
    