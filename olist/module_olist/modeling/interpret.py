import pandas as pd
import shap 
from scipy import sparse

def prepare_data_for_shap(pipeline, X):
    
    #Recupere o preprocessador 
    preprocessor = pipeline.names_steps['preprocessor']
    
    #Aplica as transformações
    X_transformed = preprocessor.transform(X)
    
    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()
        
    #Recuperar os nomes das features     
    feature_names = preprocessor.get_feature_names_out()
    
    #Converte para um DataFrame
    X_transformed = pd.DataFrame(
        X_transformed, 
        columns=feature_names, 
        index=X.index,
    )
    
    return X_transformed


def create_explainer(pipeline):
    
    model = pipeline.named_steps['model']
    
    #Criar um explicador que entenda as previsões desse modelo
    
    explainer = shap.TreeExplainer(model)
    
    return explainer

def calculate_shap_values(explainer, X_transformed):
    shape_values = explainer(X_transformed)
    return shape_values