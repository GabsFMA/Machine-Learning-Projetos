import pandas as pd 
import shap
import matplotlib.pyplot as plt
from loguru import logger 

from module_olist.config import (
    FIGURES_DIR, 
    INTERIM_DATA_DIR, 
    MODELS_DIR,
)

from module_olist.modeling.predict import (
    load_model,
)

from module_olist.modeling.interpret import (
    prepare_data_for_shap, 
    create_explainer, 
    calculate_shap_values,
)

def main():
    pass
    
if __name__ == "__main__":
    main()    