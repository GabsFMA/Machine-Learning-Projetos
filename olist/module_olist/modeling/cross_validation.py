from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from module_olist.modeling.pipeline import (
    create_model_pipelines,
)


def summarize_cv(results):
    """
    Resume os resultados da validação cruzada.

    Para cada métrica, calcula:
    - média dos folds;
    - desvio padrão dos folds.
    """

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary = []

    for metric in metrics:
        values = results[f"test_{metric}"]

        summary.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
            }
        )

    return pd.DataFrame(summary)


def find_best_threshold(y_true, y_proba):
    """
    Encontra o threshold que maximiza o F1-score.

    O threshold é testado entre 0.01 e 0.99.
    """

    best_threshold = None

    best_f1 = -1
    best_accuracy = None
    best_precision = None
    best_recall = None

    # Testa diferentes thresholds
    for threshold in np.arange(
        0.01,
        1.00,
        0.01,
    ):
        # Converte probabilidades em classes
        y_pred = (y_proba >= threshold).astype(int)

        # Calcula as métricas
        accuracy = accuracy_score(
            y_true,
            y_pred,
        )
        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        # Verifica se encontrou um F1 melhor
        if f1 > best_f1:
            best_threshold = threshold
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    return {
        "threshold": best_threshold,
        "accuracy": best_accuracy,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
    }


def cross_validate_models(
    X_train,
    y_train,
):
    """
    Seleciona o modelo por PR-AUC médio da validação cruzada e ajusta seu threshold.

    Para cada modelo:
     - Executa validação cruzada estratificada e resume as métricas dos folds.

    Depois da seleção pelo PR-AUC médio:
     - Gera probabilidades Out-of-Fold somente para o modelo selecionado;
     - Escolhe o threshold que maximiza o F1 nessas probabilidades.

    Retorna:
    - nome do melhor modelo;
    - melhor threshold.
    """

    # -------------------------------------------------
    # Pipelines
    # -------------------------------------------------

    pipelines = create_model_pipelines()

    # -------------------------------------------------
    # Estratégia de Cross Validation
    # -------------------------------------------------

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # -------------------------------------------------
    # Métricas
    # -------------------------------------------------

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    # Guarda os resultados de validação cruzada dos modelos
    cv_results = {}

    # -------------------------------------------------
    # Cross Validation
    # -------------------------------------------------

    for name, pipeline in pipelines.items():
        logger.info(f"Executando Cross Validation: {name}")

        # ---------------------------------------------
        # Métricas dos folds
        # ---------------------------------------------

        results = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=kf,
            scoring=scoring,
            return_train_score=True,
        )

        # Resume média e desvio padrão
        summary = summarize_cv(results)

        print("\n")
        print("=" * 60)
        print(f"CROSS VALIDATION - {name}")
        print("=" * 60)

        print(summary)

        cv_results[name] = {
            "results": results,
            "summary": summary,
            "pr_auc": results["test_pr_auc"].mean(),
        }

        logger.info(f"PR-AUC médio CV ({name}): {cv_results[name]['pr_auc']:.3f}")

    # -------------------------------------------------
    # Seleção do modelo independente do ajuste de threshold
    # -------------------------------------------------

    best_model_name = max(
        cv_results,
        key=lambda name: cv_results[name]["pr_auc"],
    )
    logger.success(
        f"Modelo selecionado por PR-AUC médio CV: {best_model_name} "
        f"({cv_results[best_model_name]['pr_auc']:.3f})"
    )

    logger.info(f"Calculando probabilidades OOF para ajustar o threshold: {best_model_name}")
    y_proba_oof = cross_val_predict(
        estimator=pipelines[best_model_name],
        X=X_train,
        y=y_train,
        cv=kf,
        method="predict_proba",
    )[:, 1]
    threshold_results = find_best_threshold(
        y_true=y_train,
        y_proba=y_proba_oof,
    )
    logger.info(
        "Threshold de seleção: {:.2f}; F1 OOF para ajuste: {:.3f}",
        threshold_results["threshold"],
        threshold_results["f1"],
    )

    return best_model_name, threshold_results["threshold"]
