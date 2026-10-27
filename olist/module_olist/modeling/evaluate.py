from loguru import logger
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_models(models, X_test, y_test) -> dict:
    results = {}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]

        best_threshold = None
        best_f1 = -1
        best_precision = None
        best_recall = None

        for threshold in np.arange(0.05, 0.5, 0.01):
            y_pred = (y_proba >= threshold).astype(int)

            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        roc_auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            "threshold": float(best_threshold),
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "roc_auc": roc_auc,
        }

        logger.info(f"Modelo: {name}")
        logger.info(f"Melhor threshold: {best_threshold:.2f}")
        logger.info(f"Precision: {best_precision:.3f}")
        logger.info(f"Recall: {best_recall:.3f}")
        logger.info(f"F1: {best_f1:.3f}")
        logger.info(f"ROC AUC: {roc_auc:.3f}")

    return results
