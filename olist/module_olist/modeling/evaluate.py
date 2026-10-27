from loguru import logger
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_models(models, X_test, y_test, thresholds: dict[str, float]) -> dict:
    results = {}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = thresholds[name]
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        results[name] = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        }

        logger.info(f"Modelo: {name}")
        logger.info(f"Threshold: {threshold:.2f}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1: {f1:.3f}")
        logger.info(f"ROC AUC: {roc_auc:.3f}")

    return results
