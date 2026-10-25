from pathlib import Path

import numpy as np
from tensorflow import keras

from image_classifier.config import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "cifar10_cnn.keras"


def evaluate_model(model, x_test, y_test) -> tuple[float, float]:
    """Avalia o modelo utilizando o conjunto de teste."""

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    return float(loss), float(accuracy)


def save_model(model, model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Salva o modelo treinado no formato nativo do Keras."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model_path


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Carrega um modelo Keras salvo."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    return keras.models.load_model(model_path, compile=False)


def predict(model, images):
    """Gera as classes previstas para as imagens."""

    probabilities = model.predict(
        images,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return predictions


def predict_proba(model, images):
    """Gera as probabilidades previstas para as imagens."""

    return model.predict(
        images,
        verbose=0,
    )
