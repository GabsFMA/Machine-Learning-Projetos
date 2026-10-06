import numpy as np


def evaluate_model(model, x_test, y_test):
    """Avalia o modelo utilizando o conjunto de teste."""

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    return loss, accuracy


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