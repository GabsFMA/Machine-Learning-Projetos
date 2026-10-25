from pathlib import Path

from loguru import logger
import typer

from image_classifier.dataset import CLASS_NAMES, load_cifar10
from image_classifier.modeling.predict import (
    DEFAULT_MODEL_PATH,
    evaluate_model,
    load_model,
    predict,
    predict_proba,
    save_model,
)
from image_classifier.modeling.train import train_model

app = typer.Typer(help="Treina e executa inferência com a CNN CIFAR-10.")


@app.command()
def train(
    epochs: int = 10,
    batch_size: int = 64,
    validation_split: float = 0.2,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """Treina, avalia e salva o classificador CIFAR-10."""

    x_train, y_train, x_test, y_test = load_cifar10()
    model, _ = train_model(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    loss, accuracy = evaluate_model(model, x_test, y_test)
    logger.info("Avaliação no teste — loss: {:.4f}, acurácia: {:.2%}", loss, accuracy)

    saved_path = save_model(model, model_path)
    logger.success("Modelo salvo em: {}", saved_path)


@app.command()
def predict_sample(
    index: int = 0,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """Classifica uma imagem do conjunto de teste da CIFAR-10."""

    if index < 0:
        raise typer.BadParameter("O índice deve ser maior ou igual a zero.")

    _, _, x_test, y_test = load_cifar10()
    if index >= len(x_test):
        raise typer.BadParameter(f"O índice deve ser menor que {len(x_test)}.")

    model = load_model(model_path)
    image = x_test[index : index + 1]
    probabilities = predict_proba(model, image)[0]
    predicted_class = int(predict(model, image)[0])
    actual_class = int(y_test[index])

    logger.info(
        "Imagem {} — classe real: {}, previsão: {}, confiança: {:.2%}",
        index,
        CLASS_NAMES[actual_class],
        CLASS_NAMES[predicted_class],
        float(probabilities[predicted_class]),
    )


if __name__ == "__main__":
    app()
