from image_classifier.modeling.model import create_model


def train_model(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
):
    """Cria e treina a CNN."""

    if epochs <= 0:
        raise ValueError("epochs deve ser maior que zero.")
    if batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero.")
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split deve estar no intervalo [0, 1).")

    model = create_model()

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    return model, history
