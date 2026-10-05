from image_classifier.modeling.model import create_model


def train_model(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
):
    """Cria e treina a CNN."""

    model = create_model()

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
    )

    return model, history