from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D


def create_model():
    """Cria e compila a CNN utilizada para classificação da CIFAR-10."""

    model = keras.Sequential()

    # Camada de entrada
    model.add(Input(shape=(32, 32, 3)))

    # Bloco 1
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloco 2
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloco 3
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Transição CNN -> camadas densas
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    # Compilação
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model