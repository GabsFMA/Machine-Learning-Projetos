from loguru import logger
import tensorflow as tf

CLASS_NAMES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]


def load_cifar10():
    """Carrega e normaliza a base CIFAR-10."""

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return x_train, y_train, x_test, y_test


def main() -> None:
    """Baixa (se necessário) e verifica a base CIFAR-10."""

    x_train, y_train, x_test, y_test = load_cifar10()
    logger.info(
        "CIFAR-10 carregada: treino={} / {}, teste={} / {}",
        x_train.shape,
        y_train.shape,
        x_test.shape,
        y_test.shape,
    )


if __name__ == "__main__":
    main()
