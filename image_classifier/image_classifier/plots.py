from math import ceil

import matplotlib.pyplot as plt


def plot_sample_images(images, labels, class_names, num_images=9):
    """Plota exemplos de imagens de um conjunto de dados."""

    if num_images <= 0:
        raise ValueError("num_images deve ser maior que zero.")
    if num_images > len(images) or num_images > len(labels):
        raise ValueError("num_images não pode exceder a quantidade de imagens ou rótulos.")

    columns = min(3, num_images)
    rows = ceil(num_images / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
        squeeze=False,
    )

    for i in range(num_images):
        axis = axes.flat[i]
        axis.imshow(images[i])
        axis.set_title(class_names[int(labels[i])])
        axis.axis("off")

    for axis in list(axes.flat)[num_images:]:
        figure.delaxes(axis)

    figure.suptitle("Exemplos da base CIFAR-10")
    figure.tight_layout()
    plt.show()


def plot_training_history(history):
    """Plota a evolução da acurácia e da loss durante o treinamento."""

    history_data = history.history

    plt.figure(figsize=(8, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_data["accuracy"], label="Treinamento")
    plt.plot(history_data["val_accuracy"], label="Validação")
    plt.title("Acurácia por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_data["loss"], label="Treinamento")
    plt.plot(history_data["val_loss"], label="Validação")
    plt.title("Loss por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
