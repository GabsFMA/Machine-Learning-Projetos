import matplotlib.pyplot as plt


def plot_sample_images(images, labels, class_names, num_images=9):
    """Plota exemplos de imagens de um conjunto de dados."""

    plt.figure(figsize=(10, 10))

    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.suptitle("Exemplos da base CIFAR-10")
    plt.tight_layout()
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