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