import matplotlib
import numpy as np
import pytest
import tensorflow as tf

matplotlib.use("Agg")

from image_classifier import dataset
from image_classifier import main
from image_classifier.modeling import predict as prediction
from image_classifier.modeling.model import create_model
from image_classifier.modeling.train import train_model
from image_classifier.plots import plot_sample_images


def test_load_cifar10_normalizes_images_and_flattens_labels(monkeypatch):
    images = np.array(
        [
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.full((32, 32, 3), 255, dtype=np.uint8),
        ]
    )
    labels = np.array([[2], [7]], dtype=np.uint8)
    raw_data = ((images, labels), (images, labels))
    monkeypatch.setattr(
        tf.keras.datasets.cifar10,
        "load_data",
        lambda: raw_data,
    )

    x_train, y_train, x_test, y_test = dataset.load_cifar10()

    assert x_train.shape == (2, 32, 32, 3)
    assert x_train.dtype == np.float32
    assert x_train.min() == 0
    assert x_train.max() == 1
    assert y_train.tolist() == [2, 7]
    assert x_test.shape == (2, 32, 32, 3)
    assert y_test.tolist() == [2, 7]


def test_model_outputs_probabilities_for_all_classes():
    model = create_model()

    probabilities = model(np.zeros((2, 32, 32, 3), dtype=np.float32)).numpy()

    assert probabilities.shape == (2, 10)
    np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1], atol=1e-6)


def test_save_and_load_model(tmp_path):
    model = create_model()
    model_path = tmp_path / "classifier.keras"

    saved_path = prediction.save_model(model, model_path)
    loaded_model = prediction.load_model(saved_path)

    assert saved_path == model_path
    assert loaded_model.output_shape == (None, 10)


def test_load_model_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
        prediction.load_model(tmp_path / "missing.keras")


def test_evaluate_model_returns_python_floats():
    class FakeModel:
        def evaluate(self, images, labels, verbose=0):
            return np.float32(0.4), np.float32(0.8)

    loss, accuracy = prediction.evaluate_model(
        FakeModel(),
        np.zeros((1, 32, 32, 3)),
        np.zeros(1),
    )

    assert loss == pytest.approx(0.4)
    assert accuracy == pytest.approx(0.8)
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)


def test_predict_returns_class_indices_and_probabilities():
    class FakeModel:
        def predict(self, images, verbose=0):
            return np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])

    images = np.zeros((2, 32, 32, 3))

    classes = prediction.predict(FakeModel(), images)
    probabilities = prediction.predict_proba(FakeModel(), images)

    assert classes.tolist() == [1, 0]
    assert probabilities.shape == (2, 3)


def test_plot_sample_images_adapts_to_requested_count(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    images = np.zeros((10, 32, 32, 3))
    labels = np.zeros(10, dtype=int)

    plot_sample_images(images, labels, ["classe"], num_images=10)

    assert len(plt.gcf().axes) == 10
    plt.close("all")


@pytest.mark.parametrize(
    ("num_images", "images", "labels"),
    [
        (0, np.zeros((1, 32, 32, 3)), np.zeros(1)),
        (2, np.zeros((1, 32, 32, 3)), np.zeros(1)),
    ],
)
def test_plot_sample_images_rejects_invalid_count(num_images, images, labels):
    with pytest.raises(ValueError, match="num_images"):
        plot_sample_images(images, labels, ["classe"], num_images=num_images)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"validation_split": 1}, "validation_split"),
    ],
)
def test_train_model_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        train_model(np.zeros((1, 32, 32, 3)), np.zeros(1), **kwargs)


def test_train_command_evaluates_and_saves_model(monkeypatch, tmp_path):
    images = np.zeros((2, 32, 32, 3))
    labels = np.zeros(2)
    model = object()
    saved_paths = []

    monkeypatch.setattr(
        main,
        "load_cifar10",
        lambda: (images, labels, images, labels),
    )
    monkeypatch.setattr(
        main,
        "train_model",
        lambda *args, **kwargs: (model, object()),
    )
    monkeypatch.setattr(
        main,
        "evaluate_model",
        lambda *args: (0.25, 0.9),
    )
    monkeypatch.setattr(
        main,
        "save_model",
        lambda saved_model, path: saved_paths.append((saved_model, path)) or path,
    )

    model_path = tmp_path / "model.keras"
    main.train(model_path=model_path)

    assert saved_paths == [(model, model_path)]
