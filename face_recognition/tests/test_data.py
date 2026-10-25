import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from face_recognition.dataset import get_authorized_images, get_test_images
from face_recognition.main import recognize_test_set
from face_recognition.modeling.predict import _load_deepface, recognize_image


def test_get_authorized_images_groups_supported_files(tmp_path):
    first_person = tmp_path / "alice"
    second_person = tmp_path / "bob"
    first_person.mkdir()
    second_person.mkdir()
    (first_person / "second.JPG").touch()
    (first_person / "first.png").touch()
    (first_person / "notes.txt").touch()
    (second_person / "portrait.jpeg").touch()

    images = get_authorized_images(tmp_path)

    assert list(images) == ["alice", "bob"]
    assert [path.name for path in images["alice"]] == ["first.png", "second.JPG"]
    assert [path.name for path in images["bob"]] == ["portrait.jpeg"]


def test_get_authorized_images_returns_empty_for_missing_directory(tmp_path):
    assert get_authorized_images(tmp_path / "missing") == {}


def test_get_test_images_returns_sorted_supported_files(tmp_path):
    (tmp_path / "z.jpeg").touch()
    (tmp_path / "a.PNG").touch()
    (tmp_path / "ignore.gif").touch()
    (tmp_path / "subdir").mkdir()

    assert [path.name for path in get_test_images(tmp_path)] == ["a.PNG", "z.jpeg"]


def test_recognize_image_returns_closest_verified_identity(monkeypatch, tmp_path):
    query = tmp_path / "query.jpg"
    reference_a = tmp_path / "alice.jpg"
    reference_b = tmp_path / "bob.jpg"
    for image in (query, reference_a, reference_b):
        image.touch()

    class FakeDeepFace:
        @staticmethod
        def verify(img1_path, img2_path, **kwargs):
            assert img1_path == str(query)
            assert kwargs["enforce_detection"] is True
            if img2_path == str(reference_a):
                return {"distance": 0.3, "threshold": 0.4, "verified": True}
            return {"distance": 0.2, "threshold": 0.4, "verified": True}

    monkeypatch.setitem(sys.modules, "deepface", SimpleNamespace(DeepFace=FakeDeepFace))

    result = recognize_image(
        query,
        {"alice": [reference_a], "bob": [reference_b]},
    )

    assert result.identity == "bob"
    assert result.distance == pytest.approx(0.2)
    assert result.threshold == pytest.approx(0.4)


def test_deepface_import_restores_console_error_handler(monkeypatch):
    class FakeOutput:
        encoding = "cp1252"
        errors = "strict"

        def __init__(self):
            self.error_handlers = []

        def reconfigure(self, *, errors):
            self.error_handlers.append(errors)
            self.errors = errors

    output = FakeOutput()
    deepface_module = SimpleNamespace(DeepFace=object())
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setitem(sys.modules, "deepface", deepface_module)

    assert _load_deepface() is deepface_module.DeepFace
    assert output.error_handlers == ["backslashreplace", "strict"]


def test_recognize_image_returns_unknown_for_unverified_match(monkeypatch, tmp_path):
    query = tmp_path / "query.jpg"
    reference = tmp_path / "alice.jpg"
    query.touch()
    reference.touch()

    class FakeDeepFace:
        @staticmethod
        def verify(img1_path, img2_path, **kwargs):
            return {"distance": 0.7, "threshold": 0.4, "verified": False}

    monkeypatch.setitem(sys.modules, "deepface", SimpleNamespace(DeepFace=FakeDeepFace))

    result = recognize_image(query, {"alice": [reference]})

    assert result.identity is None
    assert result.distance == pytest.approx(0.7)


def test_recognize_image_rejects_missing_query(tmp_path):
    with pytest.raises(FileNotFoundError, match="Imagem de consulta"):
        recognize_image(tmp_path / "missing.jpg", {"alice": [tmp_path / "alice.jpg"]})


def test_recognize_image_rejects_empty_gallery(tmp_path):
    query = tmp_path / "query.jpg"
    query.touch()

    with pytest.raises(ValueError, match="galeria está vazia"):
        recognize_image(query, {})


def test_recognize_test_set_writes_csv_report(monkeypatch, tmp_path):
    authorized_dir = tmp_path / "authorized"
    person_dir = authorized_dir / "alice"
    person_dir.mkdir(parents=True)
    (person_dir / "reference.jpg").touch()
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "query.jpg").touch()
    output = tmp_path / "reports" / "results.csv"

    monkeypatch.setattr(
        "face_recognition.main.recognize_image",
        lambda image_path, authorized_images, model_name: SimpleNamespace(
            identity="alice",
            distance=0.12,
            threshold=0.4,
        ),
    )

    recognize_test_set(
        test_dir=test_dir,
        authorized_dir=authorized_dir,
        output_path=output,
    )

    with output.open(encoding="utf-8-sig", newline="") as report_file:
        rows = list(csv.DictReader(report_file))

    assert rows == [
        {
            "image": str(test_dir / "query.jpg"),
            "identity": "alice",
            "distance": "0.12",
            "threshold": "0.4",
            "recognized": "True",
        }
    ]


def test_recognize_test_set_rejects_empty_test_directory(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()

    with pytest.raises(typer.BadParameter, match="Nenhuma imagem encontrada"):
        recognize_test_set(
            test_dir=test_dir,
            authorized_dir=tmp_path / "authorized",
            output_path=tmp_path / "results.csv",
        )
