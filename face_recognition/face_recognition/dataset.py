from pathlib import Path

from face_recognition.config import AUTHORIZED_DIR, TEST_DIR


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def get_authorized_images():
    """Retorna as imagens cadastradas para cada pessoa autorizada."""

    authorized_images = {}

    if not AUTHORIZED_DIR.exists():
        return authorized_images

    for person_dir in AUTHORIZED_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        images = [
            image_path
            for image_path in person_dir.iterdir()
            if image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images:
            authorized_images[person_dir.name] = sorted(images)

    return authorized_images


def get_test_images():
    """Retorna as imagens disponíveis para teste."""

    if not TEST_DIR.exists():
        return []

    return sorted(
        image_path
        for image_path in TEST_DIR.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )