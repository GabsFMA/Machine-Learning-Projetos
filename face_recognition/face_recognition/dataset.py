from pathlib import Path

from loguru import logger

from face_recognition.config import AUTHORIZED_DIR, TEST_DIR

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def get_authorized_images(
    authorized_dir: Path = AUTHORIZED_DIR,
) -> dict[str, list[Path]]:
    """Retorna as imagens da galeria, agrupadas pelo nome da pasta da pessoa."""

    authorized_images = {}

    if not authorized_dir.is_dir():
        return authorized_images

    for person_dir in authorized_dir.iterdir():
        if not person_dir.is_dir():
            continue

        images = [
            image_path
            for image_path in person_dir.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if images:
            authorized_images[person_dir.name] = sorted(images)

    return authorized_images


def get_test_images(test_dir: Path = TEST_DIR) -> list[Path]:
    """Retorna as imagens disponíveis para teste."""

    if not test_dir.is_dir():
        return []

    return sorted(
        image_path
        for image_path in test_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    """Mostra a quantidade de imagens da galeria e do diretório de teste."""

    authorized_images = get_authorized_images()
    test_images = get_test_images()

    logger.info(
        "Pessoas cadastradas: {}; imagens autorizadas: {}; imagens para consulta: {}",
        len(authorized_images),
        sum(len(images) for images in authorized_images.values()),
        len(test_images),
    )

    if not authorized_images:
        logger.warning("Galeria vazia. Adicione imagens consentidas em data/authorized/<pessoa>/.")
    if not test_images:
        logger.warning("Nenhuma imagem de consulta encontrada em data/test/.")


if __name__ == "__main__":
    main()
