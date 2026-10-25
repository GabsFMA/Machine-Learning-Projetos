import csv
from pathlib import Path
from typing import Annotated

from loguru import logger
import typer

from face_recognition.config import (
    AUTHORIZED_DIR,
    DEFAULT_MODEL_NAME,
    RECOGNITION_RESULTS_PATH,
    TEST_DIR,
)
from face_recognition.dataset import (
    get_authorized_images,
    get_test_images,
)
from face_recognition.modeling.predict import recognize_image

app = typer.Typer(help="Reconhece faces usando uma galeria local de pessoas autorizadas.")


@app.command()
def recognize(
    image: Annotated[
        Path,
        typer.Option("--image", help="Imagem a ser comparada."),
    ],
    authorized_dir: Annotated[
        Path,
        typer.Option(help="Diretório da galeria."),
    ] = AUTHORIZED_DIR,
    model_name: Annotated[
        str,
        typer.Option(help="Modelo de embeddings DeepFace."),
    ] = DEFAULT_MODEL_NAME,
) -> None:
    """Compara uma imagem com a galeria local."""

    result = recognize_image(
        image_path=image,
        authorized_images=get_authorized_images(authorized_dir),
        model_name=model_name,
    )

    if result.identity is None:
        logger.info(
            "Pessoa não reconhecida (distância {:.4f}; limite {:.4f}).",
            result.distance,
            result.threshold,
        )
        return

    logger.info(
        "Identidade correspondente: {} (distância {:.4f}; limite {:.4f}).",
        result.identity,
        result.distance,
        result.threshold,
    )


@app.command()
def recognize_test_set(
    test_dir: Annotated[
        Path,
        typer.Option(help="Diretório com imagens de consulta."),
    ] = TEST_DIR,
    authorized_dir: Annotated[
        Path,
        typer.Option(help="Diretório da galeria."),
    ] = AUTHORIZED_DIR,
    output_path: Annotated[
        Path,
        typer.Option(help="Caminho do relatório CSV."),
    ] = RECOGNITION_RESULTS_PATH,
    model_name: Annotated[
        str,
        typer.Option(help="Modelo de embeddings DeepFace."),
    ] = DEFAULT_MODEL_NAME,
) -> None:
    """Reconhece as imagens do diretório de teste e grava um relatório CSV."""

    test_images = get_test_images(test_dir)
    if not test_images:
        raise typer.BadParameter(f"Nenhuma imagem encontrada em {test_dir}.")

    gallery = get_authorized_images(authorized_dir)
    if not gallery or not any(gallery.values()):
        raise typer.BadParameter(f"Nenhuma imagem cadastrada em {authorized_dir}.")

    results = []
    for image_path in test_images:
        result = recognize_image(
            image_path=image_path,
            authorized_images=gallery,
            model_name=model_name,
        )
        results.append(
            {
                "image": str(image_path),
                "identity": result.identity or "",
                "distance": result.distance,
                "threshold": result.threshold,
                "recognized": result.identity is not None,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=("image", "identity", "distance", "threshold", "recognized"),
        )
        writer.writeheader()
        writer.writerows(results)

    logger.success("Relatório de reconhecimento salvo em: {}", output_path)


if __name__ == "__main__":
    app()
