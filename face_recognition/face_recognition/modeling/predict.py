from dataclasses import dataclass
from pathlib import Path
import sys

from face_recognition.config import DEFAULT_MODEL_NAME
from face_recognition.dataset import get_authorized_images


def _load_deepface():
    output = sys.stdout
    reconfigure = getattr(output, "reconfigure", None)
    original_errors = getattr(output, "errors", None)
    encoding = getattr(output, "encoding", None)
    needs_encoding_guard = (
        callable(reconfigure)
        and encoding is not None
        and encoding.lower().replace("-", "") != "utf8"
        and original_errors != "backslashreplace"
    )

    if needs_encoding_guard:
        reconfigure(errors="backslashreplace")

    try:
        from deepface import DeepFace
    finally:
        if needs_encoding_guard:
            reconfigure(errors=original_errors)

    return DeepFace


@dataclass(frozen=True)
class RecognitionResult:
    identity: str | None
    distance: float
    threshold: float


def recognize_image(
    image_path: Path,
    authorized_images: dict[str, list[Path]] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> RecognitionResult:
    """Compara uma imagem com a galeria e retorna identidade ou pessoa desconhecida."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Imagem de consulta não encontrada: {image_path}")

    gallery = get_authorized_images() if authorized_images is None else authorized_images
    if not gallery or not any(gallery.values()):
        raise ValueError("A galeria está vazia; cadastre imagens em data/authorized/<pessoa>/.")

    DeepFace = _load_deepface()

    best_identity = None
    best_distance = float("inf")
    best_threshold = 0.0

    for identity, reference_images in gallery.items():
        for reference_path in reference_images:
            if not reference_path.is_file():
                raise FileNotFoundError(f"Imagem da galeria não encontrada: {reference_path}")

            verification = DeepFace.verify(
                img1_path=str(image_path),
                img2_path=str(reference_path),
                model_name=model_name,
                detector_backend="opencv",
                distance_metric="cosine",
                enforce_detection=True,
                silent=True,
            )

            distance = float(verification["distance"])
            threshold = float(verification["threshold"])
            if distance < best_distance:
                best_distance = distance
                best_threshold = threshold
                best_identity = identity if verification["verified"] else None

    if best_identity is None and best_distance == float("inf"):
        raise ValueError("A galeria não contém imagens válidas.")

    return RecognitionResult(
        identity=best_identity,
        distance=best_distance,
        threshold=best_threshold,
    )
