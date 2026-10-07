from pathlib import Path

from loguru import logger


PROJ_ROOT = Path(__file__).resolve().parents[1]

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")


# Diretórios de dados
DATA_DIR = PROJ_ROOT / "data"

AUTHORIZED_DIR = DATA_DIR / "authorized"
TEST_DIR = DATA_DIR / "test"
PROCESSED_DIR = DATA_DIR / "processed"


# Diretório de modelos
MODELS_DIR = PROJ_ROOT / "models"