from pathlib import Path
import shutil
import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(
        "olistbr/brazilian-ecommerce",
        output_dir=str(RAW_DATA_DIR)
    )

    downloaded_path = Path(path)

    archive_dir = RAW_DATA_DIR / "archive"

    if archive_dir.exists():
        for file in archive_dir.iterdir():
            shutil.move(str(file), RAW_DATA_DIR / file.name)

        archive_dir.rmdir()

    print(f"Dataset disponível em: {RAW_DATA_DIR}")

    return RAW_DATA_DIR


if __name__ == "__main__":
    download_dataset()