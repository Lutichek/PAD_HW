"""Константы датасета CIFAR-10 и пути по умолчанию."""

from pathlib import Path

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
# Официальная контрольная сумма архива с https://www.cs.toronto.edu/~kriz/cifar.html
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
RAW_KEY = f"raw/{ARCHIVE_NAME}"

CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
TRAIN_SIZE = 50_000
TEST_SIZE = 10_000

# Пути по умолчанию (относительно корня проекта)
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim" / "cifar10"
PROCESSED_DIR = DATA_DIR / "processed" / "cifar10"
