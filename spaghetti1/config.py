from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"
COMPETITION = "an2dl2627"
IMAGE_SIZE = (256, 256)
SPLIT_CSVS = {"train": "train.csv", "test": "sample_submission.csv"}
EXPECTED_COUNTS = {"train": 4328, "test": 14736}
CLASS_NAMES = {
    0: "Spaghetti alla rinfusa",
    1: "Gemelli",
    2: "Gnocchi",
    3: "Ravioli",
    4: "Penne",
    5: "Fusilli con la sbarra",
    6: "Fusilli stretti",
    7: "Riccioli",
    8: "Linguine",
    9: "Lasagna",
}
