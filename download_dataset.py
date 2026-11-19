from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "huggingface_cache"

ds = load_dataset(
    "Bisher/ASVspoof_2019_LA",
    cache_dir=str(CACHE_DIR)
)

print(ds)