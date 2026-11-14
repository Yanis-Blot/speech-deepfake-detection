from datasets import load_dataset

DATASET_NAME = "Bisher/ASVspoof_2019_LA"


def load_splits(splits=("train", "validation")):
    """
    Download (first call) or read from cache (next calls) the requested splits.
    Using Huggingface API to retrieve the dataset splits (choices: train, validation, test)
    Returns a DatasetDict: {"train": ..., "validation": ...}.
    """
    ds = load_dataset(DATASET_NAME,
                      data_files={split: f"data/{split}-*" for split in splits},
                      )
    return ds

