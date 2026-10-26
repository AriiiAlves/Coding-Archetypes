"""
03 - DATASETS: load, inspect, transform and save data with 🤗 Datasets.

What's covered:
  - load_dataset from the Hub and from local files (csv / json / parquet)
  - inspect, filter, shuffle, select, train/test split
  - map() for tokenization (batched, fast, cached)
  - building a Dataset from Python objects
  - saving to disk / pushing to the Hub

Run:  python 03_datasets.py
"""

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from config import CLASSIFICATION_DATASET, CLASSIFIER_BASE_MODEL, OUTPUT_DIR, SEED

# --------------------------------------------------------------------------- #
# 1. Load from the Hub -> DatasetDict({"train": Dataset, "test": Dataset, ...})
# --------------------------------------------------------------------------- #
ds = load_dataset(CLASSIFICATION_DATASET)
print(ds)
print("Features:", ds["train"].features)
print("First row:", ds["train"][0]["text"][:200], "...")

# Only grab part of a split (handy for quick experiments):
small = load_dataset(CLASSIFICATION_DATASET, split="train[:1%]")
print("1% slice size:", len(small))

# Huge datasets: stream instead of downloading everything
# streamed = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
# for row in streamed.take(3): print(row)

# --------------------------------------------------------------------------- #
# 2. Load local files (same API, different "builder")
# --------------------------------------------------------------------------- #
# load_dataset("csv",     data_files={"train": "train.csv", "test": "test.csv"})
# load_dataset("json",    data_files="data.jsonl")
# load_dataset("parquet", data_files="data/*.parquet")
# load_dataset("imagefolder", data_dir="images/")   # folder-per-class images

# --------------------------------------------------------------------------- #
# 3. Build a dataset from Python data
# --------------------------------------------------------------------------- #
custom = Dataset.from_dict({
    "text": ["good product", "terrible service", "it was ok"],
    "label": [1, 0, 1],
})
# Also: Dataset.from_list([{...}, ...]), Dataset.from_pandas(df)
print(custom)

# --------------------------------------------------------------------------- #
# 4. Common transformations (each returns a NEW dataset; originals are immutable)
# --------------------------------------------------------------------------- #
train = ds["train"].shuffle(seed=SEED).select(range(2000))      # subsample
train = train.filter(lambda row: len(row["text"]) > 100)         # drop short rows
train = train.rename_column("label", "labels")                   # Trainer expects "labels"

# Create a validation split from train
splits = train.train_test_split(test_size=0.1, seed=SEED)
data = DatasetDict({"train": splits["train"], "validation": splits["test"]})
print(data)

# --------------------------------------------------------------------------- #
# 5. Tokenize with map(batched=True) - fast and cached on disk
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_BASE_MODEL)


def tokenize(batch):
    # Padding is left to a DataCollator at training time (dynamic padding = faster)
    return tokenizer(batch["text"], truncation=True, max_length=256)


tokenized = data.map(tokenize, batched=True, remove_columns=["text"])
print(tokenized)
print("Example input_ids:", tokenized["train"][0]["input_ids"][:20])

# Return PyTorch tensors when indexing (useful for custom training loops)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# --------------------------------------------------------------------------- #
# 6. Save / reload / share
# --------------------------------------------------------------------------- #
tokenized.save_to_disk(f"{OUTPUT_DIR}/tokenized-imdb")
# reloaded = DatasetDict.load_from_disk(f"{OUTPUT_DIR}/tokenized-imdb")
# data.push_to_hub("your-username/my-dataset", private=True)   # needs `hf auth login`
# data["train"].to_csv("train.csv"); data["train"].to_pandas()
