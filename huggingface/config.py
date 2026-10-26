"""
Central configuration shared by every script in this archetype.

Change model names here once instead of hunting through each file.
All defaults are SMALL models so everything runs on a laptop CPU.
Swap them for bigger ones (e.g. "meta-llama/Llama-3.2-1B-Instruct",
"Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3") when you have a GPU.
"""

import os

import torch

# --------------------------------------------------------------------------- #
# Models (any model id from https://huggingface.co/models works)
# --------------------------------------------------------------------------- #
# Small instruction-tuned chat model (decoder-only / causal LM)
CHAT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Encoder model used for classification fine-tuning
CLASSIFIER_BASE_MODEL = "distilbert-base-uncased"

# Sentence-embedding model used for semantic search
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------------- #
# Datasets (any dataset id from https://huggingface.co/datasets works)
# --------------------------------------------------------------------------- #
CLASSIFICATION_DATASET = "stanfordnlp/imdb"    # columns: text, label
INSTRUCTION_DATASET = "yahma/alpaca-cleaned"   # columns: instruction, input, output

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
OUTPUT_DIR = os.getenv("HF_OUTPUT_DIR", "outputs")
CLASSIFIER_OUTPUT = os.path.join(OUTPUT_DIR, "classifier")
LORA_OUTPUT = os.path.join(OUTPUT_DIR, "lora-adapter")
MERGED_OUTPUT = os.path.join(OUTPUT_DIR, "merged-model")
CUSTOM_MODEL_OUTPUT = os.path.join(OUTPUT_DIR, "custom-model")

# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
def get_device() -> str:
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


DEVICE = get_device()

# bfloat16 halves memory on modern GPUs; float32 is the safe CPU default.
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

# Keep experiments small & fast by default; raise these for real training.
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", 1000))
MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES", 200))
SEED = 42
