"""
Central configuration shared by every script in this archetype.

Change hyperparameters and paths here once instead of hunting through each file.
All defaults are SMALL (tiny models, a few epochs, subsets of data) so every
script finishes in a minute or two on a laptop CPU. Raise them when you have a GPU.
"""

import os
import random
import sys

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_DIR = os.getenv("TORCH_DATA_DIR", "data")          # torchvision downloads land here
OUTPUT_DIR = os.getenv("TORCH_OUTPUT_DIR", "outputs")   # checkpoints, exports, logs
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")
CNN_WEIGHTS = os.path.join(OUTPUT_DIR, "cnn.pt")
CHAR_MODEL_WEIGHTS = os.path.join(OUTPUT_DIR, "char_transformer.pt")

# --------------------------------------------------------------------------- #
# Training defaults (override per script when it makes sense)
# --------------------------------------------------------------------------- #
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EPOCHS = int(os.getenv("EPOCHS", 3))
LEARNING_RATE = 1e-3          # good default for AdamW on small models trained from scratch
WEIGHT_DECAY = 0.01
# Cap dataset sizes so the image scripts stay fast; set to 0 for the full dataset.
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", 10_000))
MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES", 2_000))
# DataLoader worker processes. 0 = load in the main process (simplest to debug).
# macOS and Windows start workers by re-importing the script, which needs an
# `if __name__ == "__main__":` guard these flat scripts don't have, so default to 0 there.
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2 if sys.platform == "linux" else 0))
SEED = 42


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# Mixed precision dtype: bfloat16 on modern GPUs, float16 on older NVIDIA GPUs.
AMP_DTYPE = (
    torch.float16
    if DEVICE.type == "cuda" and not torch.cuda.is_bf16_supported()
    else torch.bfloat16
)
# Mixed precision pays off on GPUs. On CPUs without native bfloat16 instructions
# (most consumer CPUs) it is many times SLOWER, so it is off there by default.
USE_AMP = DEVICE.type in ("cuda", "mps") or os.getenv("FORCE_AMP") == "1"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = SEED) -> None:
    """Seed every RNG PyTorch training touches. Call once at the top of a script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS
    # For bit-exact runs (slower), also: torch.use_deterministic_algorithms(True)
