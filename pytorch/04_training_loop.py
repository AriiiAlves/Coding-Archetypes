"""
04 - THE TRAINING LOOP: the reusable skeleton for training any model.

PyTorch has no .fit(): you write the loop. This is the version to copy.

  for epoch:
      model.train()
      for batch: forward -> loss -> zero_grad -> backward -> (clip) -> step -> scheduler.step
      model.eval(); validate under torch.inference_mode()
      save checkpoint if best; stop early if no improvement

What's covered:
  - optimizer (AdamW) with weight decay only on weights, not biases/norms
  - learning-rate schedule (linear warmup + cosine decay)
  - gradient clipping
  - metrics tracking, best-model checkpointing, early stopping
  - resuming from a checkpoint (model + optimizer + scheduler + epoch)

Run:  python 04_training_loop.py
      python 04_training_loop.py --resume   # continue from outputs/checkpoint.pt
      (keep --epochs the same when resuming: the LR schedule depends on it)
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from config import BATCH_SIZE, CHECKPOINT_PATH, DEVICE, LEARNING_RATE, OUTPUT_DIR, SEED, WEIGHT_DECAY, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

seed_everything()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Data: a synthetic 3-class problem (replace with your Dataset from script 02)
# --------------------------------------------------------------------------- #
N_FEATURES, N_CLASSES = 20, 3
g = torch.Generator().manual_seed(SEED)
X = torch.randn(5000, N_FEATURES, generator=g)
true_w = torch.randn(N_FEATURES, N_CLASSES, generator=g)
y = (X @ true_w + 0.5 * torch.randn(5000, N_CLASSES, generator=g)).argmax(dim=1)

train_ds, val_ds = random_split(TensorDataset(X, y), [0.8, 0.2], generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

# --------------------------------------------------------------------------- #
# 2. Model
# --------------------------------------------------------------------------- #
model = nn.Sequential(
    nn.Linear(N_FEATURES, 128), nn.GELU(), nn.Dropout(0.1),
    nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.1),
    nn.Linear(128, N_CLASSES),
).to(DEVICE)

# --------------------------------------------------------------------------- #
# 3. Optimizer: AdamW is the safe default. Exclude biases and norm weights
#    (1-D tensors) from weight decay - standard practice for transformers too.
#    Alternatives: SGD(momentum=0.9, nesterov=True) for CNNs, Adam, RMSprop.
# --------------------------------------------------------------------------- #
decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
    lr=LEARNING_RATE,
)

# --------------------------------------------------------------------------- #
# 4. LR schedule: linear warmup then cosine decay, stepped once PER BATCH.
#    LambdaLR returns a multiplier of the base LR. Built-in alternatives:
#    OneCycleLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau (step(val_loss) per epoch).
# --------------------------------------------------------------------------- #
total_steps = args.epochs * len(train_loader)
warmup_steps = int(0.05 * total_steps)


def lr_lambda(step: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --------------------------------------------------------------------------- #
# 5. Checkpointing: save EVERYTHING needed to resume, not just the weights
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, epoch: int, best_val_loss: float, bad_epochs: int) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "bad_epochs": bad_epochs,
    }, path)


start_epoch, best_val_loss, bad_epochs = 0, float("inf"), 0
if args.resume and os.path.exists(CHECKPOINT_PATH):
    # weights_only=True (the default) refuses arbitrary pickled objects: safe for untrusted files.
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch, best_val_loss, bad_epochs = ckpt["epoch"] + 1, ckpt["best_val_loss"], ckpt["bad_epochs"]
    print(f"resumed from epoch {ckpt['epoch']}")


# --------------------------------------------------------------------------- #
# 6. One epoch of training, one pass of evaluation
# --------------------------------------------------------------------------- #
def train_one_epoch() -> float:
    model.train()                                    # enable dropout / batchnorm updates
    total_loss, n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        logits = model(xb)                           # forward
        loss = F.cross_entropy(logits, yb)

        optimizer.zero_grad(set_to_none=True)        # clear old gradients (they accumulate!)
        loss.backward()                              # compute new gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevents exploding gradients
        optimizer.step()                             # update weights
        scheduler.step()                             # update learning rate

        total_loss += loss.item() * len(xb)          # .item() syncs with the GPU: fine once per step
        n += len(xb)
    return total_loss / n


@torch.inference_mode()                              # no graph building: faster, less memory
def evaluate() -> tuple[float, float]:
    model.eval()                                     # disable dropout, use running batchnorm stats
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in val_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(xb)
    return total_loss / n, correct / n


# --------------------------------------------------------------------------- #
# 7. The loop, with best-model saving and early stopping
# --------------------------------------------------------------------------- #
patience = 4
best_path = os.path.join(OUTPUT_DIR, "best_model.pt")

for epoch in range(start_epoch, args.epochs):
    t0 = time.perf_counter()
    train_loss = train_one_epoch()
    val_loss, val_acc = evaluate()
    lr = scheduler.get_last_lr()[0]
    print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} | "
          f"acc {val_acc:.3f} | lr {lr:.2e} | {time.perf_counter() - t0:.1f}s")

    if val_loss < best_val_loss:
        best_val_loss, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), best_path)    # weights only: for inference
    else:
        bad_epochs += 1
    save_checkpoint(CHECKPOINT_PATH, epoch, best_val_loss, bad_epochs)  # full state: for resuming

    if bad_epochs >= patience:
        print(f"early stopping: no improvement for {patience} epochs")
        break

# --------------------------------------------------------------------------- #
# 8. Reload the best weights for inference
# --------------------------------------------------------------------------- #
model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
model.eval()
with torch.inference_mode():
    probs = model(X[:3].to(DEVICE)).softmax(dim=-1)
print("predictions:", probs.argmax(dim=-1).tolist(), "true:", y[:3].tolist())
print(f"best val loss: {best_val_loss:.4f}")
