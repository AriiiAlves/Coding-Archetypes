"""
05 - CNN IMAGE CLASSIFIER: a convolutional network trained from scratch.

Applies the loop from script 04 to images (FashionMNIST, 28x28 grayscale, 10 classes).

What's covered:
  - Conv2d / BatchNorm2d / MaxPool2d / AdaptiveAvgPool2d building blocks
  - train-time data augmentation vs. plain eval transforms
  - label smoothing, OneCycleLR
  - per-class accuracy and a confusion matrix
  - predicting on a single image

Swap the dataset for your own photos with datasets.ImageFolder (one folder per
class) and in_channels=3; for real photos prefer transfer learning (script 06).

Run:  python 05_cnn_image_classifier.py
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2

from config import (
    BATCH_SIZE,
    CNN_WEIGHTS,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    MAX_EVAL_SAMPLES,
    MAX_TRAIN_SAMPLES,
    NUM_WORKERS,
    OUTPUT_DIR,
    seed_everything,
)

seed_everything()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Data. Augmentation ONLY on the training set: it creates new variations
#    so the model generalizes; evaluation must see unmodified images.
# --------------------------------------------------------------------------- #
MEAN, STD = [0.2860], [0.3530]
train_tf = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(MEAN, STD),
])
eval_tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

train_full = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=train_tf)
test_full = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=eval_tf)
CLASSES = train_full.classes

train_ds = Subset(train_full, range(MAX_TRAIN_SAMPLES)) if MAX_TRAIN_SAMPLES else train_full
test_ds = Subset(test_full, range(MAX_EVAL_SAMPLES)) if MAX_EVAL_SAMPLES else test_full

loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda", persistent_workers=NUM_WORKERS > 0)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_kwargs)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, **loader_kwargs)


# --------------------------------------------------------------------------- #
# 2. Model: [conv -> BN -> ReLU] x2 -> pool, repeated, then global pooling.
#    Each stage doubles the channels and halves the resolution.
# --------------------------------------------------------------------------- #
def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),  # bias is redundant before BatchNorm
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),                                   # halves H and W
    )


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 10, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, width),       # 28 -> 14
            conv_block(width, width * 2),         # 14 -> 7
            conv_block(width * 2, width * 4),     # 7 -> 3
        )
        # Global average pooling makes the head independent of the input resolution.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(width * 4, n_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


model = SmallCNN(n_classes=len(CLASSES)).to(DEVICE)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# --------------------------------------------------------------------------- #
# 3. Optimizer + OneCycleLR (warmup and annealing in one; strong default for CNNs)
# --------------------------------------------------------------------------- #
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, total_steps=EPOCHS * len(train_loader)
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # soft targets: reduces over-confidence


# --------------------------------------------------------------------------- #
# 4. Train / evaluate
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate():
    model.eval()
    confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for xb, yb in test_loader:
        preds = model(xb.to(DEVICE)).argmax(dim=1).cpu()
        # Row = true class, column = predicted class.
        confusion += torch.bincount(yb * len(CLASSES) + preds, minlength=len(CLASSES) ** 2).view(
            len(CLASSES), len(CLASSES)
        )
    return confusion


for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for step, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += loss.item()
        if step % 50 == 0:
            print(f"epoch {epoch} step {step:4d}/{len(train_loader)} loss {loss.item():.4f}")

    confusion = evaluate()
    acc = confusion.diag().sum().item() / confusion.sum().item()
    print(f"epoch {epoch} | train loss {running / len(train_loader):.4f} | test acc {acc:.3f}")

# --------------------------------------------------------------------------- #
# 5. Per-class accuracy shows WHERE the model struggles (shirts vs. T-shirts...)
# --------------------------------------------------------------------------- #
per_class = confusion.diag() / confusion.sum(dim=1).clamp(min=1)
for name, a in zip(CLASSES, per_class.tolist()):
    print(f"  {name:12s} {a:.2f}")

torch.save(model.state_dict(), CNN_WEIGHTS)

# --------------------------------------------------------------------------- #
# 6. Single-image prediction: add a batch dimension with unsqueeze(0)
# --------------------------------------------------------------------------- #
img, label = test_full[0]
model.eval()
with torch.inference_mode():
    probs = F.softmax(model(img.unsqueeze(0).to(DEVICE)), dim=-1)[0]
top_p, top_i = probs.topk(3)
print("true:", CLASSES[label])
for p, i in zip(top_p.tolist(), top_i.tolist()):
    print(f"  {CLASSES[i]:12s} {p:.2%}")
