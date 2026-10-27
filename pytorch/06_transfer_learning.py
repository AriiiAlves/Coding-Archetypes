"""
06 - TRANSFER LEARNING: start from a model pretrained on ImageNet.

With little data, fine-tuning a pretrained network beats training from scratch
by a wide margin. This is the default approach for real-world image tasks.

What's covered:
  - loading pretrained torchvision models with the weights API
  - using the weights' own preprocessing transforms
  - replacing the classification head
  - stage 1: freeze the backbone, train only the head
  - stage 2: unfreeze the last block, fine-tune with per-group learning rates
  - BatchNorm in frozen layers

Run:  python 06_transfer_learning.py
      (downloads CIFAR-10, ~170 MB, and ResNet-18 weights, ~45 MB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2

from config import BATCH_SIZE, DATA_DIR, DEVICE, NUM_WORKERS, seed_everything

seed_everything()

# Pretrained models are slower on CPU than script 05's tiny CNN: keep the subsets small.
TRAIN_SAMPLES, EVAL_SAMPLES, IMAGE_SIZE = 2000, 500, 112

# --------------------------------------------------------------------------- #
# 1. Pretrained model. Browse alternatives with models.list_models():
#    resnet50, efficientnet_b0, convnext_tiny, vit_b_16, mobilenet_v3_small, ...
# --------------------------------------------------------------------------- #
weights = models.ResNet18_Weights.DEFAULT          # best available ImageNet weights
model = models.resnet18(weights=weights)            # same as models.get_model("resnet18", weights="DEFAULT")
print(f"pretrained on {len(weights.meta['categories'])} ImageNet classes")

# --------------------------------------------------------------------------- #
# 2. Preprocessing MUST match what the model was pretrained with
#    (same normalization statistics). weights.transforms() gives the exact recipe.
# --------------------------------------------------------------------------- #
print(weights.transforms())                         # resize 256, crop 224, ImageNet mean/std
norm = v2.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
train_tf = v2.Compose([
    v2.ToImage(), v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)), v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True), norm,
])
eval_tf = v2.Compose([v2.ToImage(), v2.Resize(IMAGE_SIZE), v2.ToDtype(torch.float32, scale=True), norm])

train_ds = Subset(datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf), range(TRAIN_SAMPLES))
test_full = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=eval_tf)
test_ds = Subset(test_full, range(EVAL_SAMPLES))
CLASSES = test_full.classes

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS)

# --------------------------------------------------------------------------- #
# 3. Replace the head: ImageNet's 1000 classes -> our 10.
#    The attribute name differs per family: resnet .fc, efficientnet/mobilenet
#    .classifier[-1], vit .heads.head, convnext .classifier[2]. Check print(model).
# --------------------------------------------------------------------------- #
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(DEVICE)


def set_trainable(trainable_prefixes: tuple[str, ...]) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(trainable_prefixes)


def run_epoch(optimizer, freeze_bn: bool) -> float:
    model.train()
    if freeze_bn:
        # In a frozen backbone, BatchNorm would still update its running statistics
        # in train mode. Put those layers in eval mode to keep the pretrained stats.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
    total = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(train_loader)


@torch.inference_mode()
def accuracy() -> float:
    model.eval()
    correct = 0
    for xb, yb in test_loader:
        correct += (model(xb.to(DEVICE)).argmax(1).cpu() == yb).sum().item()
    return correct / len(test_ds)


print(f"accuracy before training: {accuracy():.3f}  (random head ~0.10)")

# --------------------------------------------------------------------------- #
# 4. Stage 1 - linear probe: only the new head learns. Fast and stable,
#    because the random head can't destroy the pretrained features.
# --------------------------------------------------------------------------- #
set_trainable(("fc.",))
head_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
loss = run_epoch(head_opt, freeze_bn=True)
print(f"stage 1 (head only)       | loss {loss:.3f} | acc {accuracy():.3f}")

# --------------------------------------------------------------------------- #
# 5. Stage 2 - fine-tune the last block too, with a SMALLER learning rate for
#    pretrained layers than for the new head ("discriminative learning rates").
#    Unfreeze more blocks (layer3, layer2, ...) if you have more data.
# --------------------------------------------------------------------------- #
set_trainable(("fc.", "layer4."))
ft_opt = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
], weight_decay=1e-4)
loss = run_epoch(ft_opt, freeze_bn=True)
print(f"stage 2 (layer4 + head)   | loss {loss:.3f} | acc {accuracy():.3f}")

# --------------------------------------------------------------------------- #
# 6. Use the backbone as a feature extractor (embeddings for search, clustering,
#    or a scikit-learn classifier): drop the head with nn.Identity.
# --------------------------------------------------------------------------- #
model.fc = nn.Identity()
model.eval()
with torch.inference_mode():
    feats = model(next(iter(test_loader))[0][:4].to(DEVICE))
print("feature vectors:", feats.shape)          # (4, 512)
