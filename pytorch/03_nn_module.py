"""
03 - nn.Module: how models are built.

Every model, layer and loss in PyTorch is an nn.Module. You define the layers
in __init__ and the computation in forward(). PyTorch finds the parameters,
moves them between devices and saves them for you.

What's covered:
  - the core layers: Linear, Conv2d, Embedding, norms, Dropout, activations
  - nn.Sequential, nn.ModuleList, nn.ModuleDict
  - writing a custom Module (with a residual block)
  - inspecting parameters, counting them, freezing them
  - buffers (state that is saved but not trained)
  - train() vs eval() mode
  - weight initialization
  - loss functions and torch.nn.functional

Run:  python 03_nn_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, seed_everything

seed_everything()

# --------------------------------------------------------------------------- #
# 1. Core layers (shapes in comments: B = batch size)
# --------------------------------------------------------------------------- #
linear = nn.Linear(16, 32)                       # (B, 16) -> (B, 32)
conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # (B, 3, H, W) -> (B, 8, H, W)
embed = nn.Embedding(num_embeddings=1000, embedding_dim=64)  # (B, T) int ids -> (B, T, 64)
layer_norm = nn.LayerNorm(64)                     # normalizes the last dimension (transformers)
batch_norm = nn.BatchNorm2d(8)                    # normalizes per channel over the batch (CNNs)
dropout = nn.Dropout(p=0.1)                       # active only in train() mode
act = nn.GELU()                                   # also: ReLU, SiLU, Tanh, Sigmoid
# Others you will meet: LSTM/GRU, MultiheadAttention, TransformerEncoderLayer, MaxPool2d, AdaptiveAvgPool2d

print(embed(torch.tensor([[1, 2, 3]])).shape)     # (1, 3, 64)

# --------------------------------------------------------------------------- #
# 2. nn.Sequential: chain layers when forward() is just "apply in order"
# --------------------------------------------------------------------------- #
mlp = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
)
print(mlp(torch.randn(4, 8)).shape)               # (4, 1)


# --------------------------------------------------------------------------- #
# 3. A custom Module: the pattern for anything non-sequential
#    (skip connections, multiple inputs/outputs, branching, loops).
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()                         # always call this first
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.ff(self.norm(x)))   # pre-norm residual: x + f(norm(x))


class TabularNet(nn.Module):
    """Input projection -> N residual blocks -> one head per task."""

    def __init__(self, in_features: int, hidden: int, n_blocks: int, n_classes: int):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        # ModuleList (not a plain Python list!) so the blocks are registered:
        # their parameters are found by .parameters(), .to(), state_dict(), ...
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])
        # ModuleDict: named sub-modules, handy for multi-task heads.
        self.heads = nn.ModuleDict({
            "classify": nn.Linear(hidden, n_classes),
            "regress": nn.Linear(hidden, 1),
        })
        # Buffer: saved in state_dict and moved by .to(), but NOT trained.
        # Typical uses: normalization statistics, fixed masks, positional tables.
        self.register_buffer("input_mean", torch.zeros(in_features))

        self.apply(self._init_weights)             # runs on every sub-module

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Defaults are usually fine; custom init matters most for deep or from-scratch models.
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.proj(x - self.input_mean)
        for block in self.blocks:
            h = block(h)
        return {name: head(h) for name, head in self.heads.items()}


model = TabularNet(in_features=8, hidden=64, n_blocks=3, n_classes=4).to(DEVICE)
print(model)                                          # readable tree of sub-modules

out = model(torch.randn(16, 8, device=DEVICE))        # call the module, never .forward() directly
print({k: v.shape for k, v in out.items()})

# --------------------------------------------------------------------------- #
# 4. Inspecting and freezing parameters
# --------------------------------------------------------------------------- #
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters: {n_params:,} (trainable: {n_trainable:,})")

for name, p in list(model.named_parameters())[:4]:
    print(f"  {name:30s} {tuple(p.shape)}")

# Freeze everything except the classification head (fine-tuning pattern, see script 06).
for name, p in model.named_parameters():
    p.requires_grad = name.startswith("heads.classify")
print("trainable after freezing:", sum(p.numel() for p in model.parameters() if p.requires_grad))
model.requires_grad_(True)                            # unfreeze everything again

# state_dict: an ordered dict of every parameter and buffer. This is what you save.
print("state_dict keys:", list(model.state_dict())[:3], "...")

# --------------------------------------------------------------------------- #
# 5. train() vs eval(): changes Dropout and BatchNorm behavior.
#    Forgetting model.eval() before validation is a classic bug.
# --------------------------------------------------------------------------- #
x = torch.randn(4, 8, device=DEVICE)
model.train()
print("train mode, two calls equal?", torch.equal(model(x)["regress"], model(x)["regress"]))  # False: dropout
model.eval()
with torch.inference_mode():                          # like no_grad, but even faster for pure inference
    print("eval mode, two calls equal?", torch.equal(model(x)["regress"], model(x)["regress"]))  # True

# --------------------------------------------------------------------------- #
# 6. Loss functions. Note the expected inputs, they cause most shape errors.
# --------------------------------------------------------------------------- #
logits = torch.randn(4, 3)                   # raw scores, NO softmax applied
targets = torch.tensor([0, 2, 1, 2])         # int64 class indices
print("cross entropy:", F.cross_entropy(logits, targets).item())   # = nn.CrossEntropyLoss()(...)

binary_logits = torch.randn(4)
binary_targets = torch.tensor([1.0, 0.0, 1.0, 1.0])  # floats for BCE
print("bce:", F.binary_cross_entropy_with_logits(binary_logits, binary_targets).item())

print("mse:", F.mse_loss(torch.randn(4, 1), torch.randn(4, 1)).item())
# Others: nn.L1Loss, nn.SmoothL1Loss (robust regression), nn.KLDivLoss (distillation),
# F.cross_entropy(..., label_smoothing=0.1, ignore_index=-100) for padding tokens.

# torch.nn.functional (F) has stateless versions of most layers: F.relu, F.softmax,
# F.dropout(x, training=self.training), F.scaled_dot_product_attention, ...
# Use modules for things with parameters; F for pure functions.
