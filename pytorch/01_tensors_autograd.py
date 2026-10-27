"""
01 - TENSORS & AUTOGRAD: the two foundations everything else is built on.

A tensor is an n-dimensional array that can live on a GPU.
Autograd records the operations on tensors and computes gradients for you.

What's covered:
  - creating tensors, dtypes, shapes, devices
  - indexing, reshaping, broadcasting, matrix multiplication
  - NumPy interop
  - autograd: requires_grad, backward(), .grad, no_grad, detach
  - gradient descent by hand (what an optimizer does for you later)

Run:  python 01_tensors_autograd.py
"""

import numpy as np
import torch

from config import DEVICE, seed_everything

seed_everything()

# --------------------------------------------------------------------------- #
# 1. Creating tensors
# --------------------------------------------------------------------------- #
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # from Python data (copies)
zeros = torch.zeros(2, 3)                     # also: ones, full, empty, eye
rand = torch.rand(2, 3)                       # uniform [0, 1); randn = normal
steps = torch.arange(0, 10, 2)                # tensor([0, 2, 4, 6, 8])
like = torch.zeros_like(a)                    # same shape, dtype and device as `a`

print("shape:", a.shape, "dtype:", a.dtype, "device:", a.device)

# dtypes: float32 is the default for training; int64 ("long") for class labels and indices.
labels = torch.tensor([0, 2, 1])              # int64
half = a.to(torch.bfloat16)                   # cast; .float(), .long(), .bool() are shortcuts

# --------------------------------------------------------------------------- #
# 2. Devices: tensors must be on the SAME device to interact
# --------------------------------------------------------------------------- #
x = torch.randn(3, 4, device=DEVICE)          # create directly on the device (fastest)
y = a.to(DEVICE)                              # or move an existing tensor
back_on_cpu = x.cpu()                         # .numpy() and printing large tensors need CPU

# --------------------------------------------------------------------------- #
# 3. Indexing and reshaping (same rules as NumPy)
# --------------------------------------------------------------------------- #
m = torch.arange(12).reshape(3, 4)
print(m[0])          # first row
print(m[:, 1])       # second column
print(m[m > 5])      # boolean mask -> 1D tensor
print(m.T.shape)     # transpose -> (4, 3)

flat = m.flatten()               # (12,)
col = m.reshape(-1, 1)           # -1 means "infer this dimension" -> (12, 1)
batch = m.unsqueeze(0)           # add a dimension at 0 -> (1, 3, 4); squeeze removes size-1 dims
stacked = torch.stack([m, m])    # new dimension -> (2, 3, 4)
joined = torch.cat([m, m], dim=1)  # along an existing dimension -> (3, 8)
# .view() is like .reshape() but never copies; it fails on non-contiguous tensors.

# --------------------------------------------------------------------------- #
# 4. Math: element-wise ops, reductions, broadcasting, matmul
# --------------------------------------------------------------------------- #
b = torch.tensor([10.0, 20.0])
print(a + b)                  # broadcasting: b (2,) is stretched across each row of a (2, 2)
print(a * a)                  # element-wise
print(a @ a)                  # matrix multiplication (same as torch.matmul)
print(a.sum(), a.mean(dim=0), a.max(dim=1).values)
print(a.softmax(dim=-1))      # most functions exist both as torch.fn(t) and t.fn()

# Batched matmul: (batch, n, k) @ (batch, k, m) -> (batch, n, m). Used everywhere in attention.
q = torch.randn(8, 5, 16)
k = torch.randn(8, 5, 16)
scores = q @ k.transpose(-2, -1)   # (8, 5, 5)

# In-place ops end with "_" (save memory, but avoid them on tensors autograd needs).
c = torch.ones(3)
c.add_(1)

# --------------------------------------------------------------------------- #
# 5. NumPy interop (CPU tensors share memory with the array: no copy)
# --------------------------------------------------------------------------- #
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)        # shares memory
arr[0] = 100.0
print(t)                         # tensor([100., 2., 3.]) - changed too
print(t.numpy())                 # back to NumPy; use t.detach().cpu().numpy() in general

# --------------------------------------------------------------------------- #
# 6. Autograd: PyTorch records operations and differentiates them for you
# --------------------------------------------------------------------------- #
w = torch.tensor(3.0, requires_grad=True)   # "track gradients for this tensor"
y = w ** 2 + 2 * w                          # y = w^2 + 2w
y.backward()                                # computes dy/dw and stores it in w.grad
print("dy/dw at w=3:", w.grad)              # 2w + 2 = 8

# Gradients ACCUMULATE: call backward again and w.grad adds up. Reset it between steps.
w.grad = None                               # optimizers do this in optimizer.zero_grad()

# no_grad: turn off tracking for inference or manual weight updates (faster, less memory).
with torch.no_grad():
    z = w * 2
print("tracked inside no_grad?", z.requires_grad)   # False

# detach: a copy of the value cut out of the graph (e.g. for logging a loss).
loss_value = y.detach().item()                       # .item() -> Python number (scalar tensors only)

# --------------------------------------------------------------------------- #
# 7. Gradient descent by hand: fit y = 2x + 1
#    Scripts 03 and 04 replace this with nn.Module + an optimizer.
# --------------------------------------------------------------------------- #
xs = torch.linspace(-1, 1, 100).unsqueeze(1)          # (100, 1)
ys = 2 * xs + 1 + 0.1 * torch.randn_like(xs)          # noisy targets

weight = torch.randn(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
lr = 0.1

for step in range(200):
    pred = xs * weight + bias
    loss = ((pred - ys) ** 2).mean()                  # mean squared error
    loss.backward()                                   # fills weight.grad and bias.grad
    with torch.no_grad():                             # the update itself must not be tracked
        weight -= lr * weight.grad
        bias -= lr * bias.grad
    weight.grad = None
    bias.grad = None

print(f"learned: y = {weight.item():.2f}x + {bias.item():.2f}  (target: y = 2x + 1)")
