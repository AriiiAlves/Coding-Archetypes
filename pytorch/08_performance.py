"""
08 - PERFORMANCE: train faster and fit bigger models in the same memory.

Each technique is shown on its own, with a small timing benchmark, so you can
see what it does on your hardware. Most big wins appear on a GPU; on a CPU some
of them are neutral or slower, and that is expected.

What's covered:
  - mixed precision: torch.autocast (+ GradScaler for float16)
  - gradient accumulation: large effective batches on small memory
  - activation checkpointing: trade compute for memory
  - torch.compile: graph compilation for speed
  - channels_last memory format for CNNs, TF32 matmuls
  - the PyTorch profiler: find where time actually goes

Run:  python 08_performance.py
"""

import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.checkpoint import checkpoint

from config import AMP_DTYPE, DEVICE, USE_AMP, seed_everything

seed_everything()

# On Ampere+ NVIDIA GPUs, let float32 matmuls use TF32 tensor cores (big speedup, tiny precision cost).
torch.set_float32_matmul_precision("high")


class DeepMLP(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 6, use_checkpointing: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
            for _ in range(depth)
        )
        self.head = nn.Linear(dim, 10)
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                # Don't store this block's activations; recompute them during backward.
                # Saves memory roughly proportional to depth, costs ~30% more compute.
                x = x + checkpoint(block, x, use_reentrant=False)
            else:
                x = x + block(x)
        return self.head(x)


X = torch.randn(128, 256, device=DEVICE)
Y = torch.randint(0, 10, (128,), device=DEVICE)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()        # GPU work is asynchronous: wait before reading the clock


def benchmark(name: str, step_fn, n_steps: int = 20) -> None:
    for _ in range(3):                  # warmup (first steps include allocation / compilation)
        step_fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        step_fn()
    sync()
    print(f"{name:32s} {(time.perf_counter() - t0) / n_steps * 1000:7.1f} ms/step")


def make_step(model, use_amp=False, scaler=None):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    amp_ctx = torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE) if use_amp else nullcontext()

    def step():
        with amp_ctx:                                   # forward + loss in low precision
            loss = F.cross_entropy(model(X), Y)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()               # backward happens OUTSIDE autocast
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    return step


# --------------------------------------------------------------------------- #
# 1. Baseline vs. mixed precision.
#    autocast runs matmuls/convs in bfloat16/float16 and keeps sensitive ops
#    (softmax, losses, norms) in float32. bfloat16 needs no scaler; float16 needs
#    a GradScaler so small gradients don't underflow to zero.
# --------------------------------------------------------------------------- #
benchmark("float32 baseline", make_step(DeepMLP().to(DEVICE)))

if USE_AMP:
    scaler = torch.amp.GradScaler(DEVICE.type) if AMP_DTYPE == torch.float16 else None
    benchmark(f"autocast {str(AMP_DTYPE).split('.')[-1]}", make_step(DeepMLP().to(DEVICE), use_amp=True, scaler=scaler))
else:
    print("autocast skipped: no GPU (set FORCE_AMP=1 to try bfloat16 on this CPU)")

# --------------------------------------------------------------------------- #
# 2. Activation checkpointing (memory saver; slower per step by design)
# --------------------------------------------------------------------------- #
benchmark("activation checkpointing", make_step(DeepMLP(use_checkpointing=True).to(DEVICE)))

# --------------------------------------------------------------------------- #
# 3. torch.compile: traces the model into an optimized graph (fused kernels).
#    First call is slow (compiling); later calls are faster. Needs a C++ compiler
#    on CPU. mode="reduce-overhead" uses CUDA graphs for small batches;
#    mode="max-autotune" searches for the fastest kernels (long compile).
# --------------------------------------------------------------------------- #
try:
    compiled = torch.compile(DeepMLP().to(DEVICE))
    benchmark("torch.compile", make_step(compiled))
except Exception as e:  # missing compiler, unsupported platform, ...
    print(f"torch.compile unavailable here: {type(e).__name__}: {str(e)[:80]}")

# --------------------------------------------------------------------------- #
# 4. Gradient accumulation: effective batch = micro_batch * accum_steps.
#    Use it when the batch you want doesn't fit in memory.
# --------------------------------------------------------------------------- #
model = DeepMLP().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
ACCUM_STEPS = 4
micro_batches = zip(X.chunk(ACCUM_STEPS), Y.chunk(ACCUM_STEPS))

opt.zero_grad(set_to_none=True)
for i, (xb, yb) in enumerate(micro_batches):
    loss = F.cross_entropy(model(xb), yb) / ACCUM_STEPS   # divide so gradients average, not sum
    loss.backward()                                      # gradients accumulate in .grad
    if (i + 1) % ACCUM_STEPS == 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
print(f"gradient accumulation: {ACCUM_STEPS} micro-batches of {len(X) // ACCUM_STEPS} = one step of {len(X)}")

# --------------------------------------------------------------------------- #
# 5. channels_last: NHWC memory layout, faster convolutions on GPUs with tensor cores
# --------------------------------------------------------------------------- #
cnn = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1)).to(DEVICE)
cnn = cnn.to(memory_format=torch.channels_last)
images = torch.randn(32, 3, 64, 64, device=DEVICE).to(memory_format=torch.channels_last)
print("channels_last output:", cnn(images).is_contiguous(memory_format=torch.channels_last))

# --------------------------------------------------------------------------- #
# 6. Profiler: measure before optimizing. Shows the most expensive ops.
#    Export a timeline with prof.export_chrome_trace("trace.json") and open it
#    in https://ui.perfetto.dev
# --------------------------------------------------------------------------- #
activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE.type == "cuda" else [])
step = make_step(DeepMLP().to(DEVICE))
with profile(activities=activities, record_shapes=True) as prof:
    with record_function("train_step"):                  # custom label in the trace
        step()
sort_key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=8))

# --------------------------------------------------------------------------- #
# 7. GPU memory inspection (CUDA only)
# --------------------------------------------------------------------------- #
if DEVICE.type == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    torch.cuda.reset_peak_memory_stats()
