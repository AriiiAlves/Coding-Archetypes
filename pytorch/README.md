# PyTorch Archetype: Deep Learning from Tensors to Deployment

A starter kit for building and training neural networks with **PyTorch**.
Each script is standalone, heavily commented and runnable on a laptop CPU (the defaults are small models, a few epochs and data subsets).
Copy the script closest to your goal and grow it.

## Setup

```bash
cd pytorch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # GPU users: install torch + torchvision from https://pytorch.org/get-started first
```

All paths, hyperparameters, hardware detection and seeding live in **`config.py`**. Change settings there.

## The scripts (in learning order)

| # | File | What you learn | Key tools |
|---|------|----------------|-----------|
| 01 | `01_tensors_autograd.py` | Tensors, dtypes, devices, reshaping, broadcasting, NumPy interop, autograd, gradient descent by hand | `torch.tensor`, `.backward()`, `no_grad` |
| 02 | `02_datasets_dataloaders.py` | Custom datasets, splits, batching, parallel loading, padding variable-length data, streaming, image transforms | `Dataset`, `DataLoader`, `collate_fn`, `IterableDataset`, `torchvision.transforms.v2` |
| 03 | `03_nn_module.py` | Building models: core layers, custom modules, residual blocks, multi-head outputs, buffers, freezing, init, train/eval mode, loss functions | `nn.Module`, `nn.Sequential`, `nn.ModuleList`, `torch.nn.functional` |
| 04 | `04_training_loop.py` | **The reusable training loop**: AdamW, warmup + cosine LR, gradient clipping, validation, checkpointing, resuming, early stopping | `torch.optim`, `lr_scheduler`, `torch.save` / `torch.load` |
| 05 | `05_cnn_image_classifier.py` | A CNN from scratch on FashionMNIST: augmentation, label smoothing, OneCycleLR, confusion matrix | `Conv2d`, `BatchNorm2d`, `OneCycleLR` |
| 06 | `06_transfer_learning.py` | **Fine-tuning a pretrained ResNet**: replace the head, linear probe, then partial fine-tuning with per-layer LRs, feature extraction | `torchvision.models`, weights API |
| 07 | `07_transformer_from_scratch.py` | **A tiny GPT**: embeddings, causal self-attention, transformer blocks, weight tying, text generation with temperature / top-k | `F.scaled_dot_product_attention` |
| 08 | `08_performance.py` | Mixed precision, gradient accumulation, activation checkpointing, `torch.compile`, channels_last, profiling | `torch.autocast`, `GradScaler`, `torch.compile`, `torch.profiler` |
| 09 | `09_save_export.py` | Saving and loading weights safely, exporting for deployment, ONNX, smaller bfloat16 weights | `state_dict`, `torch.export`, `torch.onnx` |
| 10 | `10_distributed_ddp.py` | **Multi-GPU / multi-node training**: process groups, sharded sampling, metric aggregation | `torchrun`, `DistributedDataParallel`, `DistributedSampler` |

Run any script with `python 0X_name.py` (script 10: `torchrun --nproc_per_node=2 10_distributed_ddp.py`).
Datasets download to `data/`, and checkpoints and exports go to `outputs/`. Both are git-ignored.
Scripts 02 and 05 download FashionMNIST (~30 MB). Script 06 downloads CIFAR-10 (~170 MB) and ResNet-18 weights (~45 MB).

Useful environment variables:

| Variable | Effect |
|----------|--------|
| `EPOCHS` / `BATCH_SIZE` | Training length and batch size for scripts 05 and 06 (defaults are 3 and 64) |
| `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` | Dataset subset sizes in script 05 (defaults are 10000 and 2000; `0` means the full dataset) |
| `NUM_WORKERS` | DataLoader worker processes (default is 2 on Linux, 0 on macOS and Windows; use `0` while debugging) |
| `TEXT_FILE` / `MAX_STEPS` | Train script 07 on your own text file, and for how many steps |
| `FORCE_AMP=1` | Try bfloat16 autocast on a CPU in script 08 |
| `TORCH_DATA_DIR` / `TORCH_OUTPUT_DIR` | Where datasets and outputs are written |

## Which approach should I use?

| Goal | Approach | Script |
|------|----------|--------|
| Tabular data or a custom prediction problem | MLP with the standard training loop | 03, 04 |
| Images with a small or medium dataset | Fine-tune a pretrained model | 06 |
| Images that look nothing like photos (medical, satellite, sensor data) | Train a CNN from scratch, or fine-tune with more layers unfrozen | 05, 06 |
| Sequences, text or time series | Transformer encoder/decoder | 07 |
| Training is slow or runs out of memory | Mixed precision, accumulation, checkpointing, compile | 08 |
| Shipping the model to production | `torch.export` or ONNX | 09 |
| More than one GPU | DDP (FSDP2 when the model doesn't fit on one GPU) | 10 |
| Pretrained language or vision transformers | Use the [`huggingface/`](../huggingface/) archetype | n/a |

## Scaling up

- **GPU**: everything moves to `cuda` or `mps` automatically through `config.DEVICE`. Raise `BATCH_SIZE` until GPU memory is nearly full, then scale the learning rate with it.
- **Mixed precision**: wrap forward + loss in `torch.autocast` (script 08). Use bfloat16 on Ampere or newer GPUs; float16 + `GradScaler` on older ones.
- **Data loading**: if the GPU sits idle, raise `NUM_WORKERS`, set `pin_memory=True`, and move heavy augmentation to the GPU.
- **Experiment tracking**: `from torch.utils.tensorboard import SummaryWriter`, or Weights & Biases / MLflow.
- **Less boilerplate**: [PyTorch Lightning](https://lightning.ai/docs/pytorch) or [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) wrap the loop from script 04 and add multi-GPU, mixed precision and logging with a few flags.
- **Huge models**: FSDP2 (`torch.distributed.fsdp.fully_shard`) shards parameters, gradients and optimizer state across GPUs.
- **Serving**: export with script 09, then serve with Triton Inference Server, ONNX Runtime, or a FastAPI app that loads the `state_dict`.

## Common pitfalls

- **Forgetting `model.eval()`** before validation or inference. Dropout stays on and BatchNorm uses batch statistics, so results are noisy and wrong.
- **Forgetting `optimizer.zero_grad()`**: gradients accumulate across steps by design.
- **Device mismatch** (`Expected all tensors to be on the same device`): move both the model and every batch to `DEVICE`.
- **Loss input shapes**: `cross_entropy` takes raw logits `(N, C)` and int64 targets `(N,)`. Don't apply softmax first. `BCEWithLogits` takes float targets of the same shape as the logits.
- **Calling `.item()` or `print(tensor)` every step on GPU** forces a sync and slows training. Accumulate on the device and read values once per epoch.
- **Different preprocessing** in training and inference, especially normalization stats for pretrained models: use `weights.transforms()`.
- **`torch.load` on untrusted files**: keep `weights_only=True` (the default). Old tutorials pass `weights_only=False`, which can run arbitrary code.
- **Saving `torch.save(model)`** instead of `model.state_dict()`: the pickle breaks when you rename or move the class.
- **DataLoader workers on macOS/Windows**: these platforms start workers by re-importing the script, so with `num_workers > 0` the script body must be inside `if __name__ == "__main__":`. The scripts here default to `NUM_WORKERS=0` there for that reason.
- **Mixed precision on a CPU** is usually slower, not faster, unless the CPU has native bfloat16 support (recent Xeon / AMX).
- **Library versions**: this archetype was tested with `torch` 2.13 and `torchvision` 0.28. Older tutorials use deprecated APIs: `pretrained=True` (now `weights=...`), `torchvision.transforms` (now `transforms.v2`), `torch.cuda.amp.autocast` (now `torch.autocast("cuda")`), `torch.jit.trace`/`script` (now `torch.export`), and `torch.ao.quantization` (now the `torchao` library).
