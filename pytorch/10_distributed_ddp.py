"""
10 - DISTRIBUTED TRAINING with DistributedDataParallel (DDP).

DDP runs one process per GPU. Each process holds a full copy of the model,
trains on a different slice of the data, and gradients are averaged across
processes during backward(). It is the standard way to use several GPUs
(or several machines).

What's covered:
  - launching with torchrun and reading the rank / world size
  - init_process_group (nccl for GPUs, gloo for CPU)
  - DistributedSampler and set_epoch
  - wrapping the model in DDP
  - saving and printing only on rank 0
  - all_reduce to aggregate metrics across processes

Run (works on CPU too, using 2 processes):
  torchrun --nproc_per_node=2 10_distributed_ddp.py
  torchrun --nproc_per_node=gpu 10_distributed_ddp.py      # one process per GPU
Multi-node: add --nnodes=2 --node_rank=0|1 --rdzv_endpoint=<host>:29500

When a single model no longer fits on one GPU, look at FSDP2
(torch.distributed.fsdp.fully_shard), which shards the parameters themselves.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from config import OUTPUT_DIR, SEED, seed_everything


def main():
    # ----------------------------------------------------------------------- #
    # 1. Process group. torchrun sets RANK, LOCAL_RANK and WORLD_SIZE for us.
    # ----------------------------------------------------------------------- #
    rank = int(os.environ["RANK"])               # global id of this process
    local_rank = int(os.environ["LOCAL_RANK"])   # id on this machine -> which GPU to use
    world_size = int(os.environ["WORLD_SIZE"])   # total number of processes

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    is_main = rank == 0

    seed_everything(SEED)   # same seed everywhere -> identical initial weights (DDP also broadcasts them)

    # ----------------------------------------------------------------------- #
    # 2. Data: DistributedSampler gives each process a different, non-overlapping shard.
    # ----------------------------------------------------------------------- #
    g = torch.Generator().manual_seed(SEED)
    X = torch.randn(4096, 32, generator=g)
    y = (X[:, :4].sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED)
    # Note: shuffle goes to the sampler, not to the DataLoader.
    # batch_size is PER PROCESS: the global batch is batch_size * world_size.
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # ----------------------------------------------------------------------- #
    # 3. Model: move to the device, THEN wrap in DDP.
    # ----------------------------------------------------------------------- #
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 2)).to(device)
    # Convert BatchNorm layers to sync statistics across GPUs, if your model has them:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # A common rule: scale the learning rate with the global batch size.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 * world_size)

    # ----------------------------------------------------------------------- #
    # 4. Training loop: identical to a single-device loop, plus set_epoch.
    # ----------------------------------------------------------------------- #
    for epoch in range(3):
        sampler.set_epoch(epoch)        # otherwise every epoch uses the same shuffle order
        model.train()
        loss_sum = torch.zeros(1, device=device)
        correct = torch.zeros(1, device=device)
        count = torch.zeros(1, device=device)

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()             # DDP all-reduces (averages) gradients here
            optimizer.step()
            loss_sum += loss.detach() * len(xb)
            correct += (logits.argmax(1) == yb).sum()
            count += len(xb)

        # Each process only saw its shard: sum the metrics across all processes.
        for t in (loss_sum, correct, count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if is_main:
            print(f"epoch {epoch} | loss {(loss_sum / count).item():.4f} | "
                  f"acc {(correct / count).item():.3f} | {int(count.item())} samples on {world_size} processes")

    # ----------------------------------------------------------------------- #
    # 5. Save from rank 0 only. model.module is the original, unwrapped model:
    #    saving it avoids a "module." prefix on every key.
    # ----------------------------------------------------------------------- #
    if is_main:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(OUTPUT_DIR, "ddp_model.pt"))
        print("saved", os.path.join(OUTPUT_DIR, "ddp_model.pt"))
    dist.barrier()                      # wait until the file is written before anyone exits
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
