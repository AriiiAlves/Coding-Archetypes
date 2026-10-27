"""
02 - DATASETS & DATALOADERS: feed data to a model efficiently.

Dataset   = "how do I get sample i?"            (__len__ + __getitem__)
DataLoader = "batch, shuffle and load in parallel" (wraps any Dataset)

What's covered:
  - a custom map-style Dataset (the pattern for CSVs, folders of files, databases)
  - TensorDataset for data already in memory
  - random_split, Subset
  - DataLoader options that matter: batch_size, shuffle, num_workers, pin_memory, drop_last
  - a custom collate_fn for variable-length sequences (padding)
  - an IterableDataset for streams too big for memory
  - torchvision datasets and transforms

Run:  python 02_datasets_dataloaders.py
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import v2

from config import BATCH_SIZE, DATA_DIR, DEVICE, NUM_WORKERS, SEED, seed_everything

seed_everything()


# --------------------------------------------------------------------------- #
# 1. A custom Dataset. Replace the body of __init__/__getitem__ with your
#    own loading logic: read a CSV row, open an image file, query a DB, ...
#    Keep __init__ cheap (store paths/indices) and do the heavy work lazily
#    in __getitem__, so DataLoader workers can parallelize it.
# --------------------------------------------------------------------------- #
class SyntheticRegression(Dataset):
    """y = sum(x) + noise. Stands in for any tabular dataset."""

    def __init__(self, n_samples: int = 1000, n_features: int = 8):
        g = torch.Generator().manual_seed(SEED)
        self.x = torch.randn(n_samples, n_features, generator=g)
        self.y = self.x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1, generator=g)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        # Return tensors (or a dict / tuple of tensors). The default collate_fn
        # stacks them into batches with a new leading batch dimension.
        return self.x[idx], self.y[idx]


ds = SyntheticRegression()
x0, y0 = ds[0]
print("one sample:", x0.shape, y0.shape)

# Already have tensors in memory? TensorDataset does the same thing in one line.
same_ds = TensorDataset(ds.x, ds.y)

# --------------------------------------------------------------------------- #
# 2. Train / validation split
# --------------------------------------------------------------------------- #
train_ds, val_ds = random_split(ds, [0.8, 0.2], generator=torch.Generator().manual_seed(SEED))
first_100 = Subset(ds, range(100))                     # any index list works
print("split sizes:", len(train_ds), len(val_ds), len(first_100))

# --------------------------------------------------------------------------- #
# 3. DataLoader: the options you will actually tune
# --------------------------------------------------------------------------- #
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,                         # shuffle training data every epoch; never the eval set
    num_workers=NUM_WORKERS,              # parallel loading processes; 0 = main process (easy debugging)
    pin_memory=DEVICE.type == "cuda",     # faster CPU -> GPU copies
    drop_last=True,                       # skip the last incomplete batch (stable BatchNorm statistics)
    persistent_workers=NUM_WORKERS > 0,   # keep workers alive between epochs
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

xb, yb = next(iter(train_loader))
print("batch:", xb.shape, yb.shape)       # (64, 8) (64, 1)

for xb, yb in train_loader:               # the standard loop shape used in every training script
    xb = xb.to(DEVICE, non_blocking=True)  # non_blocking pairs with pin_memory
    yb = yb.to(DEVICE, non_blocking=True)
    break


# --------------------------------------------------------------------------- #
# 4. Variable-length samples need a custom collate_fn (text, audio, time series).
#    The default collate can only stack tensors of identical shape.
# --------------------------------------------------------------------------- #
class VariableLengthSequences(Dataset):
    def __init__(self, n: int = 100):
        g = torch.Generator().manual_seed(SEED)
        lengths = torch.randint(3, 12, (n,), generator=g)
        self.seqs = [torch.randint(1, 50, (int(n_tok),), generator=g) for n_tok in lengths]
        self.labels = torch.randint(0, 2, (n,), generator=g)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


PAD_ID = 0


def pad_collate(batch):
    """Pad every sequence to the longest in the batch and return a mask."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)   # (B, max_len)
    mask = padded != PAD_ID                                               # True = real token
    return {"input_ids": padded, "mask": mask, "lengths": lengths, "labels": torch.stack(labels)}


seq_loader = DataLoader(VariableLengthSequences(), batch_size=4, shuffle=True, collate_fn=pad_collate)
batch = next(iter(seq_loader))
print("padded batch:", batch["input_ids"].shape, "lengths:", batch["lengths"].tolist())


# --------------------------------------------------------------------------- #
# 5. IterableDataset: for streams (huge files, network sources, generators).
#    No random access, so shuffle=True is not allowed; shuffle inside __iter__.
#    With num_workers > 0, split the stream between workers via get_worker_info().
# --------------------------------------------------------------------------- #
class CountingStream(IterableDataset):
    def __init__(self, limit: int):
        self.limit = limit

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        start, step = (0, 1) if info is None else (info.id, info.num_workers)
        for i in range(start, self.limit, step):
            yield torch.tensor([float(i)])


print("stream batch:", next(iter(DataLoader(CountingStream(20), batch_size=5))).flatten().tolist())

# --------------------------------------------------------------------------- #
# 6. torchvision: ready-made datasets and transforms (transforms.v2 is current)
#    Other built-ins: CIFAR10, MNIST, ImageNet, VOC/COCO detection, ...
#    For your own images: datasets.ImageFolder("path/") with one sub-folder per class.
# --------------------------------------------------------------------------- #
transform = v2.Compose([
    v2.ToImage(),                                   # PIL image -> tensor image (C, H, W), uint8
    v2.ToDtype(torch.float32, scale=True),          # uint8 [0, 255] -> float [0, 1]
    v2.Normalize(mean=[0.2860], std=[0.3530]),      # FashionMNIST statistics
])
fashion = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
img, label = fashion[0]
print("FashionMNIST:", len(fashion), "images,", img.shape, "label:", fashion.classes[label])
