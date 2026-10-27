"""
07 - TRANSFORMER FROM SCRATCH: a tiny GPT-style language model.

Builds a decoder-only transformer from basic layers and trains it to predict
the next character of a text. The same architecture, scaled up, is how
modern LLMs work.

What's covered:
  - tokenization (character level) and next-token training data
  - token + positional embeddings
  - causal self-attention with F.scaled_dot_product_attention (fused, fast)
  - transformer blocks: pre-LayerNorm, residuals, MLP
  - weight tying between the embedding and the output layer
  - autoregressive generation with temperature and top-k sampling

Train on your own text: set TEXT_FILE=path/to/file.txt (a few hundred KB or more).

Run:  python 07_transformer_from_scratch.py
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHAR_MODEL_WEIGHTS, DEVICE, OUTPUT_DIR, seed_everything

seed_everything()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Data: characters -> integer ids. Real LLMs use subword tokenizers
#    (BPE, e.g. tiktoken or Hugging Face tokenizers), but the model is identical.
# --------------------------------------------------------------------------- #
DEFAULT_TEXT = """\
A tensor holds numbers. A module holds tensors. A loss tells the module how wrong it is,
and the gradient tells each number which way to move. The optimizer takes a small step,
and the loop begins again. Data flows forward, gradients flow backward, and slowly the
model learns the shape of the world it was shown. Batch after batch, epoch after epoch,
the loss goes down, and what was noise becomes a pattern.
""" * 20

text_file = os.getenv("TEXT_FILE")
text = open(text_file, encoding="utf-8").read() if text_file else DEFAULT_TEXT

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]


# --------------------------------------------------------------------------- #
# 2. Hyperparameters in a dataclass: easy to save next to the weights
# --------------------------------------------------------------------------- #
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64      # context length: how many previous tokens the model sees
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128         # must be divisible by n_head
    dropout: float = 0.1


cfg = GPTConfig(vocab_size=len(chars))


def get_batch(split_data: torch.Tensor, batch_size: int = 32):
    """Random windows of text. Target y is x shifted one position to the left."""
    ix = torch.randint(len(split_data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([split_data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([split_data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# --------------------------------------------------------------------------- #
# 3. The model
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # one matmul for queries, keys, values
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head attends independently
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Fused attention (FlashAttention on supported GPUs). is_causal=True masks
        # future positions so token t can only look at tokens <= t.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # merge the heads back
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # tokens exchange information
        x = x + self.mlp(self.ln2(x))      # each token is processed on its own
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # learned positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight    # weight tying: fewer params, better results

        self.apply(self._init_weights)
        for name, p in self.named_parameters():      # GPT-2 trick: scale residual projections
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
        logits = self.lm_head(self.ln_f(self.blocks(x)))        # (B, T, vocab)
        loss = None
        if targets is not None:
            # cross_entropy wants (N, classes) and (N,): flatten batch and time
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]               # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                # only the last position matters
            if top_k is not None:                                  # keep the k most likely tokens
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = TinyGPT(cfg).to(DEVICE)
print(f"vocab {cfg.vocab_size} | parameters {sum(p.numel() for p in model.parameters()):,}")

# --------------------------------------------------------------------------- #
# 4. Training: step-based (not epoch-based), the norm for language models
# --------------------------------------------------------------------------- #
MAX_STEPS = int(os.getenv("MAX_STEPS", 400))
EVAL_EVERY = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)


@torch.inference_mode()
def estimate_loss(n_batches: int = 20) -> dict[str, float]:
    model.eval()
    out = {}
    for name, split_data in (("train", train_data), ("val", val_data)):
        out[name] = sum(model(*get_batch(split_data))[1].item() for _ in range(n_batches)) / n_batches
    model.train()
    return out


model.train()
for step in range(MAX_STEPS + 1):
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        print(f"step {step:4d} | train {losses['train']:.3f} | val {losses['val']:.3f}")
    if step == MAX_STEPS:
        break
    xb, yb = get_batch(train_data)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

# Save weights together with the config and vocabulary needed to rebuild the model.
torch.save({"model": model.state_dict(), "config": vars(cfg), "chars": chars}, CHAR_MODEL_WEIGHTS)

# --------------------------------------------------------------------------- #
# 5. Generation
# --------------------------------------------------------------------------- #
model.eval()
prompt = torch.tensor([encode(text[:8])], device=DEVICE)    # any string made of known characters
for temperature in (0.5, 1.0):
    out = model.generate(prompt, max_new_tokens=150, temperature=temperature, top_k=10)
    print(f"\n--- temperature {temperature} ---\n{decode(out[0].tolist())}")
