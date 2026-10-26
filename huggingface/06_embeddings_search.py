"""
06 - EMBEDDINGS & SEMANTIC SEARCH: the building block of RAG, clustering,
deduplication and recommendation.

What's covered:
  - sentence-transformers: encode text -> vectors, cosine similarity, top-k search
  - the same thing "by hand" with AutoModel + mean pooling (to understand / customise it)

Run:  python 06_embeddings_search.py
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer

from config import DEVICE, EMBEDDING_MODEL

documents = [
    "The Eiffel Tower is located in Paris.",
    "Transformers use self-attention to process sequences.",
    "Pasta should be cooked in salted boiling water.",
    "LoRA reduces the number of trainable parameters when fine-tuning.",
    "The Louvre is the world's most visited museum.",
]
query = "How can I fine-tune a large model cheaply?"

# --------------------------------------------------------------------------- #
# 1. High-level: sentence-transformers
# --------------------------------------------------------------------------- #
embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

doc_emb = embedder.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
query_emb = embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
print("Embedding shape:", doc_emb.shape)  # (num_docs, dim)

# Top-k semantic search (cosine similarity)
hits = util.semantic_search(query_emb, doc_emb, top_k=3)[0]
print(f"\nQuery: {query}")
for hit in hits:
    print(f"  {hit['score']:.3f}  {documents[hit['corpus_id']]}")

# For large corpora, store vectors in a vector DB / index (FAISS, Chroma,
# Qdrant, pgvector...) instead of keeping them in memory.

# --------------------------------------------------------------------------- #
# 2. Low-level: what sentence-transformers does under the hood
#    token embeddings -> mean pooling (ignoring padding) -> L2 normalise
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE).eval()


def embed(texts: list[str]) -> torch.Tensor:
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        token_embeddings = model(**batch).last_hidden_state          # (batch, seq, dim)
    mask = batch["attention_mask"].unsqueeze(-1).float()             # (batch, seq, 1)
    pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(pooled, p=2, dim=1)


manual_docs = embed(documents)
manual_query = embed([query])
scores = (manual_query @ manual_docs.T).squeeze(0)  # dot product of unit vectors = cosine
best = scores.argmax().item()
print(f"\nManual pooling best match ({scores[best]:.3f}): {documents[best]}")
