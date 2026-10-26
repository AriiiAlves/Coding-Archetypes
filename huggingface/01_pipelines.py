"""
01 - PIPELINES: the fastest way to use a pretrained model.

`pipeline()` bundles tokenizer + model + pre/post-processing behind one call.
Use it for prototyping or whenever you don't need control over the internals.

Run:  python 01_pipelines.py
"""

from transformers import pipeline

from config import CHAT_MODEL, DEVICE

# --------------------------------------------------------------------------- #
# 1. Task with a default model (HF picks a sensible model for the task)
#    Common tasks: "sentiment-analysis", "text-classification", "ner",
#    "question-answering", "summarization", "translation", "fill-mask",
#    "zero-shot-classification", "text-generation", "feature-extraction",
#    "image-classification", "automatic-speech-recognition", ...
# --------------------------------------------------------------------------- #
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",  # pin a model for reproducibility
    device=DEVICE,
)
print(classifier(["I love open models!", "This bug ruined my day."]))
# -> [{'label': 'POSITIVE', 'score': 0.99}, {'label': 'NEGATIVE', 'score': 0.99}]

# --------------------------------------------------------------------------- #
# 2. Zero-shot classification: classify into labels the model never trained on
# --------------------------------------------------------------------------- #
zero_shot = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=DEVICE)
print(zero_shot(
    "The new GPU doubles training throughput.",
    candidate_labels=["technology", "sports", "politics"],
))

# --------------------------------------------------------------------------- #
# 3. Text generation with a chat model.
#    Passing a list of {"role", "content"} messages makes the pipeline apply
#    the model's chat template automatically.
# --------------------------------------------------------------------------- #
generator = pipeline("text-generation", model=CHAT_MODEL, device=DEVICE)

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Explain what a tokenizer does in one sentence."},
]
out = generator(
    messages,
    max_new_tokens=60,
    do_sample=True,     # sampling -> more varied output; False = greedy
    temperature=0.7,
    top_p=0.9,
)
# The pipeline returns the full conversation; the last message is the reply.
print(out[0]["generated_text"][-1]["content"])

# --------------------------------------------------------------------------- #
# 4. Batching: pass a list and set batch_size for throughput
# --------------------------------------------------------------------------- #
texts = ["Great movie."] * 8
print(classifier(texts, batch_size=4))
