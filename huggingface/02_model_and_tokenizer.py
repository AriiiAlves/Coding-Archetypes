"""
02 - MODEL + TOKENIZER: full control over loading, tokenizing and generating.

What's covered:
  - AutoTokenizer / AutoModelForCausalLM (the "Auto" classes pick the right
    architecture from the model's config.json)
  - tokenization basics (ids, attention mask, decode)
  - chat templates
  - generation parameters & GenerationConfig
  - token streaming
  - optional 4-bit quantization (bitsandbytes, NVIDIA GPU only)

Run:  python 02_model_and_tokenizer.py
"""

from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextIteratorStreamer,
)

from config import CHAT_MODEL, DEVICE, DTYPE

USE_4BIT = False  # set True on an NVIDIA GPU with `bitsandbytes` installed

# --------------------------------------------------------------------------- #
# 1. Load tokenizer
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)

# Many decoder models ship without a pad token; reuse EOS so batching works.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenization round-trip
enc = tokenizer("Hello Hugging Face!", return_tensors="pt")
print("input_ids:     ", enc["input_ids"])
print("attention_mask:", enc["attention_mask"])
print("tokens:        ", tokenizer.convert_ids_to_tokens(enc["input_ids"][0]))
print("decoded:       ", tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True))

# --------------------------------------------------------------------------- #
# 2. Load model
# --------------------------------------------------------------------------- #
if USE_4BIT:
    # QLoRA-style 4-bit loading: ~4x less memory, slight quality loss.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL,
        quantization_config=quant_config,
        device_map="auto",  # let accelerate spread layers over available devices
    )
else:
    model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=DTYPE).to(DEVICE)

model.eval()  # disable dropout for inference
print(f"\nLoaded {CHAT_MODEL}: {model.num_parameters() / 1e6:.1f}M params on {model.device}")

# --------------------------------------------------------------------------- #
# 3. Chat template: turns a list of messages into the exact prompt format the
#    model was trained on (special tokens, role markers, ...).
# --------------------------------------------------------------------------- #
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Give me 3 tips for writing clean Python."},
]
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # append the "assistant:" marker so the model answers
)
print("\n--- Rendered prompt ---\n" + prompt_text)

inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

# --------------------------------------------------------------------------- #
# 4. Generate
# --------------------------------------------------------------------------- #
gen_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,         # <1 sharper, >1 more random
    top_p=0.9,               # nucleus sampling
    top_k=50,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

with torch.no_grad():
    output_ids = model.generate(**inputs, generation_config=gen_config)

# Strip the prompt tokens so we only decode the newly generated part.
new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
print("\n--- Reply ---\n" + tokenizer.decode(new_tokens, skip_special_tokens=True))

# --------------------------------------------------------------------------- #
# 5. Streaming: print tokens as they are produced (for chat UIs / CLIs)
# --------------------------------------------------------------------------- #
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
thread = Thread(
    target=model.generate,
    kwargs=dict(**inputs, generation_config=gen_config, streamer=streamer),
)
thread.start()
print("\n--- Streaming reply ---")
for chunk in streamer:
    print(chunk, end="", flush=True)
thread.join()
print()

# --------------------------------------------------------------------------- #
# 6. Batched generation: left padding is required for decoder-only models
# --------------------------------------------------------------------------- #
tokenizer.padding_side = "left"
prompts = [
    tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    for q in ["What is 2+2?", "Name a primary color."]
]
batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    out = model.generate(**batch, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.pad_token_id)
for row in out:
    print("->", tokenizer.decode(row[batch["input_ids"].shape[1]:], skip_special_tokens=True))
