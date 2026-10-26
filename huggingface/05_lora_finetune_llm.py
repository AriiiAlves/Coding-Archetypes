"""
05 - LoRA FINE-TUNING of an open LLM with PEFT (parameter-efficient fine-tuning).

Instead of updating all weights, LoRA injects small trainable low-rank matrices
into chosen layers. Typically <1% of parameters are trained, so:
  - it fits on consumer GPUs (with 4-bit loading = "QLoRA", even 7B+ models)
  - the result is a tiny adapter (MBs) you can share / swap / merge

Pipeline:
  instruction dataset -> chat template -> tokenize (mask prompt tokens)
  -> base model (+ optional 4-bit) -> LoraConfig -> get_peft_model -> Trainer
  -> save adapter -> reload & generate

Tip: TRL's `SFTTrainer` (pip install trl) wraps this same flow in fewer lines and
adds packing, DPO/GRPO preference tuning, etc. This file uses plain Trainer so
every step stays visible.

Run:  python 05_lora_finetune_llm.py
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from config import CHAT_MODEL, DEVICE, DTYPE, INSTRUCTION_DATASET, LORA_OUTPUT, MAX_TRAIN_SAMPLES, SEED

USE_QLORA = False   # True = load base model in 4-bit (NVIDIA GPU + bitsandbytes)
MAX_LENGTH = 512

# --------------------------------------------------------------------------- #
# 1. Tokenizer
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------------------------------- #
# 2. Data: convert each row into chat messages, then tokenize.
#    We set labels = -100 on the prompt tokens so the loss is computed ONLY on
#    the assistant's answer (the model learns to answer, not to echo prompts).
# --------------------------------------------------------------------------- #
raw = load_dataset(INSTRUCTION_DATASET, split="train").shuffle(seed=SEED).select(range(MAX_TRAIN_SAMPLES))


def to_messages(row):
    user = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
    return [{"role": "user", "content": user}], [{"role": "assistant", "content": row["output"]}]


def tokenize(row):
    prompt_msgs, answer_msgs = to_messages(row)
    prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(prompt_msgs + answer_msgs, tokenize=True)
    # Newer tokenizers may return a BatchEncoding instead of a plain list
    if not isinstance(prompt_ids, list):
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]

    full_ids = full_ids[:MAX_LENGTH]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels[:MAX_LENGTH],
    }


train_ds = raw.map(tokenize, remove_columns=raw.column_names)
# Drop rows where the answer got fully truncated away (nothing to learn from)
train_ds = train_ds.filter(lambda row: any(label != -100 for label in row["labels"]))

# Pads input_ids AND labels (labels padded with -100) per batch
collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

# --------------------------------------------------------------------------- #
# 3. Base model (optionally 4-bit quantized = QLoRA)
# --------------------------------------------------------------------------- #
if USE_QLORA:
    model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)  # stabilises training on quantized weights
else:
    model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=DTYPE).to(DEVICE)

model.config.use_cache = False  # KV-cache is useless during training & conflicts with checkpointing

# --------------------------------------------------------------------------- #
# 4. LoRA config - the main knobs
# --------------------------------------------------------------------------- #
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                 # rank: capacity of the adapter (8-64 typical)
    lora_alpha=32,        # scaling; rule of thumb alpha = 2*r
    lora_dropout=0.05,
    # Which linear layers get adapters. Names depend on architecture
    # (print(model) to see them). "all-linear" targets every linear layer.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # e.g. "trainable params: 4.9M || all params: 139M || trainable%: 3.5"

# --------------------------------------------------------------------------- #
# 5. Train
# --------------------------------------------------------------------------- #
args = TrainingArguments(
    output_dir=LORA_OUTPUT,
    num_train_epochs=1,
    learning_rate=2e-4,                  # LoRA tolerates ~10x higher LR than full fine-tuning
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch = 4 * 4 = 16
    warmup_steps=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=DTYPE == torch.bfloat16,
    gradient_checkpointing=DEVICE == "cuda",  # trade compute for memory on large models
    report_to="none",
    seed=SEED,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
trainer.train()

# Saves ONLY the adapter weights (a few MB) + adapter_config.json
model.save_pretrained(LORA_OUTPUT)
tokenizer.save_pretrained(LORA_OUTPUT)

# --------------------------------------------------------------------------- #
# 6. Reload: base model + adapter, then chat with it
# --------------------------------------------------------------------------- #
base = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=DTYPE).to(DEVICE)
tuned = PeftModel.from_pretrained(base, LORA_OUTPUT)
tuned.eval()

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Give three tips for staying productive."}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = tuned.generate(**inputs, max_new_tokens=120, do_sample=False, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# Next step: merge the adapter into the base weights for deployment -> see 07_hub_save_merge.py
