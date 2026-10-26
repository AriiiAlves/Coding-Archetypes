"""
04 - FULL FINE-TUNING with Trainer: adapt a pretrained encoder to YOUR labels.

Pattern (reusable for almost any supervised task):
  dataset -> tokenize -> AutoModelFor<Task> -> TrainingArguments -> Trainer
  -> train() -> evaluate() -> save_model() -> pipeline() for inference

Swap AutoModelForSequenceClassification for other heads:
  AutoModelForTokenClassification   (NER / tagging)
  AutoModelForQuestionAnswering     (extractive QA)
  AutoModelForSeq2SeqLM             (summarization / translation, use Seq2SeqTrainer)

Run:  python 04_finetune_classifier.py
"""

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    pipeline,
)

from config import (
    CLASSIFICATION_DATASET,
    CLASSIFIER_BASE_MODEL,
    CLASSIFIER_OUTPUT,
    MAX_EVAL_SAMPLES,
    MAX_TRAIN_SAMPLES,
    SEED,
)

# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
raw = load_dataset(CLASSIFICATION_DATASET)
train_ds = raw["train"].shuffle(seed=SEED).select(range(MAX_TRAIN_SAMPLES))
eval_ds = raw["test"].shuffle(seed=SEED).select(range(MAX_EVAL_SAMPLES))

# Human-readable label names <-> ids (saved into the model config)
label_names = raw["train"].features["label"].names  # ["neg", "pos"]
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_BASE_MODEL)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)


train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

# Pads each batch to its longest sequence (faster than padding everything to max_length)
collator = DataCollatorWithPadding(tokenizer)

# --------------------------------------------------------------------------- #
# 2. Model: pretrained body + NEW randomly initialised classification head
#    (the "Some weights were newly initialized" warning is expected)
# --------------------------------------------------------------------------- #
model = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFIER_BASE_MODEL,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)

# Optional: freeze the backbone and train only the head (much faster, less accurate)
# for param in model.base_model.parameters():
#     param.requires_grad = False

# --------------------------------------------------------------------------- #
# 3. Metrics
# --------------------------------------------------------------------------- #
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        **accuracy.compute(predictions=preds, references=labels),
        **f1.compute(predictions=preds, references=labels, average="weighted"),
    }


# --------------------------------------------------------------------------- #
# 4. Training configuration - the most important knobs
# --------------------------------------------------------------------------- #
args = TrainingArguments(
    output_dir=CLASSIFIER_OUTPUT,
    num_train_epochs=2,
    learning_rate=2e-5,                  # typical for full fine-tuning of BERT-like models
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,       # raise to simulate bigger batches on small GPUs
    weight_decay=0.01,
    warmup_steps=0.1,                    # float < 1 = ratio of total steps; int = exact steps
    lr_scheduler_type="linear",
    eval_strategy="epoch",               # evaluate every epoch
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=20,
    bf16=False,                          # True on Ampere+ GPUs
    fp16=False,                          # True on older NVIDIA GPUs
    report_to="none",                    # "tensorboard", "wandb", "mlflow", ...
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,          # saved alongside the model
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# --------------------------------------------------------------------------- #
# 5. Train, evaluate, save
# --------------------------------------------------------------------------- #
trainer.train()                          # resume with trainer.train(resume_from_checkpoint=True)
print(trainer.evaluate())
trainer.save_model(CLASSIFIER_OUTPUT)    # writes config, weights (safetensors) and tokenizer
# trainer.push_to_hub()                  # needs hub_model_id in TrainingArguments + login

# --------------------------------------------------------------------------- #
# 6. Use the fine-tuned model
# --------------------------------------------------------------------------- #
clf = pipeline("text-classification", model=CLASSIFIER_OUTPUT)
print(clf(["An absolute masterpiece.", "Two hours I will never get back."]))
