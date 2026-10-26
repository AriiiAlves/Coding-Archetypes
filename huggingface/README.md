# Hugging Face Archetype — Open Models & Customization

A starter kit for using and customizing **open models from the Hugging Face Hub**.
Each script is standalone, heavily commented and runnable on a laptop CPU (the defaults are tiny models).
Copy the script closest to your goal and grow it.

## Setup

```bash
cd huggingface
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # GPU users: install torch from https://pytorch.org first
hf auth login                     # optional: only for gated/private models and uploads
```

All model and dataset names, paths and hardware detection live in **`config.py`**. Change models there.

## The scripts (in learning order)

| # | File | What you learn | Key tools |
|---|------|----------------|-----------|
| 01 | `01_pipelines.py` | One-line inference for any task: classification, zero-shot, chat | `pipeline()` |
| 02 | `02_model_and_tokenizer.py` | Loading models manually, tokenization, chat templates, generation params, streaming, batching, 4-bit quantization | `AutoTokenizer`, `AutoModelForCausalLM`, `GenerationConfig`, `TextIteratorStreamer`, `BitsAndBytesConfig` |
| 03 | `03_datasets.py` | Loading Hub/local data, filtering, splitting, tokenizing with `map`, saving | `datasets` |
| 04 | `04_finetune_classifier.py` | **Full fine-tuning** of an encoder on your labels, with metrics and early stopping | `Trainer`, `TrainingArguments`, `evaluate` |
| 05 | `05_lora_finetune_llm.py` | **LoRA / QLoRA fine-tuning** of a chat LLM on instruction data, loss on answers only | `peft` (`LoraConfig`, `get_peft_model`) |
| 06 | `06_embeddings_search.py` | Sentence embeddings and semantic search (the core of RAG), plus manual mean pooling | `sentence-transformers` |
| 07 | `07_hub_save_merge.py` | Merging LoRA adapters, saving, pushing to and downloading from the Hub, searching models | `huggingface_hub`, `merge_and_unload` |
| 08 | `08_custom_model.py` | **Custom architectures**: your own config and model class with a pretrained backbone, custom and multi-task heads, layer freezing, manual training loop, Auto-class registration | `PretrainedConfig`, `PreTrainedModel` |
| 09 | `09_gradio_app.py` | A chat web UI that can be deployed to Hugging Face Spaces | `gradio` |

Run any script with `python 0X_name.py`. Outputs go to `outputs/`, which is git-ignored.
Script 07 merges the adapter that script 05 produces.

Useful environment variables:

| Variable | Effect |
|----------|--------|
| `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` | Dataset size used by the training scripts (defaults are 1000 and 200) |
| `HF_OUTPUT_DIR` | Where checkpoints and models are written |
| `HF_TOKEN` | Hub access token (alternative to `hf auth login`) |
| `HF_HOME` | Cache location (default is `~/.cache/huggingface`) |
| `HF_HUB_OFFLINE=1` | Use only cached files |

## Which customization approach should I use?

| Goal | Approach | Script |
|------|----------|--------|
| Use a model as-is | `pipeline` or `generate` | 01, 02 |
| Change behavior through instructions only | Prompting with a system message and chat template | 02 |
| New labels or a new task on a small model | Full fine-tuning | 04 |
| Teach an LLM a style, format or domain cheaply | LoRA (on a GPU with little memory: QLoRA) | 05 |
| Let the model use your documents without training | Embeddings and retrieval (RAG) | 06 |
| A new head, several outputs, or a mix of models | Custom `PreTrainedModel` | 08 |

## Scaling up

- **Bigger models**: change `CHAT_MODEL` in `config.py`, for example `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct` (gated: accept the license first). With 8–24 GB of GPU memory, set `USE_QLORA = True` in script 05.
- **Mixed precision**: on Ampere or newer GPUs, `bf16=True` in `TrainingArguments`.
- **Multiple GPUs**: run `accelerate config` once, then `accelerate launch 04_finetune_classifier.py`.
- **Experiment tracking**: set `report_to="tensorboard"` or `"wandb"`.
- **Less boilerplate for LLM training**: [TRL](https://huggingface.co/docs/trl) (`SFTTrainer`, `DPOTrainer`, `GRPOTrainer`) wraps the flow in script 05 and adds preference and RL tuning.
- **Production serving**: vLLM or Text Generation Inference (TGI) can load the merged model from script 07 directly.

## Common pitfalls

- **Pad token**: decoder models often lack one. Set `tokenizer.pad_token = tokenizer.eos_token`.
- **Batched generation**: decoder-only models need `tokenizer.padding_side = "left"`.
- **Prompt format**: always use `apply_chat_template` for chat models. Hand-written prompts silently hurt quality.
- **"Some weights were newly initialized" warning**: this is expected when you add a new head (scripts 04 and 08).
- **Running out of memory**: lower the batch size and raise `gradient_accumulation_steps`, turn on `gradient_checkpointing`, or load in 4-bit.
- **Library versions**: this archetype was tested with `transformers` 5.x, `peft` 0.21, `datasets` 5.x and `huggingface_hub` 2.x. Older tutorials use renamed arguments (`torch_dtype` is now `dtype`, `evaluation_strategy` is now `eval_strategy`, `tokenizer=` in `Trainer` is now `processing_class=`, and `warmup_ratio` is now a float `warmup_steps`).
