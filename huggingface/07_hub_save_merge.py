"""
07 - HUB, SAVING & MERGING: move models between your disk and the Hugging Face Hub.

What's covered:
  - authentication
  - save_pretrained / from_pretrained (local folders)
  - merging a LoRA adapter into the base model (from 05_lora_finetune_llm.py)
  - push_to_hub for models, tokenizers and adapters
  - downloading single files / whole repos with huggingface_hub
  - searching the Hub programmatically

Run:  python 07_hub_save_merge.py   (run 05 first to have an adapter to merge)
"""

import os

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CHAT_MODEL, LORA_OUTPUT, MERGED_OUTPUT

# --------------------------------------------------------------------------- #
# 1. Authentication (needed for private/gated models and for uploading)
#    Option A (terminal):  hf auth login
#    Option B (env var):   export HF_TOKEN=hf_xxx
#    Option C (code):      from huggingface_hub import login; login()
#    Gated models (e.g. Llama) also require accepting the licence on the model page.
# --------------------------------------------------------------------------- #
api = HfApi()
try:
    print("Logged in as:", api.whoami()["name"])
except Exception:
    print("Not logged in - public downloads still work, uploads won't.")

# --------------------------------------------------------------------------- #
# 2. Merge LoRA adapter -> standalone model (no PEFT needed at inference time)
# --------------------------------------------------------------------------- #
if os.path.isdir(LORA_OUTPUT):
    base = AutoModelForCausalLM.from_pretrained(CHAT_MODEL)  # load in full precision for merging
    merged = PeftModel.from_pretrained(base, LORA_OUTPUT).merge_and_unload()
    merged.save_pretrained(MERGED_OUTPUT)                     # safetensors by default
    AutoTokenizer.from_pretrained(LORA_OUTPUT).save_pretrained(MERGED_OUTPUT)
    print(f"Merged model saved to {MERGED_OUTPUT}")

    # The merged folder now loads like any Hub model:
    reloaded = AutoModelForCausalLM.from_pretrained(MERGED_OUTPUT)
else:
    print(f"No adapter at {LORA_OUTPUT} - run 05_lora_finetune_llm.py first to test merging.")

# --------------------------------------------------------------------------- #
# 3. Upload to the Hub (uncomment & set your repo id)
# --------------------------------------------------------------------------- #
REPO_ID = "your-username/my-finetuned-model"
# model.push_to_hub(REPO_ID, private=True)
# tokenizer.push_to_hub(REPO_ID, private=True)
# peft_model.push_to_hub(REPO_ID + "-lora")            # adapters are tiny to share
# api.upload_folder(folder_path=MERGED_OUTPUT, repo_id=REPO_ID, repo_type="model")

# --------------------------------------------------------------------------- #
# 4. Download files without loading a model
# --------------------------------------------------------------------------- #
config_path = hf_hub_download(repo_id=CHAT_MODEL, filename="config.json")
print("Downloaded config to:", config_path)

# Whole repo (only some file types) -> local folder; useful for offline use / Docker images
local_dir = snapshot_download(repo_id=CHAT_MODEL, allow_patterns=["*.json", "*.txt"])
print("Snapshot at:", local_dir)

# Offline mode afterwards:  export HF_HUB_OFFLINE=1
# Cache location:           ~/.cache/huggingface  (override with HF_HOME)

# --------------------------------------------------------------------------- #
# 5. Search the Hub
# --------------------------------------------------------------------------- #
for m in api.list_models(pipeline_tag="text-generation", search="SmolLM2", sort="downloads", limit=5):
    print(f"  {m.id}  (downloads: {m.downloads})")
