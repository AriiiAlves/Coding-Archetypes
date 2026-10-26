"""
08 - CUSTOM MODELS: build your own architecture on top of a pretrained backbone
and make it a first-class Hugging Face model (save_pretrained / from_pretrained /
Trainer / push_to_hub all work).

What's covered:
  - a custom PretrainedConfig (your hyperparameters, saved as config.json)
  - a custom PreTrainedModel: pretrained encoder + your own head
  - freezing layers / layer-wise control
  - returning a ModelOutput so Trainer computes the loss automatically
  - a minimal manual PyTorch training loop (when Trainer is too rigid)
  - registering with AutoConfig / AutoModel

Run:  python 08_custom_model.py
"""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from config import CLASSIFIER_BASE_MODEL, CUSTOM_MODEL_OUTPUT, DEVICE


# --------------------------------------------------------------------------- #
# 1. Config: everything needed to rebuild the model. Saved as config.json.
# --------------------------------------------------------------------------- #
class MultiTaskConfig(PretrainedConfig):
    model_type = "multitask-encoder"  # unique name used by the Auto* registry

    def __init__(
        self,
        backbone_name: str = CLASSIFIER_BASE_MODEL,
        num_labels: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_backbone_layers: int = 0,   # freeze the first N transformer layers
        **kwargs,
    ):
        self.backbone_name = backbone_name
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.freeze_backbone_layers = freeze_backbone_layers
        super().__init__(num_labels=num_labels, **kwargs)


# --------------------------------------------------------------------------- #
# 2. Model: pretrained backbone + custom MLP head + extra regression head
# --------------------------------------------------------------------------- #
class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config: MultiTaskConfig):
        super().__init__(config)
        # Build the backbone from ITS config (weights loaded below / by from_pretrained)
        backbone_config = AutoConfig.from_pretrained(config.backbone_name)
        self.backbone = AutoModel.from_config(backbone_config)
        hidden = backbone_config.hidden_size

        # Head 1: classification (custom MLP instead of a single linear layer)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_labels),
        )
        # Head 2: a scalar score (e.g. intensity, rating) - shows multi-head design
        self.regressor = nn.Linear(hidden, 1)

        self.post_init()  # HF weight init for the new layers

    def freeze_layers(self):
        """Freeze embeddings + first N transformer layers (cheaper training, less forgetting)."""
        n = self.config.freeze_backbone_layers
        if n <= 0:
            return
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        # DistilBERT: backbone.transformer.layer ; BERT/RoBERTa: backbone.encoder.layer
        layers = getattr(self.backbone, "transformer", getattr(self.backbone, "encoder", None)).layer
        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, scores=None, **kwargs):
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden_states[:, 0]  # first token ([CLS]) as sentence representation

        logits = self.classifier(cls)
        score_pred = self.regressor(cls).squeeze(-1)

        # Trainer uses `loss` if present -> combine both task losses
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            if scores is not None:
                loss = loss + 0.5 * nn.functional.mse_loss(score_pred, scores.float())

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=None)


# --------------------------------------------------------------------------- #
# 3. Register so AutoConfig / AutoModel can load it by model_type
# --------------------------------------------------------------------------- #
AutoConfig.register("multitask-encoder", MultiTaskConfig)
AutoModel.register(MultiTaskConfig, MultiTaskModel)

# --------------------------------------------------------------------------- #
# 4. Instantiate: random head + PRETRAINED backbone weights
# --------------------------------------------------------------------------- #
config = MultiTaskConfig(
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2},
    freeze_backbone_layers=4,
)
model = MultiTaskModel(config)
model.backbone = AutoModel.from_pretrained(config.backbone_name)  # load pretrained weights
model.freeze_layers()
model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M")

tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

# --------------------------------------------------------------------------- #
# 5. Minimal manual training loop (works the same with Trainer, see 04)
# --------------------------------------------------------------------------- #
texts = ["I loved it", "It was fine", "Worst purchase ever", "Absolutely fantastic", "Meh", "Terrible"]
labels = torch.tensor([2, 1, 0, 2, 1, 0])
scores = torch.tensor([0.9, 0.5, 0.0, 1.0, 0.4, 0.1])

batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5)

model.train()
for step in range(10):
    out = model(**batch, labels=labels.to(DEVICE), scores=scores.to(DEVICE))
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    print(f"step {step:2d}  loss {out.loss.item():.4f}")

# --------------------------------------------------------------------------- #
# 6. Save & reload like any HF model
# --------------------------------------------------------------------------- #
model.save_pretrained(CUSTOM_MODEL_OUTPUT)
tokenizer.save_pretrained(CUSTOM_MODEL_OUTPUT)

reloaded = AutoModel.from_pretrained(CUSTOM_MODEL_OUTPUT).to(DEVICE).eval()
with torch.no_grad():
    logits = reloaded(**tokenizer(["What a great day"], return_tensors="pt").to(DEVICE)).logits
print("Prediction:", reloaded.config.id2label[logits.argmax(-1).item()])

# To share custom code on the Hub so others can load it with trust_remote_code=True:
#   MultiTaskConfig.register_for_auto_class()
#   MultiTaskModel.register_for_auto_class("AutoModel")
#   model.push_to_hub("your-username/multitask-encoder")
