"""
09 - SAVING, LOADING & EXPORTING: get a trained model out of your training script.

Three levels, from most flexible to most portable:
  1. state_dict      - weights only; needs your Python class to rebuild. Use for training.
  2. torch.export    - a self-contained graph, no Python class needed. Use for deployment.
  3. ONNX            - runs outside PyTorch (onnxruntime, TensorRT, browsers, C#, ...).

Also: storing weights in lower precision for a smaller model.

Run:  python 09_save_export.py
"""

import os

import torch
import torch.nn as nn

from config import OUTPUT_DIR, seed_everything

seed_everything()
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Classifier(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 64, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


model = Classifier().eval()
example = torch.randn(4, 16)
reference = model(example)

# --------------------------------------------------------------------------- #
# 1. state_dict: the standard way. Save the constructor arguments too, so the
#    model can be rebuilt without guessing.
# --------------------------------------------------------------------------- #
path = os.path.join(OUTPUT_DIR, "classifier.pt")
torch.save({"config": {"in_features": 16, "hidden": 64, "n_classes": 3}, "model": model.state_dict()}, path)

ckpt = torch.load(path, map_location="cpu", weights_only=True)   # map_location: load GPU weights on CPU
restored = Classifier(**ckpt["config"])
restored.load_state_dict(ckpt["model"])        # strict=True by default: errors on missing/extra keys
restored.eval()
print("state_dict round trip OK:", torch.allclose(restored(example), reference))

# Loading weights into a slightly different model (e.g. a new head):
#   missing, unexpected = new_model.load_state_dict(state, strict=False)
# Never torch.save(model) (the whole object): it pickles code paths and breaks on refactors.
# For sharing on the Hugging Face Hub, safetensors is the preferred file format:
#   from safetensors.torch import save_file; save_file(model.state_dict(), "model.safetensors")

# --------------------------------------------------------------------------- #
# 2. torch.export: capture the model as a graph (ExportedProgram).
#    Declare which dimensions may change size (here the batch dimension).
# --------------------------------------------------------------------------- #
batch = torch.export.Dim("batch", min=1, max=1024)
exported = torch.export.export(model, (example,), dynamic_shapes={"x": {0: batch}})
export_path = os.path.join(OUTPUT_DIR, "classifier.pt2")
torch.export.save(exported, export_path)

loaded_program = torch.export.load(export_path)          # no Classifier class needed
out = loaded_program.module()(torch.randn(10, 16))       # a different batch size works
print("torch.export OK, output:", tuple(out.shape))
# For C++ / no-Python deployment, compile it ahead of time:
#   torch._inductor.aoti_compile_and_package(exported, package_path="model.pt2")

# --------------------------------------------------------------------------- #
# 3. ONNX (pip install onnx onnxscript; run with pip install onnxruntime)
# --------------------------------------------------------------------------- #
onnx_path = os.path.join(OUTPUT_DIR, "classifier.onnx")
try:
    torch.onnx.export(
        model, (example,), onnx_path,
        input_names=["x"], output_names=["logits"],
        dynamic_shapes={"x": {0: batch}},
    )
    print("ONNX export OK:", onnx_path)
    # import onnxruntime as ort
    # session = ort.InferenceSession(onnx_path)
    # logits = session.run(None, {"x": example.numpy()})[0]
except ImportError as e:
    print("ONNX export skipped:", e)

# --------------------------------------------------------------------------- #
# 4. Smaller weights: store the model in bfloat16 (half the size, minimal accuracy cost).
#    For int8 / int4 quantization, use the torchao library (pip install torchao):
#      from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
#      quantize_(model, Int8DynamicActivationInt8WeightConfig())
#    (torch.ao.quantization, used in older tutorials, is deprecated.)
# --------------------------------------------------------------------------- #
half_model = Classifier(**ckpt["config"])
half_model.load_state_dict(ckpt["model"])
half_model = half_model.to(torch.bfloat16).eval()
half_path = os.path.join(OUTPUT_DIR, "classifier_bf16.pt")
torch.save(half_model.state_dict(), half_path)
print(f"size: float32 {os.path.getsize(path) / 1024:.1f} KB -> bfloat16 {os.path.getsize(half_path) / 1024:.1f} KB")
diff = (half_model(example.to(torch.bfloat16)).float() - reference).abs().max().item()
print(f"max difference in bfloat16: {diff:.4f}")    # inputs must match the weights' dtype

# --------------------------------------------------------------------------- #
# 5. Inference checklist
# --------------------------------------------------------------------------- #
model.eval()                                   # 1. eval mode (dropout off, batchnorm frozen)
with torch.inference_mode():                   # 2. no autograd bookkeeping
    probs = model(example).softmax(dim=-1)     # 3. same preprocessing as in training!
print("predictions:", probs.argmax(dim=-1).tolist())
