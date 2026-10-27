# Coding-Archetypes

A collection of **code archetypes**: small, heavily commented starter projects that show the essential tools of a technology.

They are templates to build complex applications from.

## What an archetype is

Each archetype is a folder that:

- **Covers the most important tools** of one technology or library. These are the 20% of the API you use 80% of the time.
- **Is runnable as-is**, with small defaults, so it works on an ordinary laptop.
- **Is commented for learning and reuse.** Comments explain *why* each step exists and what to change for your use case.
- **Is split into numbered scripts**, ordered from basic to advanced, so you can copy only the piece you need.
- **Centralizes settings** (model names, paths, hyperparameters) in one config file.
- **Has its own README** with setup steps, a map of the scripts, how to scale up and common pitfalls.

## Archetypes

| Folder | Technology | Covers |
|--------|------------|--------|
| [`huggingface/`](huggingface/) | Hugging Face (Transformers, Datasets, PEFT, Hub, Gradio) | Running open models, tokenization and generation, datasets, full fine-tuning, LoRA/QLoRA, embeddings and semantic search, custom model architectures, publishing to the Hub, demo UIs |
| [`pytorch/`](pytorch/) | PyTorch (torch, torchvision) | Tensors and autograd, datasets and dataloaders, building models with `nn.Module`, a reusable training loop with checkpointing, CNNs, transfer learning, a transformer from scratch, mixed precision and `torch.compile`, export to `torch.export` / ONNX, multi-GPU training with DDP |
| [`react/`](react/) | React 19 + TypeScript (Vite, React Router, Vitest) | Components and props, state and events, effects and refs, forms and actions, data fetching with Suspense, context and reducers, custom hooks, performance, routing with loaders, testing with Testing Library |

## How to use

1. Pick the archetype for your stack and follow its README to install it.
2. Run the scripts in order to see each tool working.
3. Copy the script closest to your goal into your project, then change the config and data.

## Adding a new archetype

Create a folder named after the technology that contains:

```
<technology>/
├── README.md          # setup, script map, scaling tips, pitfalls
├── requirements.txt   # or package.json, go.mod, ...
├── config.*           # central settings
└── 01_*.ext ...       # numbered, self-contained, commented examples
```

Then add a row to the table above.
