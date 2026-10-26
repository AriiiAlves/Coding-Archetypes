"""
09 - GRADIO DEMO: wrap a model in a shareable web UI in a few lines.
Gradio is what Hugging Face Spaces use, so this file can be deployed to a Space as-is
(rename to app.py and add requirements.txt).

Run:  python 09_gradio_app.py   -> open http://127.0.0.1:7860
"""

import gradio as gr
from transformers import pipeline

from config import CHAT_MODEL, DEVICE

generator = pipeline("text-generation", model=CHAT_MODEL, device=DEVICE)


def chat(message: str, history: list) -> str:
    """`history` holds previous turns as {"role", "content"} dicts (messages format)."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn in history:
        if isinstance(turn, dict):            # messages format
            messages.append({"role": turn["role"], "content": turn["content"]})
        else:                                 # legacy (user, assistant) tuples
            messages += [{"role": "user", "content": turn[0]}, {"role": "assistant", "content": turn[1]}]
    messages.append({"role": "user", "content": message})
    out = generator(messages, max_new_tokens=200, do_sample=True, temperature=0.7)
    return out[0]["generated_text"][-1]["content"]


demo = gr.ChatInterface(
    fn=chat,
    title="Open model chat",
    description=f"Running `{CHAT_MODEL}` locally with 🤗 Transformers.",
)

if __name__ == "__main__":
    demo.launch()  # share=True gives a temporary public URL
