---
description: Guide for vLLM-based Inference, Fine-tuning, and Web UI Setup
---

# vLLM Guide: Inference, Fine-Tuning, and Web UI

This guide explains how to use **vLLM** for high-speed inference, how to fine-tune models compatible with vLLM, and how to set up a Web UI.

## 1. Concept Clarification

*   **vLLM is for Inference**: vLLM is a high-throughput and memory-efficient *inference* engine. It is **not** a training library. You cannot "fine-tune a vLLM model" directly using vLLM itself.
*   **Workflow**:
    1.  **Train**: Use `transformers`, `unsloth`, or `axolotl` to fine-tune a base model (e.g., Gemma 2, Llama 3) and save the adapters/merged model.
    2.  **Serve**: Load the *trained* model into **vLLM** to serve it as an API (OpenAI-compatible).
    3.  **UI**: Connect a Web UI (like Open WebUI or Gradio) to the vLLM API.

## 2. Fine-Tuning for vLLM

Since vLLM loads standard Hugging Face models (safetensors), the fine-tuning process is the same as usual.

1.  **Fine-tune** your model (as you are doing with `finetunetest.py`).
2.  **Merge & Save**: vLLM works best with merged models (Base Model + LoRA Adapter merged).
    *   *Note*: vLLM *can* load LoRA adapters dynamically, but merging is often simpler for deployment.

```python
# Example: Merging LoRA into Base Model (after training)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it")
model = PeftModel.from_pretrained(base_model, "your_adapter_folder")
model = model.merge_and_unload() # Merge weights

model.save_pretrained("merged_model_for_vllm")
tokenizer.save_pretrained("merged_model_for_vllm")
```

## 3. Serving with vLLM (OpenAI Compatible API)

Once you have your model (base or merged), you can serve it using vLLM.

**Installation:**
```bash
pip install vllm
```

**Running the Server:**
```bash
# Serve a base model or your merged model
python -m vllm.entrypoints.openai.api_server \
    --model ./merged_model_for_vllm \
    --served-model-name gemma-2-finetuned \
    --port 8000 \
    --dtype bfloat16
```
*   `--model`: Path to your local model or Hugging Face model ID.
*   `--served-model-name`: The name you will use in the API calls (e.g., "gpt-3.5-turbo" replacement).

## 4. Web UI Options

You don't need to build a UI from scratch. There are excellent open-source options that connect to the vLLM API.

### Option A: Open WebUI (Recommended)
A ChatGPT-style UI that supports OpenAI-compatible APIs (which vLLM provides).

1.  **Install via Docker** (Easiest):
    ```bash
    docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
    ```
2.  **Connect**:
    *   Open browser at `http://localhost:3000`.
    *   Go to Settings -> Connections.
    *   Set API URL to `http://host.docker.internal:8000/v1`.
    *   You should see your `gemma-2-finetuned` model available.

### Option B: Simple Gradio UI (Custom)
If you want to build a simple custom UI in Python:

```python
import gradio as gr
from openai import OpenAI

# Connect to local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

def chat(message, history):
    response = client.chat.completions.create(
        model="gemma-2-finetuned",
        messages=[{"role": "user", "content": message}],
        stream=True
    )
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

demo = gr.ChatInterface(chat)
demo.launch()
```

## Summary Checklist

1.  [ ] **Fine-tune** model using Unsloth/Transformers.
2.  [ ] **Merge** LoRA adapter into base model (optional but recommended).
3.  [ ] **Run vLLM** server pointing to your model path.
4.  [ ] **Run Web UI** (Open WebUI or Gradio) connected to vLLM port (8000).
