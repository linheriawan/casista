# Configuration Model Explanation

## 🤖 Two Different Model Types

### **1. Chat/Conversation Model**
```toml
[assistant]
model = "qwen3:4b"  # ← Ollama model for chat, reasoning, conversation
```
- **Purpose**: Powers chat conversations, reasoning, prompt enhancement
- **Type**: Ollama LLM models (qwen, llama, etc.)
- **Used in**: Chat mode, Speech mode, AI-assisted image prompting

### **2. Image Generation Models**
```toml
[image]
models = ["dreamlike-art/dreamlike-anime-1.0", "hakurei/waifu-diffusion"]  # ← HuggingFace Diffusion models
```
- **Purpose**: Generate actual images from text prompts
- **Type**: HuggingFace Diffusion models (Stable Diffusion, SDXL, etc.)
- **Used in**: Image mode generation, img2img, upscaling

## ✅ **What Actually Happens:**

### **In Image Mode:**
1. **Chat Model** (`qwen3:4b`) enhances your prompt: "cat" → "a photorealistic cat with detailed fur, high quality, 8k"
2. **Image Model** (`stable-diffusion-v1-5`) generates the actual image from enhanced prompt
3. **Both models work together** for better results!

### **Model Selection:**
- **Chat model**: Set via `[assistant].model` or `--model` flag
- **Image model**: Set via `[image].models` or `switch model <name>` command in interactive mode

## 🔧 **Capability Control (Fixed):**

### **Single Source of Truth:**
```toml
[capabilities]
image_generation = true  # ← ONLY enablement flag needed
```

### **Removed Redundancy:**
```toml
[image]
# enabled = true  ← REMOVED! Was confusing duplicate
models = [...]    # ← Image model list only
```

## 💡 **Clear Separation:**

| Config Section | Purpose | Model Type | Example |
|---|---|---|---|
| `[assistant].model` | Chat/Reasoning | Ollama LLM | `qwen3:4b` |
| `[image].models` | Image Generation | HuggingFace Diffusion | `stable-diffusion-v1-5` |
| `[capabilities].image_generation` | Enable/Disable | Boolean | `true`/`false` |

## 🎯 **Best Practice:**
- **Keep them separate** - they serve different purposes
- **Chat model** = conversation intelligence
- **Image models** = visual generation
- **Capabilities** = feature toggles