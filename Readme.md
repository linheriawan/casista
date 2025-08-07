# 🤖 Casista - AI Assistant System

A modular AI assistant framework supporting chat, voice, and image generation. Create and manage multiple AI assistants with different personalities and capabilities.

## ✨ Features

- 🗣️ **Multi-Modal Support**: Chat, voice conversation, and image generation
- 🎭 **Multiple Assistants**: Create different AI personas for various tasks
- 🧠 **Model Flexibility**: Use any Ollama model
- 📁 **File Operations**: Create, read, update, delete files via natural language
- 🎤 **Voice Features**: Speech recognition and text-to-speech
- 📚 **Knowledge Base**: RAG system for document integration

## 🚀 Quick Start

### 1. System Dependencies

**Install Ollama** (Required):
```bash
# Install Ollama from https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Download your preferred models
ollama pull qwen2.5-coder:3b
ollama pull llama3.2:3b
```

**Audio Libraries** (Required for voice features):
```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# Fedora/CentOS
sudo dnf install portaudio-devel
```

### 2. Installation

```bash
git clone https://github.com/linheriawan/casista
cd casista

# Setup environment and dependencies
python3 setup.py

# Install globally (creates 'coder' command)
python3 install.py
```

**Verify Installation**:
```bash
coder --help
coder --version
```

### 3. Create Your First Assistant

```bash
# Create an assistant
coder create-agent

# List available assistants
coder list-agents
```

## 💻 Usage

### Basic Chat
```bash
# Start chat with an assistant
coder mycoder chat

# One-shot query
coder mycoder chat --query "Create a Python web scraper"
```

### Voice Mode
```bash
# Voice conversation
coder anna speech
```

### Image Generation
```bash
# Generate image
coder artist image --prompt "A cyberpunk cityscape at sunset"
```

## 🔧 Voice Setup

### Speech Recognition Options

**Online (Default):**
- Google Speech API - requires internet, very accurate

**Local (Privacy-first):**
```bash
# Option 1: Whisper (best accuracy, offline)
pip install openai-whisper
coder --set-speech-backend anna whisper

# Option 2: Vosk (fast, lightweight, offline)
pip install vosk
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
coder --set-speech-backend anna vosk
```

## 📁 File Operations

The assistant can manipulate files using special syntax:

```
User: Create a Python hello world script
Assistant: I'll create a simple Python script for you.

```create:hello.py
print("Hello, World!")
```
```

**Available operations:**
- `create:filename.ext` - Create files
- `update:filename.ext` - Update files  
- `read:filename.ext` - Read files
- `delete:filename.ext` - Delete files
- `mkdir:dirname` - Create directories

## 🔧 Troubleshooting

### Installation Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 setup.py
```

### Audio Issues (macOS)
```bash
# Install audio libraries
brew install portaudio
./venv/bin/pip install pyaudio

# Check microphone permissions in System Preferences
```

### Model Issues
```bash
# Download missing model
ollama pull qwen2.5-coder:3b
```

## 📚 More Information

- **Development Guide**: See `CLAUDE.md` for complete technical documentation
- **Project Roadmap**: See `PROJECT_PLAN.md` for development plans
- **Issues**: Open GitHub issues for bugs and feature requests

---

**Enjoy your AI assistant system! 🚀**