# 🤖 Casista - Multi-Modal AI Assistant

A sophisticated local AI assistant system powered by Ollama with support for chat, voice, and image modes. Create multiple AI personas with different models and maintain separate conversation contexts.

## ✨ Features

- 🗣️ **Multi-Modal Support**: Chat, voice conversation, and image processing
- 🎭 **Multiple AI Personas**: Create different assistants with unique personalities
- 🧠 **Model Flexibility**: Use any Ollama model (qwen2.5-coder, llama3.2, etc.)
- 💬 **Smart Context Management**: Separate conversation history per assistant
- 🎤 **Advanced Voice Features**: 177+ system voices, speech rate control, same-line progress
- 📁 **File Operations**: Create, read, update, delete files via natural language
- 👤 **User Identity**: Smart user detection with manual override
- ⚡ **Global Access**: Install once, use from anywhere

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** and download models:
   ```bash
   # Install Ollama (https://ollama.ai)
   ollama pull qwen2.5-coder:3b
   ollama pull llama3.2:3b
   ```

2. **macOS Audio Libraries** (for voice mode):
   ```bash
   brew install portaudio
   ```

### Installation

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd casista
   python3 setup.py          # Creates venv and installs dependencies
   ```

2. **Install globally**:
   ```bash
   python3 install.py        # Creates 'coder' command globally
   # or for system-wide: python3 install.py --system
   ```

3. **Verify installation**:
   ```bash
   coder --help
   coder --list-models
   ```

## 💻 Usage

### Chat Mode
```bash
# Interactive text chat with file operations
coder qwen2.5-coder:3b mycoder chat

# One-shot query
coder llama3.2:3b advisor chat --query "Create a Python web scraper"

# Disable file operations for safe chat
coder llama3.2:3b advisor chat --no-file-ops
```

### Voice Mode
```bash
# Interactive voice conversation
coder qwen2.5-coder:3b anna speech

# Voice commands during conversation:
# - "What voice are you using?" - Shows current voice
# - "Speak slower" / "Speak faster" - Adjusts speech rate
# - "Who am I" / "Set user [name]" - Identity management
# - "Exit" / "Goodbye" - End conversation
```

### Voice Selection
```bash
# List all available system voices (177+ voices)
python3 voice_selector.py list

# Test a specific voice by ID
python3 voice_selector.py test 132

# Set voice for an assistant
python3 voice_selector.py set anna 132

# Popular voices:
# ID 132: Samantha (English US, Female)
# ID 82: Karen (English AU, Female)
# ID 14: Daniel (English UK, Male)
```

### Image Generation Mode
```bash
# Interactive image generation
coder qwen2.5-coder:3b artist image

# One-shot image generation
coder llama3.2:3b artist image --query "Create a cyberpunk cityscape"

# The assistant will respond with generation commands:
# User: "Generate a sunset over mountains"
# Assistant: I'll create that image for you.
# ```generate:sunset_mountains.png
# A beautiful sunset over mountain peaks, vibrant orange and purple sky, peaceful landscape, high detail
# ```
```

### Assistant Management
```bash
# Different assistants for different purposes
coder qwen2.5-coder:7b architect chat    # Architecture discussions
coder llama3.2:3b reviewer chat          # Code reviews  
coder qwen2.5-coder:3b coder chat        # Coding tasks
coder qwen2.5-coder:3b artist image      # Image generation

# Reset specific assistant context
coder qwen2.5-coder:3b mycoder chat --reset

# Show assistant configuration
coder qwen2.5-coder:3b mycoder chat --config
```

## 📁 File Operations

In chat mode, the assistant can manipulate files using special syntax:

### Create Files
```
User: Create a Python hello world script
Assistant: I'll create a simple Python script for you.

```create:hello.py
print("Hello, World!")
```
```

### Update Files  
```
User: Add comments to that file
Assistant: I'll add comments to the Python script.

```update:hello.py
# Simple Hello World script
print("Hello, World!")  # Print greeting message
```
```

### Other Operations
- **Read files**: ````read:filename.ext```
- **Delete files**: ````delete:filename.ext```  
- **Create directories**: ````mkdir:dirname```

## 🎭 Assistant Personalities

Customize assistant personalities for voice mode:

```bash
# Set personality presets
python3 set_voice_personality.py anna friendly
python3 set_voice_personality.py anna professional  
python3 set_voice_personality.py anna casual

# Available presets: friendly, professional, casual, enthusiastic, wise

# Custom personality
python3 set_voice_personality.py anna "You are anna, a helpful coding mentor who explains things clearly and encourages learning."
```

## 📂 Project Structure

```
casista/
├── main.py                 # Main CLI interface
├── assistant_base.py       # Assistant framework
├── coder.py               # Legacy single assistant
├── setup.py               # Environment setup
├── install.py             # Global installation
├── voice_selector.py      # Voice selection tool
├── set_voice_personality.py # Personality customization
├── .ai_context/           # Assistant contexts (created on first use)
│   ├── assistant_name/
│   │   ├── context.json   # Conversation history
│   │   └── config.json    # Assistant configuration
│   └── other_assistant/
├── venv/                  # Python virtual environment
└── CLAUDE.md             # Development documentation
```

## 🛠️ Development

### Dependencies
- **Core**: ollama, rich, typer, prompt_toolkit
- **Speech** (optional): speechrecognition, pyttsx3, pyaudio
- **Image** (optional): diffusers, torch, transformers, accelerate

### Adding New Features
1. Extend `BaseAssistant` class in `assistant_base.py`
2. Add new modes in `create_assistant()` factory function
3. Update CLI arguments in `main.py`

### Troubleshooting

**Speech issues on macOS**:
```bash
# Install audio libraries
brew install portaudio
./venv/bin/pip install pyaudio

# Check microphone permissions in System Preferences > Security & Privacy
```

**Image generation issues**:
```bash
# Install image dependencies
./venv/bin/pip install diffusers torch transformers accelerate

# Test image generation
python3 test_image_generation.py

# Available lightweight models (auto-selected):
# • TinySD: ~800MB (fastest, good quality)
# • DreamLike Anime: ~2GB (anime/artistic style)
# • OpenJourney v4: ~2GB (photorealistic)
# • Waifu Diffusion: ~2GB (anime characters)
# • SD v1.5: ~4GB (fallback, highest quality)
```

**Virtual environment issues**:
```bash
# Recreate virtual environment
rm -rf venv
python3 setup.py
```

**Model not found**:
```bash
# Check available models
coder --list-models
ollama list

# Download missing model
ollama pull qwen2.5-coder:3b
```

## 🌟 Examples

### Development Workflow
```bash
# Start coding session with file operations
coder qwen2.5-coder:7b developer chat
💬 [heriawan]: Create a REST API with FastAPI

# Voice consultation for architecture
coder llama3.2:3b architect speech
🎤 "Design a microservices architecture for an e-commerce platform"

# Code review
coder qwen2.5-coder:3b reviewer chat --query "Review this Python file"
```

### Multi-Language Support
```bash
# Spanish voice assistant
python3 voice_selector.py set carlos 97  # Monica (Spanish)
coder llama3.2:3b carlos speech

# French voice assistant  
python3 voice_selector.py set marie 165  # Thomas (French)
coder qwen2.5-coder:3b marie speech
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🆘 Support

- Check `CLAUDE.md` for development documentation
- Open issues on GitHub for bugs/features
- See troubleshooting section above for common issues

---

**Enjoy your local AI assistant! 🚀**
