# 🤖 Casista - Modular AI Assistant System

A sophisticated AI assistant framework with modular architecture, supporting chat, voice, and image generation modes. Create and manage multiple AI assistants with different personalities, models, and capabilities.

## ✨ Features

- 🏗️ **Modular Architecture**: Clean separation of concerns with library/, helper/, and configuration/ modules
- 🗣️ **Multi-Modal Support**: Chat, voice conversation, and image generation
- 🎭 **Assistant Management**: Create, configure, and manage multiple AI personas
- 🧠 **Model Flexibility**: Use any Ollama model with per-assistant preferences
- 💬 **Smart Context Management**: Separate conversation history and configuration per assistant
- 📚 **RAG Knowledge**: Global knowledge base system with vector search capabilities
- 🎤 **Advanced Voice Features**: 177+ system voices, multiple speech recognition backends
- 📁 **File Operations**: Create, read, update, delete files via natural language
- ⚙️ **TOML Configuration**: Human-readable configuration files with multi-line string support
- 📍 **Flexible Working Directory**: Session-based directory management with in-chat navigation

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

**Quick Setup**:
```bash
git clone <your-repo-url>
cd casista

# Setup environment and dependencies (creates venv, installs packages)
python3 setup.py

# Install globally (creates 'coder' command)
python3 install.py
# or system-wide: python3 install.py --system
```

**Verify Installation**:
```bash
coder --help
coder --version
coder setup  # Run interactive setup wizard
```

### 3. Create Your First Assistant

```bash
# Interactive assistant creation
coder create-agent

# Or create manually
coder --create-agent

# List available assistants
coder list-agents
```

## 💻 Usage

### Basic Chat Sessions
```bash
# Start chat with an assistant (working dir defaults to current)
coder mycoder chat

# Specify working directory for this session
coder mycoder chat --working-dir /path/to/project

# One-shot query
coder mycoder chat --query "Create a Python web scraper"

# Disable file operations for safe chat
coder mycoder chat --no-file-ops
```

### In-Session Directory Navigation
```bash
# During a chat session, change working directory:
💬 [user] (project): /dir=../other-project
📁 Changed working directory to: /path/to/other-project

💬 [user] (other-project): /dir=./subdir
📁 Changed working directory to: /path/to/other-project/subdir
```

### Voice Mode
```bash
# Interactive voice conversation
coder anna speech

# Voice commands during conversation:
# - "What voice are you using?" - Shows current voice
# - "Speak slower" / "Speak faster" - Adjusts speech rate
# - "Who am I" / "Set user [name]" - Identity management
# - "Exit" / "Goodbye" - End conversation
```

### Image Generation
```bash
# Generate image with prompt
coder artist image --prompt "A cyberpunk cityscape at sunset"

# Interactive image mode
coder artist image
```

## 🛠️ Management Commands

### Assistant Management
```bash
# List all assistants
coder list-agents

# Show assistant details
coder --show-agent mycoder

# Clone an assistant
coder --clone-agent mycoder webcoder

# Delete an assistant
coder --delete-agent oldcoder

# Configure assistant interactively
coder --configure mycoder
```

### Model Management
```bash
# List available models
coder list-models

# Download/pull a model
coder --download-model qwen2.5-coder:7b

# Set model for an assistant
coder --set-model mycoder qwen2.5-coder:7b

# Show model information
coder --model-info qwen2.5-coder:3b
```

### Voice Configuration
```bash
# List available TTS voices
coder list-voices

# Test a voice
coder --test-voice 5

# Set voice for an assistant
coder --set-voice anna 5

# List speech recognition backends
coder --list-speech-backends

# Set speech backend (google/whisper/vosk)
coder --set-speech-backend anna whisper

# Interactive voice configuration
coder --configure-voice anna
```

### Knowledge Management (RAG)
```bash
# Index a directory into RAG knowledge
coder --index-knowledge ./docs python_docs

# List available knowledge files
coder list-knowledge

# Set RAG files for an assistant
coder --set-rag mycoder python_docs,web_dev

# Search in knowledge base
coder --search-knowledge python_docs "async functions"
```

### Personality Management
```bash
# List available personalities
coder --list-personalities

# Set personality for an assistant
coder --set-personality mycoder creative
```

## 📁 File Operations

The assistant can manipulate files using special code block syntax:

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

## 🏗️ Architecture

### Modular Structure
```
casista/
├── main.py                     # Unified CLI interface
├── library/                    # Core functionality
│   ├── assistant_cfg.py        # Assistant configuration management
│   ├── config_loader.py        # Base TOML configuration loader
│   ├── model_cfg.py            # Model management
│   ├── personality_cfg.py      # Personality system
│   ├── prompt_cfg.py           # Prompt templates
│   ├── coding/                 # Code generation and file operations
│   ├── conversation/           # Chat management and context
│   └── image_generation/       # Image generation capabilities
├── helper/                     # Management utilities
│   ├── manage_agent.py         # Agent creation and configuration
│   ├── manage_voice.py         # Voice and speech management
│   ├── manage_model.py         # AI model management
│   └── rag_knowledge.py        # RAG knowledge management
├── configuration/              # TOML configuration files
│   ├── default.model.toml      # Model definitions
│   ├── default.personality.toml # Personality configurations
│   └── default.prompt.toml     # Prompt templates
├── knowledge/                  # Global RAG knowledge (.ragfile storage)
├── .ai_context/               # Per-assistant contexts
│   └── [assistant_name]/      # Individual assistant data
└── requirements.txt
```

### Key Benefits
- **Separation of Concerns**: Each module has a specific responsibility
- **TOML Configuration**: Human-readable config files with multi-line strings
- **Global vs Session Data**: Knowledge is shared, working directories are per-session
- **Modular Dependencies**: Install only what you need

## 📦 Dependencies

### Core Dependencies (Always Required)
- **ollama**: AI model interface
- **rich**: Terminal formatting and UI
- **toml**: TOML configuration file parsing

### Speech Dependencies (Optional)
Install for voice features:
```bash
# System audio libraries (install first)
brew install portaudio  # macOS
# or: sudo apt-get install portaudio19-dev  # Ubuntu

# Python packages
pip install speechrecognition pyttsx3 pyaudio

# Local speech recognition (optional, for privacy)
pip install openai-whisper  # Best accuracy, offline
pip install vosk           # Fast, lightweight, offline
```

### Image Dependencies (Optional)
Install for image generation:
```bash
pip install diffusers torch transformers accelerate
```

## 🎤 Voice Setup Options

### Online Speech Recognition
- **Google Speech API**: Built-in, requires internet, very accurate
- Default choice, works out of the box

### Local Speech Recognition (Privacy-First)
- **OpenAI Whisper**: Best accuracy, completely offline (~74MB-1.5GB models)
- **Vosk**: Fast and lightweight, offline (~50MB models)

```bash
# Set up Whisper (recommended for privacy)
pip install openai-whisper
coder --set-speech-backend anna whisper

# Set up Vosk (faster, smaller)
pip install vosk
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
coder --set-speech-backend anna vosk
```

See `LOCAL_SPEECH_SETUP.md` for detailed speech recognition setup.

## 🔧 Troubleshooting

### Installation Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 setup.py

# System-wide dependencies
brew install portaudio  # macOS audio
sudo apt-get install portaudio19-dev  # Ubuntu audio
```

### Model Issues
```bash
# Check available models
ollama list
coder list-models

# Download missing model
ollama pull qwen2.5-coder:3b
```

### Audio Issues (macOS)
```bash
# Install audio libraries
brew install portaudio
./venv/bin/pip install pyaudio

# Check microphone permissions
# System Preferences > Security & Privacy > Microphone
```

### Configuration Issues
```bash
# Reset assistant configuration
coder --delete-agent problematic_assistant
coder create-agent

# Check assistant settings
coder --show-agent mycoder
```

## 🌟 Examples

### Development Workflow
```bash
# Create a coding assistant
coder create-agent
# Name: developer, Model: qwen2.5-coder:7b, Personality: coder

# Start development session
coder developer chat --working-dir /path/to/project
💬 [user]: Create a REST API with FastAPI

# Switch to different project during session
💬 [user]: /dir=../other-project
💬 [user]: Add tests for the user model
```

### Multi-Assistant Setup
```bash
# Create specialized assistants
coder create-agent  # Name: architect, Personality: system_designer
coder create-agent  # Name: reviewer, Personality: code_reviewer  
coder create-agent  # Name: writer, Personality: technical_writer

# Use them for different tasks
coder architect chat --query "Design microservices architecture"
coder reviewer chat --query "Review this code for security issues"
coder writer speech  # Voice mode for documentation discussion
```

### RAG Knowledge Setup
```bash
# Index your documentation
coder --index-knowledge ./docs company_docs
coder --index-knowledge ./api-reference api_docs

# Create assistant with knowledge access
coder create-agent  # Name: support
coder --set-rag support company_docs,api_docs

# Now the assistant has access to your documentation
coder support chat --query "How do I authenticate with our API?"
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the modular architecture patterns
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## 🆘 Support

- **Documentation**: Check `CLAUDE.md` for development guidance
- **Voice Setup**: See `LOCAL_SPEECH_SETUP.md` for speech recognition
- **Issues**: Open GitHub issues for bugs/features
- **Architecture**: Follow the modular patterns in `library/`, `helper/`, `configuration/`

---

**Enjoy your modular AI assistant system! 🚀**