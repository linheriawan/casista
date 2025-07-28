# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Setup
```bash
# Setup environment and dependencies
python3 setup.py

# Install globally (creates 'coder' command)
python3 install.py

# Check installation
coder --help
```

### Basic Usage
```bash
# Interactive chat mode with file operations
coder qwen2.5-coder:3b mycoder chat

# Voice conversation mode
coder qwen2.5-coder:3b archie speech

# One-shot query
coder llama3.2:3b advisor chat --query "Create a Python web scraper"

# List available models
coder --list-models

# Reset context for a specific assistant
coder qwen2.5-coder:3b mycoder chat --reset
```

## Architecture

### New Modular System

**Multi-Modal Architecture**
- **BaseAssistant**: Abstract base class for all assistant types
- **ChatAssistant**: Text-based chat with optional file operations
- **SpeechAssistant**: Voice interaction with speech recognition and text-to-speech
- **ImageAssistant**: Image processing (placeholder for future)

**Assistant Factory (`assistant_base.py`)**
- Creates appropriate assistant based on mode (chat/speech/image)
- Manages model switching and configuration
- Handles context persistence per assistant instance

**Main CLI (`main.py`)**
- Unified command interface: `coder [model] [name] [mode] [options]`
- Supports multiple models and assistant personas
- Global installation via `install.py`

### Context Management Per Assistant
```
.ai_context/
├── assistant_name/
│   ├── context.json    # Conversation history
│   └── config.json     # Assistant configuration
└── other_assistant/
    ├── context.json
    └── config.json
```

### Legacy System

**Original Application (`coder.py`)**
- Single-model chat assistant with file operations
- Direct Ollama integration with qwen2.5-coder:3b
- Context stored in `./asst/`

## Core Features

### File Operations (Chat Mode)
```bash
# File operations via code blocks:
```create:filename.ext
content here
```

```update:filename.ext  
full new content
```

```read:filename.ext
```

```delete:filename.ext
```

```mkdir:dirname
```
```

### Assistant Management
- **Multiple Models**: Switch between any Ollama model
- **Named Assistants**: Create different personas (coder, advisor, designer)
- **Persistent Context**: Each assistant maintains separate conversation history
- **Working Directory**: File operations relative to specified directory

### Installation Options
- **User Installation**: `~/.local/bin/coder` (default)
- **System Installation**: `/usr/local/bin/coder` (with `--system` flag)
- **Development Mode**: Direct execution via `./venv/bin/python main.py`

## Project Structure
```
casista/
├── main.py              # New CLI wrapper
├── assistant_base.py    # Assistant framework
├── setup.py            # Environment setup
├── install.py          # Global installation
├── coder.py            # Legacy single assistant
├── .ai_context/        # Per-assistant contexts
│   └── assistant_name/
├── asst/              # Legacy context storage
└── venv/              # Python virtual environment
```

## Dependencies

### Core Dependencies
- **ollama**: AI model interface
- **rich**: Terminal formatting and UI components  
- **typer**: CLI framework
- **prompt_toolkit**: Enhanced input with history

### Speech Dependencies (Optional)
- **speechrecognition**: Speech-to-text conversion
- **pyttsx3**: Text-to-speech synthesis
- **pyaudio**: Audio I/O (may require system audio libraries)

**macOS audio setup:**
```bash
brew install portaudio
pip install speechrecognition pyttsx3 pyaudio
```

## Usage Patterns

### Development Workflow
```bash
# Start coding session
coder qwen2.5-coder:7b developer chat

# Design consultation  
coder llama3.2:3b designer chat --query "UI design for mobile app"

# Code review
coder qwen2.5-coder:3b reviewer chat --query "Review this Python file" --working-dir ./project
```

### File Operations
```bash
# Auto-confirm all file operations
coder qwen2.5-coder:3b coder chat --auto-confirm

# Disable file operations for safe chat
coder llama3.2:3b advisor chat --no-file-ops
```

### Voice Mode
```bash
# Start voice conversation
coder qwen2.5-coder:3b archie speech

# Voice commands:
# - Say "exit", "quit", "goodbye", or "bye" to end
# - Say "reset" to clear conversation history
# - Say "who am i" or "set user [name]" for identity management
```

**Voice Features:**
- **Speech Recognition**: Converts your voice to text using Google Speech API
- **Text-to-Speech**: AI responses are spoken using system TTS with voice selection
- **Same-Line Updates**: Progress indicators update on same line (no spam)
- **Ambient Noise Adjustment**: Automatically adapts to background noise
- **Timeout Handling**: 10-second listening window per interaction
- **Voice Selection**: Choose from your downloaded system voices
- **Speed Control**: Voice commands to adjust speech rate

**Voice Selection Tool:**
```bash
# List all available system voices
python3 voice_selector.py list

# Test a specific voice (by ID from list)
python3 voice_selector.py test 15

# Set voice for an assistant
python3 voice_selector.py set anna 15
```

**Voice Commands During Conversation:**
- "What voice are you using?" → Shows current voice
- "Speak slower" / "Speak faster" → Adjusts speech rate  
- "Change voice" → Instructions for voice selection