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

### Current Structure
```
casista/
├── main.py              # New CLI wrapper
├── assistant_base.py    # Assistant framework
├── setup.py            # Environment setup
├── install.py          # Global installation
├── coder.py            # Legacy single assistant
├── rag_system.py       # RAG implementation
├── voice_selector.py   # Voice management utilities
├── .ai_context/        # Per-assistant contexts
│   └── assistant_name/
├── asst/              # Legacy context storage
└── venv/              # Python virtual environment
```

### Planned Enhanced Structure (from design.md)
```
casista/
├── main.py    
├── library/                    # Core functionality modules
│   ├── coding/                # Code generation and analysis
│   ├── conversation/          # Chat and dialogue management
│   └── image_generation/      # Image processing capabilities
├── configuration/             # System configuration
│   ├── model_traits/         # System prompts, personalities
│   └── system_config/        # Model directory, settings
├── helper/                   # Management utilities
│   ├── manage_agent/         # Agent creation and configuration
│   ├── manage_voice/         # Voice and speech management
│   ├── manage_model/         # AI model management  
│   └── rag/                  # RAG knowledge management
├── requirements.txt
├── setup.py
└── install.py
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
```

## RAG (Retrieval-Augmented Generation) Features

### Document Processing
- Support for processing various document types including .docx, .pdf, and more
- Use the `--rag` or `--document` flag to enable RAG mode with specific documents

### Testing RAG
- Syntax: `coder [model] [name] chat --rag /path/to/document.pdf`
- Examples:
  - Test RAG with a PDF document
  - Test RAG with a Word document (.docx)
  - Combine multiple documents in a single RAG query

### Supported Document Types
- PDF files
- Microsoft Word documents (.docx)
- Text files (.txt)
- Markdown files (.md)
- CSV files (.csv)
- Basic spreadsheet formats

### RAG Usage Examples
```bash
# RAG with a PDF document
coder qwen2.5-coder:3b researcher chat --rag /docs/research_paper.pdf

# RAG with multiple documents
coder llama3.2:3b analyst chat --rag /reports/q1_report.docx --rag /reports/q2_report.pdf

# Specific document query
coder qwen2.5-coder:3b helper chat --query "Summarize key points" --rag /documents/overview.docx
```

### RAG Performance Optimization
- Intelligent document chunking
- Semantic search for relevant document sections
- Caching of processed document embeddings
- Adjustable context window for document processing

## TODO: Planned Enhancements

### Architecture Refactoring
- [ ] **Modularize Core Functionality**
  - [ ] Create `library/coding/` module for code generation and analysis
  - [ ] Create `library/conversation/` module for chat management
  - [ ] Create `library/image_generation/` module for image processing
  - [ ] Refactor existing code into modular structure

- [ ] **Configuration Management**
  - [ ] Implement `configuration/model_traits/` for system prompts and personalities
  - [ ] Create `configuration/system_config/` for model directory and settings
  - [ ] Centralize configuration loading and validation

### Helper Utilities Enhancement
- [ ] **Agent Management System**
  - [ ] `helper/manage_agent/create_agent` - Agent creation wizard
  - [ ] `helper/manage_agent/set_agent_model` - Bind agent to AI model
  - [ ] `helper/manage_agent/set_agent_voice` - Configure agent voice
  - [ ] `helper/manage_agent/set_agent_rag` - Configure agent RAG knowledge

- [ ] **Voice Management System**
  - [ ] `helper/manage_voice/list_tts_voices` - List local TTS voices
  - [ ] `helper/manage_voice/list_sr_models` - List Speech Recognition models
  - [ ] `helper/manage_voice/download_sr_model` - Download SR models
  - [ ] `helper/manage_voice/remove_sr_model` - Remove SR models

- [ ] **Model Management System**
  - [ ] `helper/manage_model/set_cache_dir` - Configure HuggingFace cache
  - [ ] `helper/manage_model/list_models` - Enhanced model listing
  - [ ] `helper/manage_model/download_model` - Model download utility
  - [ ] `helper/manage_model/remove_model` - Model removal utility

- [ ] **RAG Knowledge Management**
  - [ ] `helper/rag/create_knowledge` - Create RAG knowledge base
  - [ ] `helper/rag/list_knowledge` - List available knowledge bases
  - [ ] `helper/rag/update_knowledge` - Update existing knowledge
  - [ ] `helper/rag/remove_knowledge` - Remove knowledge bases

### Feature Enhancements
- [ ] **Image Generation Support**
  - [ ] Integrate with image generation models
  - [ ] Support for visual assistant modes
  - [ ] Image processing and analysis capabilities

- [ ] **Advanced Configuration**
  - [ ] Per-agent personality system
  - [ ] Custom system prompt management
  - [ ] Advanced model trait configuration

- [ ] **CLI Improvements**
  - [ ] Interactive setup wizard
  - [ ] Enhanced help system
  - [ ] Configuration validation and migration tools