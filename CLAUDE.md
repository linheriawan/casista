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

### Modular System (Refactored 2024)

**Architecture Overview**
- **Modular Design**: Separated concerns into `library/`, `helper/`, and `configuration/` modules
- **TOML Configuration**: All configuration moved from Python to TOML files
- **Assistant Management**: Comprehensive CLI for creating, configuring, and managing assistants
- **RAG Knowledge**: Global knowledge base system with .ragfile vector storage
- **Session Management**: Working directory handling with per-session overrides

**Core Components**
- **library/**: Core functionality (coding, conversation, image generation, configuration loading)
- **helper/**: Management utilities (agents, models, voice, RAG knowledge)
- **configuration/**: TOML configuration files (models, personalities, prompts)
- **main.py**: Unified CLI interface with comprehensive management commands

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

### Current Refactored Structure
```
casista/
├── main.py                     # Unified CLI interface
├── setup.py                   # Environment setup
├── install.py                 # Global installation
├── library/                   # Core functionality modules
│   ├── assistant_cfg.py       # Complete assistant configuration
│   ├── config_loader.py       # Base TOML configuration loader
│   ├── model_cfg.py           # Model configuration management
│   ├── personality_cfg.py     # Personality management
│   ├── prompt_cfg.py          # Prompt template system
│   ├── coding/                # Code generation and analysis
│   │   ├── code_gen.py        # Code generation utilities
│   │   └── file_ops.py        # File operation parsing/execution
│   ├── conversation/          # Chat and dialogue management
│   │   ├── chat_manager.py    # Chat session management
│   │   ├── context.py         # Conversation context handling
│   │   └── ollama_client.py   # Ollama API integration
│   └── image_generation/      # Image processing capabilities
│       ├── generation.py      # Image generation core
│       └── models.py          # Image model management
├── helper/                    # Management utilities
│   ├── manage_agent.py        # Agent creation and configuration
│   ├── manage_voice.py        # Voice and speech management
│   ├── manage_model.py        # AI model management
│   └── rag_knowledge.py       # RAG knowledge management
├── configuration/             # TOML configuration files
│   ├── default.model.toml     # Model definitions
│   ├── default.personality.toml # Personality configurations
│   └── default.prompt.toml    # Prompt templates
├── knowledge/                 # Global RAG knowledge base (.ragfile storage)
├── .ai_context/              # Per-assistant contexts
│   └── [assistant_name]/     # Individual assistant data
├── requirements.txt
└── venv/                     # Python virtual environment
```

### Cleaned Up Files (Removed in Refactor)
- `assistant_base.py` - Legacy monolithic assistant system
- `coder.py` - Legacy single assistant implementation
- `coder_shortcuts.py` - Legacy shortcuts
- `rag_system.py` - Replaced by `helper/rag_knowledge.py`
- `voice_selector.py` - Replaced by `helper/manage_voice.py`
- `set_voice_personality.py` - Legacy utility, functionality moved to CLI
- Test files: `test_image_generation.py`, `create_test_*.py`

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