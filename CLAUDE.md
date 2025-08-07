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

### Refactored Modular System (2024)

**Current Architecture Overview**
- **Clean Separation**: main.py (routing) + operation_handler.py (business logic) + dedicated handlers
- **TOML Configuration**: All configuration moved from Python to TOML files
- **Assistant Management**: Comprehensive CLI for creating, configuring, and managing assistants
- **RAG Knowledge**: Global knowledge base system with .ragfile vector storage
- **Unified Sessions**: Consolidated conversation handling with mode switching

**Current Core Components**
- **main.py** (294 lines): Pure routing - argument parsing, management commands, session routing
- **library/operation_handler.py** (722 lines): Session implementations and business logic
- **library/speech_handler.py** (272 lines): Dedicated TTS/STT management class
- **helper/**: Management utilities (agents, models, voice, RAG knowledge)
- **configuration/**: TOML configuration files (models, personalities, prompts, sys.definition)

**Current Main CLI (`main.py`) - Router Only**
- **Argument Parsing**: All `parser.add_argument()` definitions
- **Management Commands**: Direct handling of --create-agent, --list-models, etc.
- **Session Routing**: Calls `operation_handler.run_session()` for assistant interactions
- **Direct Operations**: Direct calls for `--prompt` (image generation) and `--setup`

**Operation Handler (`library/operation_handler.py`) - Unified Session Management**
- **run_session()**: Main session orchestrator with component preparation
- **prepare()**: Centralized component setup returning all session resources
- **_run_interactive_loop()**: Unified conversation handling for all modes (chat/speech/image)
- **_display_session_info()**: Component-based session information display
- **generate_single_image()**: One-shot image generation (text-only, no speech contamination)
- **handle_one_shot_query()**: Clean one-shot query handling (text-only, no speech contamination)
- **ResponseRenderer Integration**: All output uses unified rendering system

**Speech Handler (`library/speech_handler.py`) - TTS/STT Management**
- **Modular Speech**: Dedicated class for all speech functionality
- **Backend Support**: Google, Whisper, Vosk speech recognition backends
- **Voice Management**: TTS engine setup, voice selection, speech rate control
- **Enhanced Features**: Microphone calibration, noise handling, cleanup management

### Advanced Response Rendering Architecture

**Enhanced Session Flow (main.py → operation_handler.py → response_renderer.py):**
```python
# main.py - Pure Router
def main():
    parser = create_parser()  # All argument definitions here
    args = parser.parse_args()
    
    # Route to management commands or sessions
    if handle_management_commands(args):  # Management routing in main.py
        return
    
    if args.prompt and args.assistant_name:  # Direct image generation
        operation_handler.generate_single_image(args)
    if args.query and args.assistant_name:  # Direct one-shot query
        operation_handler.handle_one_shot_query(args)  # Text-only, no speech
    elif args.assistant_name:  # Interactive sessions
        operation_handler.run_session(args)

# operation_handler.py - Unified Session Management with Component Preparation
def run_session(args):
    # Prepare all components: session, model, speech, generator, chat, renderer
    components = self.prepare(args)
    # Display session info using components (not args.mode)
    self._display_session_info(components, args)
    # Run unified interactive loop for all modes
    return self._run_interactive_loop(components, args)

# Advanced rendering integration throughout
renderer = ResponseRenderer(console, speech_handler)
renderer.render_response(mode, assistant_name, response_data, style)
```

**Advanced Rendering Features:**
- **Unified Output**: All AI responses use ResponseRenderer for consistent display
- **Multiple Styles**: chat, table, panel, stream rendering modes
- **Speech Integration**: Unified TTS handling without scattered calls
- **Layout Zones**: htop-style window positioning for complex displays
- **Progress Rendering**: Real-time progress bars with line override
- **Stream Support**: Live updating displays for real-time responses
- **Context Preservation**: conversation history maintained across mode switches

**Direct Operations (No Interactive Session):**
- **One-shot Image**: `coder artist --prompt "cat"` → `generate_single_image(args)`
- **One-shot Query**: `coder mycoder --query "explain this code"` → `handle_one_shot_query(args)`
- **Setup Wizard**: `coder --setup` → `run_setup()`

**Clean Handler with Advanced Rendering:**
```python
# main.py - Direct calls (one-shot query now clean from speech)
if args.prompt and args.assistant_name:
    return operation_handler.generate_single_image(args)
if args.query and args.assistant_name:
    return operation_handler.handle_one_shot_query(args)  # Text-only

# operation_handler.py - Business logic with ResponseRenderer
def handle_one_shot_query(self, args):
    session = SessionManager(session_config)
    response_data = chat_manager.send_message(args.query)
    
    # Advanced rendering replaces scattered console.print calls
    renderer = ResponseRenderer(console)
    renderer.render_response("text", session.assistant_name, response_data, "chat")
    
# response_renderer.py - Advanced display capabilities
class ResponseRenderer:
    def render_response(mode, assistant_name, response_data, style):
        # Modes: "text", "speech", "text_and_speech" 
        # Styles: "chat", "table", "panel", "stream"
    def render_stream(assistant_name, stream_generator):
        # Real-time streaming with live updates
    def render_progress(task_name, progress, override_line=True):
        # htop-style progress bars
    def create_layout_zone(zone_name, start_line, height):
        # Window-style positioning for complex layouts
```

### Context Management Per Assistant
```
.ai_context/
├── assistant_name/
│   ├── context.json    # Conversation history
│   └── config.toml     # Assistant configuration
└── other_assistant/
    ├── context.json
    └── config.toml
```

## Core Features

### Advanced Response Rendering

The ResponseRenderer class provides sophisticated display capabilities:

**Rendering Styles:**
- **Chat Style**: Traditional `🤖 [assistant]: message` format
- **Table Style**: Rich tables for structured data display
- **Panel Style**: Bordered panels with titles for important information
- **Stream Style**: Real-time streaming with live updates

**Advanced Features:**
- **Unified Speech**: Integrated TTS without scattered speech_handler.speak() calls
- **Progress Bars**: htop-style progress with line override capability
- **Layout Zones**: Window-style positioning for complex layouts
- **Error/Success/Warning**: Consistent status message formatting

**Example Usage:**
```python
# Replace scattered console.print calls
renderer = ResponseRenderer(console, speech_handler)
renderer.render_response("text_and_speech", assistant_name, response_data, "table")

# Advanced features
renderer.render_progress("Loading model", 75, override_line=True)
zone = renderer.create_layout_zone("status", 1, 3)
renderer.render_in_zone("status", "System ready", clear_zone=True)
```

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
├── main.py                     # CLI Router (294 lines) - args parsing & routing only
├── setup.py                   # Environment setup
├── install.py                 # Global installation
├── library/                   # Core functionality modules
│   ├── operation_handler.py   # Unified Session Logic (633 lines)
│   ├── response_renderer.py   # Advanced Response Rendering (307 lines)
│   ├── speech_handler.py      # TTS/STT Management (272 lines)
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
├── test/                     # testing/POC of some new idea
├── .ai_context/              # Per-assistant contexts
│   └── [assistant_name]/     # Individual assistant data
├── requirements.txt
└── venv/                     # Python virtual environment
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
coder developer chat

# Design consultation with one-shot query
coder designer chat --query "UI design for mobile app"

# Code review with working directory
coder reviewer chat --query "Review this Python file" --working-dir ./project
```

### Unified Session Mode Switching
```bash
# Start in chat mode, switch dynamically within session
coder mycoder chat
# Within session commands:
# /mode=speech    # Switch to voice mode
# /mode=image     # Switch to image generation  
# /mode=chat      # Switch back to chat
# /dir=/new/path  # Change working directory
# reset           # Clear conversation history
```

### File Operations
```bash
# Auto-confirm all file operations
coder coder chat --auto-confirm

# Disable file operations for safe chat
coder advisor chat --no-file-ops
```

### Voice Mode
```bash
# Direct voice conversation
coder archie speech

# Or switch within session
coder archie chat
# Then: /mode=speech

# Voice commands:
# - Say "exit", "quit", "goodbye", or "bye" to end
# - Say "reset" to clear conversation history  
# - Say "/mode=chat" or "/mode=image" to switch modes
```

### Image Generation
```bash
# Direct one-shot image generation
coder artist image --prompt "A cyberpunk cityscape at sunset"

# Interactive image studio
coder artist image
# Commands: 'list models', 'switch model <name>', mode switching

# Or switch within session
coder artist chat
# Then: /mode=image
```

**Voice Features:**
- **Speech Recognition**: Converts your voice to text using Google Speech API
- **Text-to-Speech**: AI responses are spoken using system TTS with voice selection
- **Same-Line Updates**: Progress indicators update on same line (no spam)
- **Ambient Noise Adjustment**: Automatically adapts to background noise
- **Timeout Handling**: 10-second listening window per interaction
- **Voice Selection**: Choose from your downloaded system voices
- **Speed Control**: Voice commands to adjust speech rate

**Voice Commands During Conversation:**
- "What voice are you using?" → Shows current voice
- "Speak slower" / "Speak faster" → Adjusts speech rate  
- "Change voice" → Instructions for voice selection

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

## Image Generation Mode

### How to Use Image Generation Mode
- Activate image generation mode using the `--image` or `--gen-image` flag
- Specify the desired image model (if multiple are available)
- Provide a detailed text prompt describing the image
- Optionally set image parameters like size, style, or resolution

### Image Generation Examples
```bash
# Basic image generation
coder stable-diffusion:1.5 artist gen-image --prompt "A serene mountain landscape at sunset"

# Specify image dimensions
coder stable-diffusion:1.5 artist gen-image --prompt "A futuristic cityscape" --width 1024 --height 768

# Choose a specific style
coder stable-diffusion:1.5 artist gen-image --prompt "Portrait of a cyberpunk character" --style "digital art"
```

### Supported Image Generation Features
- Multiple image generation models
- Customizable image dimensions
- Style and aesthetic control
- Seed-based reproducibility
- Format selection (PNG, JPEG, etc.)

## System Architecture & Best Practices

### ResponseRenderer - Unified Output System

**CRITICAL: Always Use ResponseRenderer Instead of console.print**

The ResponseRenderer class provides unified, consistent output throughout the codebase with integrated speech support and advanced formatting capabilities.

**Why ResponseRenderer?**
- ✅ **Unified Speech Integration**: Automatic TTS in speech mode
- ✅ **Consistent Formatting**: Standardized styling across all output
- ✅ **Advanced Layouts**: Tables, panels, progress bars, streaming
- ✅ **Mode Awareness**: Context-aware rendering based on current mode
- ✅ **Error Handling**: Structured error/warning/success messages
- ✅ **No Scattered Calls**: Eliminates console.print spread throughout code

**Usage Patterns:**

```python
# ALWAYS instantiate ResponseRenderer first
renderer = ResponseRenderer(console, speech_handler)

# Replace console.print with appropriate methods:

# OLD: console.print("[red]❌ Error occurred[/]")
# NEW: renderer.render_error("Error occurred")

# OLD: console.print("[green]✅ Success![/]") 
# NEW: renderer.render_success("Success!")

# OLD: console.print("[yellow]⚠️ Warning[/]")
# NEW: renderer.render_warning("Warning")

# AI responses - unified speech integration
renderer.render_response("text_and_speech", assistant_name, response_data, "chat")

# System information as tables
system_info = {
    "Model": "qwen2.5-coder:3b",
    "Temperature": "0.7",
    "Working Directory": "/home/user/project"
}
renderer.render_system_info(system_info, style="table")

# Progress bars with override capability
renderer.render_progress("Loading model", 75, override_line=True)

# Advanced layout zones for complex displays
zone = renderer.create_layout_zone("status", 1, 3)
renderer.render_in_zone("status", "System ready", clear_zone=True)
```

**Available Methods:**
- `render_response()` - AI responses with speech integration
- `render_error()` - Error messages with consistent formatting
- `render_success()` - Success messages
- `render_warning()` - Warning messages  
- `render_system_info()` - Structured data (tables/panels)
- `render_stream()` - Real-time streaming with live updates
- `render_progress()` - Progress bars with htop-style line override
- `render_table()` - Rich tables for structured data
- `create_layout_zone()` - Window-style positioning
- `render_in_zone()` - Content in specific layout zones

**Rendering Styles:**
- **"chat"** - Traditional `🤖 [assistant]: message` format
- **"table"** - Rich tables with borders and styling
- **"panel"** - Bordered panels with titles for important info
- **"stream"** - Real-time streaming with live updates

**Speech Integration:**
```python
# Automatic speech in speech mode - no manual speech_handler.speak() calls
mode = "text_and_speech" if current_mode == "speech" else "text"
renderer.render_response(mode, assistant_name, response_data, "chat")

# ResponseRenderer handles all TTS automatically based on mode
```

**File-Specific Usage:**

**library/operation_handler.py:**
- All session management, error handling, user feedback
- System information display (session info, model details)
- Success/error reporting for image generation
- Warning messages for fallback scenarios

**main.py:**
- Management command results (agent creation, model listing)
- Setup wizard information display
- Version information and system status

**helper/ modules:**
- Agent management status reporting
- Model download progress
- Voice configuration feedback
- RAG indexing status

**NEVER Use These Patterns:**
```python
# ❌ NEVER: Direct console.print calls
console.print("[red]Error[/]")
console.print(f"[green]Success: {result}[/]")

# ❌ NEVER: Scattered speech_handler.speak() calls
speech_handler.speak(response)

# ❌ NEVER: Raw text output without structure
print(f"Model: {model}, Temp: {temp}")
```

**ALWAYS Use These Patterns:**
```python
# ✅ ALWAYS: ResponseRenderer methods
renderer = ResponseRenderer(console, speech_handler)
renderer.render_error("Error occurred")
renderer.render_success("Operation completed", details)

# ✅ ALWAYS: Structured data display
info = {"Model": model, "Temperature": temp}
renderer.render_system_info(info, style="table")

# ✅ ALWAYS: Unified response handling
renderer.render_response(mode, assistant_name, response_data, style)
```

**Implementation in New Code:**
1. **Import ResponseRenderer**: `from library.response_renderer import ResponseRenderer`
2. **Instantiate Early**: Create renderer instance at method start
3. **Replace All Output**: Use appropriate render methods instead of console.print
4. **Structure Data**: Use dictionaries for system info, tables, panels
5. **Integrate Speech**: Use mode-aware rendering for automatic TTS

### Component-Based Architecture

**Centralized Component Preparation:**

The `prepare()` method in OperationHandler centralizes all component setup and returns a unified dictionary:

```python
def prepare(self, args) -> Dict[str, Any]:
    """Prepare all session components and configuration."""
    # Initialize core components
    session = SessionManager(session_config)
    model_info = session.get_model_info()
    chat_manager = session.get_chat_manager()
    file_ops = session.get_file_operations()
    
    # Initialize mode-specific components
    speech_handler = None
    if args.mode == "speech":
        voice_config = session.get_voice_config()
        speech_handler = SpeechHandler(voice_config)
        # Setup and validation handled here
    
    generator = None
    if args.mode == "image":
        generator = session.get_image_generator()
    
    # Create unified renderer with speech integration
    renderer = ResponseRenderer(console, speech_handler)
    
    # Determine mode info (no args.mode checking in display methods)
    mode_name = "Image Generation Mode" if args.mode == "image" else \
                "Speech Mode" if speech_handler else "Chat Mode"
    mode_icon = "🎨" if args.mode == "image" else \
                "🎤" if speech_handler else "🤖"
    
    return {
        'session': session,
        'model_info': model_info,
        'speech_handler': speech_handler,
        'generator': generator,
        'chat_manager': chat_manager,
        'file_ops': file_ops,
        'user_name': session.get_user_name(),
        'renderer': renderer,
        'current_mode': args.mode,
        'mode_name': mode_name,
        'mode_icon': mode_icon,
        'last_generated_image': None,
        'current_image_model': None
    }
```

**Component Usage Benefits:**
- ✅ **Single Source of Truth**: All components prepared once, used everywhere
- ✅ **No args.mode Checking**: Display methods use prepared components  
- ✅ **Unified Speech Integration**: Speech handler passed to renderer automatically
- ✅ **Mode Consistency**: Mode names/icons calculated once, used consistently
- ✅ **Clean Separation**: Preparation logic separate from execution logic
- ✅ **Easy Testing**: Components can be mocked/injected for testing

**Usage in Methods:**
```python
def _display_session_info(self, components: Dict[str, Any], args) -> None:
    """Display session information using prepared components."""
    session = components['session']
    renderer = components['renderer'] 
    mode_name = components['mode_name']  # No args.mode checking!
    
    # Use renderer for all output
    system_info = {
        "Assistant": session.assistant_name,
        "Mode": mode_name,
        "Model": components['model_info']['model']
    }
    renderer.render_system_info(system_info, style="table")

def _run_interactive_loop(self, components: Dict[str, Any], args) -> bool:
    """Unified conversation loop using prepared components."""
    # Extract all needed components
    session = components['session']
    chat_manager = components['chat_manager']
    renderer = components['renderer']
    current_mode = components['current_mode']
    
    # Use components throughout - no direct component creation
    while True:
        # Handle user input, mode switching, conversations
        # All output through renderer
        renderer.render_response(mode, session.assistant_name, response_data)
```

### System Prompt & Context Management

**System Prompt Handling Best Practices:**
- **Set Once**: System prompt established at conversation start
- **Not Displayed**: Internal instruction, never shown to user  
- **Persistent**: Stays active throughout entire conversation
- **Role Establishment**: Model knows its role/personality/capabilities

**Implementation:**
```python
# System prompt automatically managed by ChatManager
messages = context_manager.ensure_system_prompt(system_prompt)
# Result: [{"role": "system", "content": "You are Anna, a teacher..."}, ...]
```

### Context.json - Conversation Memory

**Structure:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Anna, a patient teacher...",
      "timestamp": "2025-08-01T..."
    },
    {
      "role": "user", 
      "content": "What's machine learning?",
      "timestamp": "2025-08-01T..."
    },
    {
      "role": "assistant",
      "content": "Machine learning is...",
      "parsed_sections": {
        "reasoning": "I should explain this step by step...",
        "clean_answer": "Machine learning is..."
      },
      "timestamp": "2025-08-01T..."
    }
  ],
  "assistant_name": "anna",
  "last_updated": "2025-08-01T..."
}
```

**Benefits:**
- **Session Continuity**: Resume conversations after app restart
- **Model Memory**: AI remembers previous exchanges
- **Context Awareness**: Model can refer to earlier topics
- **Learning**: Model adapts to user's communication style

### Configuration Model Types

**Two Different Model Types:**

1. **Chat/Conversation Model**
```toml
[assistant]
model = "qwen3:4b"  # Ollama model for chat, reasoning, conversation
```
- Powers chat conversations, reasoning, prompt enhancement
- Type: Ollama LLM models (qwen, llama, etc.)
- Used in: Chat mode, Speech mode, AI-assisted image prompting

2. **Image Generation Models**
```toml
[image]
models = ["dreamlike-art/dreamlike-anime-1.0", "hakurei/waifu-diffusion"]  # HuggingFace Diffusion models
```
- Generate actual images from text prompts
- Type: HuggingFace Diffusion models (Stable Diffusion, SDXL, etc.)
- Used in: Image mode generation, img2img, upscaling

**In Image Mode:**
1. **Chat Model** (`qwen3:4b`) enhances your prompt: "cat" → "a photorealistic cat with detailed fur, high quality, 8k"
2. **Image Model** (`stable-diffusion-v1-5`) generates the actual image from enhanced prompt
3. **Both models work together** for better results!

### Capability Control
```toml
[capabilities]
image_generation = true  # Single source of truth for enablement
```

### Advanced Speech Recognition Setup

**Local Speech Recognition Options:**

**OpenAI Whisper (Recommended for Privacy):**
```bash
pip install openai-whisper
# Set Whisper as backend
python3 main.py --set-speech-backend assistant_name whisper
```

**Model Sizes:**
- `tiny` - ~39 MB, fastest, lowest accuracy
- `base` - ~74 MB, good balance (default)
- `small` - ~244 MB, better accuracy
- `medium` - ~769 MB, great accuracy  
- `large` - ~1550 MB, best accuracy

**Vosk (Fast, Real-time):**
```bash
pip install vosk
# Download model
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
mv vosk-model-en-us-0.22 vosk-model
# Set Vosk as backend
python3 main.py --set-speech-backend assistant_name vosk
```

**Comparison:**
| Backend | Type | Accuracy | Speed | Size | Internet |
|---------|------|----------|-------|------|----------|
| Google | Online | ⭐⭐⭐⭐⭐ | Fast | 0MB | Required |
| Whisper | Local | ⭐⭐⭐⭐⭐ | Slow | 74MB-1.5GB | None |
| Vosk | Local | ⭐⭐⭐ | Fast | ~50MB | None |

**Privacy Benefits of Local Speech:**
- ✅ No data sent to Google/internet
- ✅ Works offline completely  
- ✅ Faster processing (no network delay)
- ✅ Better privacy/security
- ✅ No API rate limits
```