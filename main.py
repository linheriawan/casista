#!/usr/bin/env python3
"""
Main CLI wrapper for the multi-modal AI assistant system
Usage: python main.py [model] [assistant-name] [chat|speech|image] [options]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional
import ollama
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from assistant_base import create_assistant, BaseAssistant

console = Console()

def list_available_models():
    """List available Ollama models"""
    try:
        models = ollama.list()
        if not models.get('models'):
            console.print("[yellow]⚠️ No Ollama models found. Install models first.[/]")
            return []
        
        table = Table(title="Available Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="dim")
        
        for model in models['models']:
            name = getattr(model, 'model', 'Unknown')
            size_bytes = getattr(model, 'size', 0)
            size = f"{size_bytes / (1024**3):.1f}GB" if size_bytes else 'Unknown'
            modified_at = getattr(model, 'modified_at', None)
            if modified_at:
                modified = str(modified_at)[:19]
            else:
                modified = 'Unknown'
            table.add_row(name, size, modified)
        
        console.print(table)
        return [getattr(m, 'model', '') for m in models['models']]
    except Exception as e:
        console.print(f"[red]❌ Error listing models: {e}[/]")
        return []

def interactive_mode(assistant: BaseAssistant, auto_confirm: bool = False):
    """Run assistant in interactive mode"""
    user_name = assistant.get_user_name()
    mode_name = assistant.__class__.__name__.replace('Assistant', '').lower()
    
    console.print(Panel.fit(
        f"[bold cyan]{assistant.name}[/] - [dim]{assistant.model}[/]\n"
        f"User: [green]{user_name}[/] | Mode: [green]{mode_name}[/]\n"
        f"Directory: [dim]{assistant.working_dir}[/]\n"
        f"Commands: 'exit', 'reset', 'config', 'who am i', 'set user [name]'",
        title="🤖 AI Assistant"
    ))
    
    # Special handling for speech mode
    if mode_name == 'speech':
        speech_interactive_mode(assistant, auto_confirm)
        return
    
    # Regular chat mode
    while True:
        try:
            # Update user name in case it changed
            current_user = assistant.get_user_name()
            user_input = input(f"\n💬 [{current_user}]: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print(f"[dim]Goodbye from {assistant.name}! 👋[/]")
                break
            
            if user_input.lower() == 'reset':
                assistant.reset_context()
                continue
            
            if user_input.lower() == 'config':
                show_config(assistant)
                continue
            
            if not user_input:
                continue
            
            # Process input through the assistant
            assistant.process_input(user_input, auto_confirm=auto_confirm)
            
        except KeyboardInterrupt:
            console.print(f"\n[dim]Goodbye from {assistant.name}! 👋[/]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")

def speech_interactive_mode(assistant, auto_confirm: bool = False):
    """Special interactive mode for speech assistant with htop-like UI"""
    from assistant_base import SpeechAssistant
    
    if not isinstance(assistant, SpeechAssistant):
        console.print("[red]❌ Not a speech assistant[/]")
        return
    
    # Initial greeting setup
    assistant._add_chat_line("🎤 Voice Mode Active", "bold green")
    assistant._add_chat_line("Say 'exit' or press Ctrl+C to quit", "dim")
    assistant._add_chat_line("Say 'reset' to clear conversation history", "dim")
    
    # Initial greeting
    greeting = f"Hello {assistant.get_user_name()}! I'm {assistant.name}. How can I help you today?"
    assistant._add_chat_line(f"🤖 {assistant.name}: {greeting}", "cyan")
    assistant._update_status("🔊 Initial greeting...")
    assistant.speak(greeting)
    
    while True:
        try:
            assistant._update_status("🎤 Ready - say something...")
            
            # Use the new UI-aware process_input method
            response = assistant.process_input(auto_confirm=auto_confirm)
            
            # The process_input method handles the UI updates now
            if response:
                # Check for exit commands within the response handling
                last_user_input = ""
                if assistant.chat_lines:
                    for line in reversed(assistant.chat_lines):
                        if line.plain.startswith("👤 You:"):
                            last_user_input = line.plain[7:].strip().lower()
                            break
                
                if last_user_input in ['exit', 'quit', 'goodbye', 'bye']:
                    farewell = f"Goodbye {assistant.get_user_name()}! It was nice talking with you."
                    assistant._add_chat_line(f"🤖 {assistant.name}: {farewell}", "cyan")
                    assistant._update_status("🔊 Saying goodbye...")
                    assistant.speak(farewell)
                    break
                
                if last_user_input == 'reset':
                    assistant.reset_context()
                    assistant._add_chat_line("✅ Conversation history cleared", "yellow")
                    assistant.speak("Conversation history cleared.")
                    continue
            
        except KeyboardInterrupt:
            print("\n")
            console.print("\n[yellow]👋 Speech mode ended[/]")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/]")
            assistant.speak("I encountered an error. Let me try again.")

def show_config(assistant: BaseAssistant):
    """Show assistant configuration"""
    config_table = Table(title=f"{assistant.name} Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    for key, value in assistant.config.items():
        if isinstance(value, list):
            value = ", ".join(value)
        config_table.add_row(key, str(value))
    
    console.print(config_table)

def list_voices():
    """List available TTS voices"""
    try:
        from voice_selector import list_available_voices
        list_available_voices()
    except ImportError:
        console.print("[red]❌ voice_selector.py not found[/]")

def set_voice(assistant_name: str, voice_id: str):
    """Set voice for an assistant"""
    try:
        from voice_selector import list_available_voices, set_assistant_voice
        voices = list_available_voices()
        
        try:
            voice_index = int(voice_id)
            if 0 <= voice_index < len(voices):
                voice = voices[voice_index]
                if set_assistant_voice(assistant_name, voice.id, voice.name):
                    console.print(f"[green]✅ Voice set successfully![/]")
                else:
                    console.print(f"[red]❌ Failed to set voice[/]")
            else:
                console.print(f"[red]❌ Invalid voice ID. Use 0-{len(voices)-1}[/]")
        except ValueError:
            console.print("[red]❌ Voice ID must be a number[/]")
            
    except ImportError:
        console.print("[red]❌ voice_selector.py not found[/]")

def show_models():
    """Show Ollama models and cache information"""
    try:
        import ollama
        import subprocess
        import json
        from pathlib import Path
        
        console.print("[bold cyan]🤖 Ollama Models[/]")
        
        # Get ollama models
        try:
            models = ollama.list()
            model_table = Table(title="Installed Models")
            model_table.add_column("Name", style="cyan")
            model_table.add_column("Size", style="green")
            model_table.add_column("Modified", style="dim")
            
            for model in models.get('models', []):
                name = model.get('name', 'Unknown')
                size = model.get('size', 0)
                # Convert size to human readable
                size_str = f"{size / 1024 / 1024 / 1024:.1f} GB" if size > 0 else "Unknown"
                modified = model.get('modified_at', 'Unknown')
                if 'T' in str(modified):
                    modified = str(modified).split('T')[0]  # Just date part
                
                model_table.add_row(name, size_str, str(modified))
            
            console.print(model_table)
            
        except Exception as e:
            console.print(f"[red]❌ Error getting Ollama models: {e}[/]")
        
        # Show HuggingFace cache info if available
        console.print(f"\n[bold cyan]💾 HuggingFace Cache[/]")
        
        hf_cache_dir = Path.home() / ".cache" / "huggingface"
        if hf_cache_dir.exists():
            try:
                # Calculate cache size
                total_size = 0
                model_count = 0
                
                for item in hf_cache_dir.rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
                        if item.suffix in ['.bin', '.safetensors', '.onnx']:
                            model_count += 1
                
                cache_size = f"{total_size / 1024 / 1024 / 1024:.1f} GB"
                
                cache_table = Table()
                cache_table.add_column("Location", style="cyan")
                cache_table.add_column("Size", style="green")
                cache_table.add_column("Files", style="yellow")
                
                cache_table.add_row(str(hf_cache_dir), cache_size, str(model_count))
                console.print(cache_table)
                
            except Exception as e:
                console.print(f"[yellow]⚠️ Could not read HF cache: {e}[/]")
        else:
            console.print("[dim]No HuggingFace cache found[/]")
            
    except ImportError:
        console.print("[red]❌ Ollama not installed[/]")

def list_speech_backends():
    """List available speech recognition backends"""
    console.print("[bold cyan]🎤 Speech Recognition Backends[/]")
    
    backend_table = Table(title="Available Backends")
    backend_table.add_column("Backend", style="cyan")
    backend_table.add_column("Type", style="green")
    backend_table.add_column("Status", style="yellow")
    backend_table.add_column("Notes", style="dim")
    
    # Check Google
    try:
        import speech_recognition as sr
        backend_table.add_row("google", "Online", "✅ Available", "Requires internet")
    except ImportError:
        backend_table.add_row("google", "Online", "❌ Missing", "pip install speechrecognition")
    
    # Check Whisper
    try:
        import whisper
        backend_table.add_row("whisper", "Local", "✅ Available", "Best accuracy, slower")
    except ImportError:
        backend_table.add_row("whisper", "Local", "❌ Missing", "pip install openai-whisper")
    
    # Check Vosk
    try:
        import vosk
        backend_table.add_row("vosk", "Local", "✅ Available", "Fast, needs model download")
    except ImportError:
        backend_table.add_row("vosk", "Local", "❌ Missing", "pip install vosk")
    
    console.print(backend_table)
    
    console.print("\n[bold yellow]💡 Usage:[/]")
    console.print("  --set-speech-backend [assistant] [backend]")
    console.print("  Example: --set-speech-backend jeany whisper")

def set_speech_backend(assistant_name: str, backend: str):
    """Set speech recognition backend for an assistant"""
    valid_backends = ['whisper', 'vosk','google']
    
    if backend not in valid_backends:
        console.print(f"[red]❌ Invalid backend. Choose from: {', '.join(valid_backends)}[/]")
        return
    
    context_dir = Path(".ai_context") / assistant_name
    config_file = context_dir / "config.json"
    
    if not config_file.exists():
        console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
        console.print(f"Create it first: python3 main.py qwen2.5-coder:3b {assistant_name} speech")
        return
    
    # Load and update config
    config = json.loads(config_file.read_text())
    config['speech_backend'] = backend
    
    # Add backend-specific settings
    if backend == 'whisper':
        config['whisper_model'] = config.get('whisper_model', 'base')
    elif backend == 'vosk':
        config['vosk_model_path'] = config.get('vosk_model_path', './vosk-model')
    
    # Save updated config
    config_file.write_text(json.dumps(config, indent=2))
    
    console.print(f"[green]✅ Set {assistant_name}'s speech backend to: {backend}[/]")
    
    # Show installation instructions if needed
    if backend == 'whisper':
        try:
            import whisper
        except ImportError:
            console.print("[yellow]💡 Install Whisper: pip install openai-whisper[/]")
    elif backend == 'vosk':
        try:
            import vosk
        except ImportError:
            console.print("[yellow]💡 Install Vosk: pip install vosk[/]")
            console.print("[dim]Download models from: https://alphacephei.com/vosk/models[/]")

def get_hf_cache_dir():
    """Get HuggingFace cache directory"""
    return Path.home() / ".cache" / "huggingface" / "hub"

def list_cached_image_models():
    """List all cached image generation models"""
    cache_dir = get_hf_cache_dir()
    
    if not cache_dir.exists():
        console.print("[yellow]⚠️ HuggingFace cache directory not found[/]")
        return []
    
    models = []
    
    # Look for model directories
    for item in cache_dir.iterdir():
        if item.is_dir() and ("stable-diffusion" in item.name.lower() or 
                              "tiny-sd" in item.name.lower() or
                              "openjourney" in item.name.lower() or
                              "dreamlike" in item.name.lower() or
                              "waifu" in item.name.lower()):
            
            # Calculate size
            total_size = 0
            try:
                for file in item.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
            except:
                total_size = 0
            
            models.append({
                'name': item.name,
                'path': item,
                'size': total_size
            })
    
    return models

def format_size(size_bytes):
    """Format bytes to human readable"""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def list_image_models():
    """List cached image generation models"""
    models = list_cached_image_models()
    
    if not models:
        console.print("[yellow]⚠️ No image generation models found in cache[/]")
        console.print("[dim]Models will be downloaded automatically when first used[/]")
        return
    
    table = Table(title="Cached Image Generation Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Status", style="yellow")
    
    for model in models:
        # Extract model name from full path name
        display_name = model['name']
        if 'models--' in display_name:
            parts = display_name.split('models--')[1].replace('--', '/')
            display_name = parts
        
        table.add_row(
            display_name,
            format_size(model['size']),
            "✅ Cached"
        )
    
    console.print(table)

def remove_image_model(model_name: str):
    """Remove image model from cache"""
    models = list_cached_image_models()
    
    # Find matching models
    matches = [m for m in models if model_name.lower() in m['name'].lower()]
    
    if not matches:
        console.print(f"[red]❌ No models found matching '{model_name}'[/]")
        return
    
    for model in matches:
        display_name = model['name']
        if 'models--' in display_name:
            parts = display_name.split('models--')[1].replace('--', '/')
            display_name = parts
            
        console.print(f"[yellow]Removing:[/] {display_name}")
        console.print(f"[dim]Size: {format_size(model['size'])}[/]")
        
        try:
            import shutil
            shutil.rmtree(model['path'])
            console.print(f"[green]✅ Removed {display_name} ({format_size(model['size'])} freed)[/]")
        except Exception as e:
            console.print(f"[red]❌ Failed to remove {display_name}: {e}[/]")

def preload_image_model(model_name: str):
    """Preload/download image model"""
    console.print(f"[cyan]📥 Downloading model: {model_name}[/]")
    
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        # Load the model (this will download and cache it)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True
        )
        
        console.print(f"[green]✅ Successfully downloaded {model_name}[/]")
        console.print(f"[dim]Model is now cached and ready for use[/]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download {model_name}: {e}[/]")

def set_image_models(assistant_name: str, models_str: str):
    """Set image model priority for assistant"""
    models = [m.strip() for m in models_str.split(',')]
    
    context_dir = Path(".ai_context") / assistant_name
    config_file = context_dir / "config.json"
    
    if not config_file.exists():
        console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
        console.print(f"Create it first: python3 main.py qwen2.5-coder:3b {assistant_name} image")
        return
    
    # Load and update config
    config = json.loads(config_file.read_text())
    config['image_models'] = models
    
    # Save updated config
    config_file.write_text(json.dumps(config, indent=2))
    
    console.print(f"[green]✅ Set {assistant_name}'s image model priority:[/]")
    for i, model in enumerate(models, 1):
        console.print(f"  {i}. {model}")
    
    console.print(f"\n[dim]Models will be tried in this order when generating images[/]")

def enable_rag(assistant_name: str):
    """Enable RAG for an assistant"""
    context_dir = Path(".ai_context") / assistant_name
    config_file = context_dir / "config.json"
    
    if not config_file.exists():
        console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
        console.print(f"Create it first: python3 main.py qwen2.5-coder:3b {assistant_name} chat")
        return
    
    # Load and update config
    config = json.loads(config_file.read_text())
    config['rag_enabled'] = True
    
    # Save updated config
    config_file.write_text(json.dumps(config, indent=2))
    
    console.print(f"[green]✅ RAG enabled for {assistant_name}[/]")
    console.print("[dim]Install dependencies if needed: pip install sentence-transformers numpy[/]")

def disable_rag(assistant_name: str):
    """Disable RAG for an assistant"""
    context_dir = Path(".ai_context") / assistant_name
    config_file = context_dir / "config.json"
    
    if not config_file.exists():
        console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
        return
    
    # Load and update config
    config = json.loads(config_file.read_text())
    config['rag_enabled'] = False
    
    # Save updated config
    config_file.write_text(json.dumps(config, indent=2))
    
    console.print(f"[green]✅ RAG disabled for {assistant_name}[/]")

def show_rag_status(assistant_name: str):
    """Show RAG status for an assistant"""
    context_dir = Path(".ai_context") / assistant_name
    config_file = context_dir / "config.json"
    
    if not config_file.exists():
        console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
        return
    
    config = json.loads(config_file.read_text())
    rag_enabled = config.get('rag_enabled', False)
    
    console.print(f"[bold cyan]📚 RAG Status for {assistant_name}[/]")
    console.print(f"Status: {'✅ Enabled' if rag_enabled else '❌ Disabled'}")
    
    if rag_enabled:
        # Try to get knowledge base stats
        try:
            from rag_system import RAGKnowledgeBase
            kb_dir = Path(".ai_context") / "knowledge"
            kb = RAGKnowledgeBase(kb_dir)
            stats = kb.get_stats()
            
            console.print(f"Documents: {stats['total_documents']}")
            console.print(f"Embedded: {stats['embedded_documents']}")
            console.print(f"Sources: {stats['unique_sources']}")
            console.print(f"Model: {stats['embedding_model']}")
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not load knowledge base: {e}[/]")

def index_project(assistant_name: str, project_path: str):
    """Index a project for RAG"""
    path = Path(project_path)
    if not path.exists():
        console.print(f"[red]❌ Project path not found: {project_path}[/]")
        return
    
    try:
        from rag_system import RAGKnowledgeBase
        
        kb_dir = Path(".ai_context") / "knowledge"
        kb = RAGKnowledgeBase(kb_dir)
        
        console.print(f"[cyan]📚 Indexing project for {assistant_name}...[/]")
        kb.index_codebase(path)
        
        stats = kb.get_stats()
        console.print(f"[green]✅ Indexing complete![/]")
        console.print(f"Total documents: {stats['total_documents']}")
        
    except ImportError:
        console.print("[red]❌ RAG dependencies not installed[/]")
        console.print("[dim]Install with: pip install sentence-transformers numpy[/]")
    except Exception as e:
        console.print(f"[red]❌ Indexing failed: {e}[/]")

def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s qwen2.5-coder:3b coder chat                    # Text chat with file ops
  %(prog)s llama3.2:3b advisor chat --query "Help me"     # One-shot query
  %(prog)s qwen2.5-coder:7b designer image               # Image mode (future)
  %(prog)s --list-models                                  # List available models
        """
    )
    
    # Main arguments
    parser.add_argument("model", nargs="?", help="Ollama model to use (e.g., qwen2.5-coder:3b)")
    parser.add_argument("name", nargs="?", help="Assistant name/persona")
    parser.add_argument("mode", nargs="?", choices=["chat", "speech", "image"], 
                       default="chat", help="Assistant mode (default: chat)")
    
    # Options
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    parser.add_argument("--query", "-q", help="One-shot query instead of interactive mode")
    parser.add_argument("--auto-confirm", "-y", action="store_true", help="Auto-confirm file operations")
    parser.add_argument("--reset", action="store_true", help="Reset assistant context")
    parser.add_argument("--working-dir", "-w", type=Path, help="Working directory (default: current)")
    parser.add_argument("--config", action="store_true", help="Show assistant configuration")
    parser.add_argument("--no-file-ops", action="store_true", help="Disable file operations for chat mode")
    
    # Voice and model commands
    parser.add_argument("--set-voice", nargs=2, metavar=("ASSISTANT", "VOICE_ID"), help="Set voice for assistant")
    parser.add_argument("--list-voices", action="store_true", help="List available TTS voices")
    parser.add_argument("--show-models", action="store_true", help="Show Ollama models and cache info")
    parser.add_argument("--set-speech-backend", nargs=2, metavar=("ASSISTANT", "BACKEND"), 
                       help="Set speech recognition backend (google/whisper/vosk)")
    parser.add_argument("--list-speech-backends", action="store_true", help="List available speech recognition backends")
    
    # Image model commands
    parser.add_argument("--list-image-models", action="store_true", help="List cached image generation models")
    parser.add_argument("--remove-image-model", metavar="MODEL_NAME", help="Remove image model from cache")
    parser.add_argument("--preload-image-model", metavar="MODEL_NAME", help="Preload/download image model")
    parser.add_argument("--set-image-models", nargs=2, metavar=("ASSISTANT", "MODELS"), 
                       help="Set model priority for assistant (comma-separated list)")
    
    # RAG commands
    parser.add_argument("--enable-rag", metavar="ASSISTANT", help="Enable RAG for assistant")
    parser.add_argument("--disable-rag", metavar="ASSISTANT", help="Disable RAG for assistant")
    parser.add_argument("--rag-status", metavar="ASSISTANT", help="Show RAG status for assistant")
    parser.add_argument("--index-project", nargs=2, metavar=("ASSISTANT", "PATH"), help="Index project for RAG")
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_models:
        list_available_models()
        return
    
    if args.list_voices:
        list_voices()
        return
    
    if args.show_models:
        show_models()
        return
    
    if args.set_voice:
        assistant_name, voice_id = args.set_voice
        set_voice(assistant_name, voice_id)
        return
    
    if args.list_speech_backends:
        list_speech_backends()
        return
    
    if args.set_speech_backend:
        assistant_name, backend = args.set_speech_backend
        set_speech_backend(assistant_name, backend)
        return
    
    if args.list_image_models:
        list_image_models()
        return
    
    if args.remove_image_model:
        remove_image_model(args.remove_image_model)
        return
    
    if args.preload_image_model:
        preload_image_model(args.preload_image_model)
        return
    
    if args.set_image_models:
        assistant_name, models = args.set_image_models
        set_image_models(assistant_name, models)
        return
    
    # RAG commands
    if args.enable_rag:
        enable_rag(args.enable_rag)
        return
    
    if args.disable_rag:
        disable_rag(args.disable_rag)
        return
    
    if args.rag_status:
        show_rag_status(args.rag_status)
        return
    
    if args.index_project:
        assistant_name, project_path = args.index_project
        index_project(assistant_name, project_path)
        return
    
    # Validate required arguments
    if not args.model or not args.name:
        console.print("[red]❌ Error: model and name are required[/]")
        console.print("Example: python main.py qwen2.5-coder:3b coder chat")
        console.print("Use --list-models to see available models")
        parser.print_help()
        return
    
    # Set working directory
    working_dir = args.working_dir or Path.cwd()
    if not working_dir.exists():
        console.print(f"[red]❌ Working directory does not exist: {working_dir}[/]")
        return
    
    try:
        # Create assistant
        file_ops = not args.no_file_ops if args.mode == "chat" else False
        if args.mode == "chat":
            from assistant_base import ChatAssistant
            assistant = ChatAssistant(args.model, args.name, working_dir, file_ops)
        else:
            assistant = create_assistant(args.model, args.name, args.mode, working_dir)
        
        # Handle special commands
        if args.reset:
            assistant.reset_context()
            return
        
        if args.config:
            show_config(assistant)
            return
        
        # Run in appropriate mode
        if args.query:
            # One-shot mode
            console.print(f"[dim]Using {args.model} as {args.name} in {working_dir}[/]")
            assistant.process_input(args.query, auto_confirm=args.auto_confirm)
        else:
            # Interactive mode
            interactive_mode(assistant, args.auto_confirm)
    
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted by user[/]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/]")
        console.print("[yellow]💡 Make sure Ollama is running and the model is installed[/]")

if __name__ == "__main__":
    main()
