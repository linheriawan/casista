#!/usr/bin/env python3
"""
Main CLI wrapper for the multi-modal AI assistant system
Usage: python main.py [model] [assistant-name] [chat|speech|image] [options]
"""

import os
import sys
import argparse
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
    """Special interactive mode for speech assistant"""
    from assistant_base import SpeechAssistant
    
    if not isinstance(assistant, SpeechAssistant):
        console.print("[red]❌ Not a speech assistant[/]")
        return
    
    console.print("\n[bold green]🎤 Voice Mode Active[/]")
    console.print("[dim]Say 'exit' or press Ctrl+C to quit[/]")
    console.print("[dim]Say 'reset' to clear conversation history[/]")
    
    # Initial greeting
    greeting = f"Hello {assistant.get_user_name()}! I'm {assistant.name}. How can I help you today?"
    assistant.speak(greeting)
    
    while True:
        try:
            print(f"\n👂 {assistant.name} is listening...", end="", flush=True)
            
            # Listen for voice input
            text = assistant.listen()
            
            if not text:
                continue
            
            # Check for exit commands
            if text.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                farewell = f"Goodbye {assistant.get_user_name()}! It was nice talking with you."
                assistant.speak(farewell)
                break
            
            # Handle special commands
            if text.lower() == 'reset':
                assistant.reset_context()
                assistant.speak("I've cleared our conversation history.")
                continue
            
            # Process the voice input
            assistant.process_input(text, auto_confirm=auto_confirm)
            
        except KeyboardInterrupt:
            farewell = f"Goodbye {assistant.get_user_name()}!"
            console.print(f"\n[dim]Interrupted by user[/]")
            assistant.speak(farewell)
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")
            assistant.speak("I'm sorry, I encountered an error. Let's try again.")

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
    
    args = parser.parse_args()
    
    # Handle list models
    if args.list_models:
        list_available_models()
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