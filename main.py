#!/usr/bin/env python3
"""
Enhanced CLI wrapper for the multi-modal AI assistant system.

Usage: coder [assistant_name] [mode] [options]
       coder [management_command] [options]
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

# Import management utilities
from helper.manage_agent import AgentManager
from helper.manage_voice import VoiceManager
from helper.manage_model import ModelManager
from helper.rag_knowledge import RAGKnowledgeManager
from library.assistant_cfg import AssistantConfig

console = Console()


def create_parser():
    """Create argument parser with all commands."""
    parser = argparse.ArgumentParser(
        prog="coder",
        description="AI Assistant with RAG, Voice, and Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic Usage
  coder mycoder chat                                    # Start chat session
  coder mycoder chat --working-dir /projects/webapp     # Chat with specific working directory
  coder mycoder speech                                  # Start voice session
  coder mycoder image --prompt "A beautiful landscape"  # Generate image
  
  # Agent Management
  coder create-agent                                    # Interactive agent creation
  coder list-agents                                     # List all agents
  coder clone-agent mycoder webcoder                    # Clone agent
  coder delete-agent oldcoder                           # Delete agent
  
  # Knowledge Management
  coder index-knowledge ./docs python_docs             # Index directory to RAG file
  coder list-knowledge                                  # List RAG files
  coder set-rag mycoder python_docs,web_dev           # Set RAG files for agent
  
  # Configuration
  coder set-model mycoder qwen2.5-coder:7b            # Set model for agent
  coder set-personality mycoder creative               # Set personality for agent
  coder set-voice mycoder 5                           # Set voice for agent
  coder configure mycoder                              # Interactive configuration
        """
    )
    
    # Main assistant execution
    parser.add_argument("assistant_name", nargs="?", help="Assistant name to use")
    parser.add_argument("mode", nargs="?", choices=["chat", "speech", "image"], 
                       default="chat", help="Assistant mode (default: chat)")
    
    # Session options
    parser.add_argument("--working-dir", "-w", type=Path, help="Working directory for this session")
    parser.add_argument("--query", "-q", help="One-shot query instead of interactive mode")
    parser.add_argument("--auto-confirm", "-y", action="store_true", help="Auto-confirm file operations")
    parser.add_argument("--no-file-ops", action="store_true", help="Disable file operations")
    
    # Image generation options
    parser.add_argument("--prompt", help="Image generation prompt")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--model", help="Specific model to use for this session")
    
    # Management commands (these take precedence over assistant execution)
    management = parser.add_argument_group("Management Commands")
    
    # Agent management
    management.add_argument("--create-agent", action="store_true", help="Create new agent interactively")
    management.add_argument("--list-agents", action="store_true", help="List all agents")
    management.add_argument("--show-agent", metavar="AGENT", help="Show agent information")
    management.add_argument("--clone-agent", nargs=2, metavar=("SOURCE", "TARGET"), help="Clone agent")
    management.add_argument("--delete-agent", metavar="AGENT", help="Delete agent")
    management.add_argument("--configure", metavar="AGENT", help="Configure agent interactively")
    
    # Model management  
    management.add_argument("--list-models", action="store_true", help="List available models")
    management.add_argument("--download-model", metavar="MODEL", help="Download/pull model")
    management.add_argument("--remove-model", metavar="MODEL", help="Remove model")
    management.add_argument("--model-info", metavar="MODEL", help="Show model information")
    management.add_argument("--set-model", nargs=2, metavar=("AGENT", "MODEL"), help="Set model for agent")
    management.add_argument("--clean-hf-cache", action="store_true", help="Clean HuggingFace model cache")
    
    # Voice management
    management.add_argument("--list-voices", action="store_true", help="List TTS voices")
    management.add_argument("--test-voice", metavar="VOICE_ID", help="Test TTS voice")
    management.add_argument("--set-voice", nargs=2, metavar=("AGENT", "VOICE_ID"), help="Set voice for agent")
    management.add_argument("--list-speech-backends", action="store_true", help="List speech recognition backends")
    management.add_argument("--set-speech-backend", nargs=2, metavar=("AGENT", "BACKEND"), help="Set speech backend")
    management.add_argument("--configure-voice", metavar="AGENT", help="Configure voice for agent")
    
    # Knowledge management
    management.add_argument("--index-knowledge", nargs=2, metavar=("SOURCE_DIR", "RAGFILE_NAME"), 
                           help="Index directory into RAG file")
    management.add_argument("--list-knowledge", action="store_true", help="List RAG files")
    management.add_argument("--delete-knowledge", metavar="RAGFILE", help="Delete RAG file")
    management.add_argument("--set-rag", nargs=2, metavar=("AGENT", "RAGFILES"), help="Set RAG files for agent (comma-separated)")
    management.add_argument("--search-knowledge", nargs=2, metavar=("RAGFILE", "QUERY"), help="Search in RAG file")
    
    # Personality management
    management.add_argument("--list-personalities", action="store_true", help="List available personalities")
    management.add_argument("--set-personality", nargs=2, metavar=("AGENT", "PERSONALITY"), help="Set personality for agent")
    
    # System commands
    management.add_argument("--setup", action="store_true", help="Run initial setup")
    management.add_argument("--version", action="store_true", help="Show version information")
    
    return parser


def handle_management_commands(args):
    """Handle management commands."""
    agent_manager = AgentManager()
    voice_manager = VoiceManager()
    model_manager = ModelManager()
    rag_manager = RAGKnowledgeManager()
    assistant_config = AssistantConfig()
    
    # Agent management
    if args.create_agent:
        return agent_manager.create_agent_interactive()
    
    if args.list_agents:
        return agent_manager.list_agents()
    
    if args.show_agent:
        return agent_manager.show_agent_info(args.show_agent)
    
    if args.clone_agent:
        source, target = args.clone_agent
        return agent_manager.clone_agent(source, target)
    
    if args.delete_agent:
        return agent_manager.delete_agent(args.delete_agent)
    
    if args.configure:
        return agent_manager.configure_agent_interactive(args.configure)
    
    # Model management
    if args.list_models:
        return model_manager.list_all_models()
    
    if args.download_model:
        return model_manager.download_model(args.download_model)
    
    if args.remove_model:
        return model_manager.remove_model(args.remove_model)
    
    if args.model_info:
        return model_manager.show_model_info(args.model_info)
    
    if args.set_model:
        agent, model = args.set_model
        return agent_manager.set_agent_model(agent, model)
    
    if args.clean_hf_cache:
        return model_manager.clean_hf_cache()
    
    # Voice management
    if args.list_voices:
        return voice_manager.display_tts_voices()
    
    if args.test_voice:
        return voice_manager.test_tts_voice(args.test_voice)
    
    if args.set_voice:
        agent, voice_id = args.set_voice
        return voice_manager.set_assistant_voice(agent, voice_id)
    
    if args.list_speech_backends:
        return voice_manager.display_sr_models()
    
    if args.set_speech_backend:
        agent, backend = args.set_speech_backend
        return voice_manager.set_assistant_speech_backend(agent, backend)
    
    if args.configure_voice:
        return voice_manager.configure_voice_interactive(args.configure_voice)
    
    # Knowledge management
    if args.index_knowledge:
        source_dir, ragfile_name = args.index_knowledge
        return rag_manager.index_directory(Path(source_dir), ragfile_name)
    
    if args.list_knowledge:
        ragfiles = rag_manager.list_ragfiles()
        if ragfiles:
            from rich.table import Table
            table = Table(title="Available RAG Knowledge Files")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Files", style="yellow")
            table.add_column("Created", style="dim")
            
            for ragfile in ragfiles:
                table.add_row(
                    ragfile["name"],
                    ragfile["size"],
                    str(ragfile["file_count"]),
                    ragfile["created_at"][:10]
                )
            console.print(table)
        else:
            console.print("[yellow]⚠️ No RAG files found[/]")
        return True
    
    if args.delete_knowledge:
        return rag_manager.delete_ragfile(args.delete_knowledge)
    
    if args.set_rag:
        agent, ragfiles_str = args.set_rag
        ragfiles = [f.strip() for f in ragfiles_str.split(",")]
        return agent_manager.set_agent_rag(agent, ragfiles)
    
    if args.search_knowledge:
        ragfile, query = args.search_knowledge
        results = rag_manager.search_knowledge(ragfile, query)
        
        if results:
            console.print(f"[cyan]🔍 Search results for '{query}' in {ragfile}:[/]\n")
            for i, result in enumerate(results[:5], 1):
                console.print(f"[bold]{i}. {result['file_path']}[/]")
                console.print(f"[dim]Similarity: {result['similarity']:.3f}[/]")
                console.print(f"{result['content'][:200]}...\n")
        else:
            console.print(f"[yellow]⚠️ No results found for '{query}'[/]")
        return True
    
    # Personality management
    if args.list_personalities:
        from library.personality_cfg import PersonalityConfig
        personality_config = PersonalityConfig()
        return personality_config.display_personalities()
    
    if args.set_personality:
        agent, personality = args.set_personality
        return agent_manager.set_agent_personality(agent, personality)
    
    # System commands
    if args.setup:
        return run_setup()
    
    if args.version:
        console.print("[bold cyan]AI Assistant System v2.0[/]")
        console.print("Modular architecture with RAG, Voice, and Image Generation")
        return True
    
    return False


def run_assistant_session(args):
    """Run an assistant session."""
    if not args.assistant_name:
        console.print("[red]❌ Assistant name required[/]")
        console.print("Use --create-agent to create a new assistant")
        console.print("Use --list-agents to see available assistants")
        return False
    
    # Load assistant configuration
    assistant_config = AssistantConfig()
    
    # Use current directory if no working directory specified
    # Get the user's original working directory from install script
    import os
    
    # The install script sets ORIGINAL_CWD to the directory where the user ran coder
    original_cwd = os.environ.get('ORIGINAL_CWD')
    if original_cwd and Path(original_cwd).exists():
        user_cwd = Path(original_cwd)
    else:
        # Fallback to current Python working directory
        user_cwd = Path.cwd()
    
    session_working_dir = args.working_dir or user_cwd
    
    
    session_config = assistant_config.load_assistant_for_session(
        args.assistant_name, 
        session_working_dir
    )
    
    if not session_config:
        # Auto-create assistant with default settings
        console.print(f"[yellow]⚠️ Assistant '{args.assistant_name}' not found. Creating with default settings...[/]")
        
        # Determine default model and personality based on mode
        default_model = args.model or "qwen2.5-coder:3b"
        default_personality = "coder"
        
        # Use the user's current directory for auto-creation
        working_dir = session_working_dir
        
        # Create assistant with default configuration
        success = assistant_config.create_assistant(
            assistant_name=args.assistant_name,
            model=default_model,
            personality=default_personality,
            working_dir=working_dir
        )
        
        if not success:
            console.print(f"[red]❌ Failed to create assistant '{args.assistant_name}'[/]")
            return False
        
        # Load the newly created assistant
        session_config = assistant_config.load_assistant_for_session(
            args.assistant_name, 
            session_working_dir
        )
        
        if not session_config:
            console.print(f"[red]❌ Failed to load newly created assistant '{args.assistant_name}'[/]")
            return False
    
    # Override model if specified
    if args.model:
        session_config["assistant"]["model"] = args.model
        console.print(f"[cyan]🔄 Using model for this session: {args.model}[/]")
    
    # Run based on mode
    if args.mode == "chat":
        return run_chat_session(session_config, args)
    elif args.mode == "speech":
        return run_speech_session(session_config, args)
    elif args.mode == "image":
        return run_image_session(session_config, args)
    
    return False


def run_chat_session(session_config, args):
    """Run chat session."""
    from library.conversation.chat_manager import ChatManager
    from library.coding.file_ops import FileOperations
    
    assistant_name = session_config["assistant"]["name"]
    model = session_config["assistant"]["model"]
    working_dir = Path(session_config["session"]["working_dir"])
    context_dir = Path(session_config["paths"]["context_dir"])
    
    console.print(Panel.fit(
        f"[bold cyan]{assistant_name}[/] - Chat Mode\n"
        f"Model: [green]{model}[/]\n"
        f"Working Directory: [yellow]{working_dir}[/]\n"
        f"Commands: 'exit', 'reset', '/dir=<path>' to change directory",
        title="🤖 AI Assistant"
    ))
    
    # Initialize chat manager
    chat_manager = ChatManager(model, assistant_name, context_dir)
    
    # Initialize file operations if enabled
    file_ops = None
    if session_config.get("capabilities", {}).get("file_operations") and not args.no_file_ops:
        file_ops = FileOperations(working_dir)
        console.print("[dim]📁 File operations enabled[/]")
    
    # One-shot query mode
    if args.query:
        system_prompt = session_config.get("personality", {}).get("system_prompt", "")
        response = chat_manager.send_message(args.query, system_prompt)
        
        # Handle file operations
        if file_ops and response:
            code_blocks = file_ops.parse_code_blocks(response)
            if code_blocks:
                file_ops.apply_code_changes(code_blocks, args.auto_confirm)
        
        return True
    
    # Interactive mode
    system_prompt = session_config.get("personality", {}).get("system_prompt", "")
    user_name = session_config["assistant"]["user_name"]
    
    while True:
        try:
            user_input = input(f"\n💬 [{user_name}] ({working_dir.name}): ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print(f"[dim]Goodbye! 👋[/]")
                break
            
            if user_input.lower() == 'reset':
                chat_manager.clear_conversation()
                continue
            
            # Handle directory change command
            if user_input.startswith('/dir='):
                new_dir_str = user_input[5:].strip()
                try:
                    # Handle different path types
                    if new_dir_str.startswith('~/'):
                        # Expand home directory
                        new_dir = Path(new_dir_str).expanduser()
                    elif new_dir_str.startswith('./') or new_dir_str.startswith('../'):
                        # Relative paths - relative to current working directory
                        new_dir = working_dir / new_dir_str
                    elif new_dir_str == '.':
                        # Stay in current directory
                        new_dir = working_dir
                    elif new_dir_str == '~':
                        # Go to home directory
                        new_dir = Path.home()
                    else:
                        # Absolute path or relative without ./
                        new_dir = Path(new_dir_str)
                        if not new_dir.is_absolute():
                            # Make relative to current working directory
                            new_dir = working_dir / new_dir_str
                    
                    new_dir = new_dir.resolve()
                    
                    if new_dir.exists() and new_dir.is_dir():
                        working_dir = new_dir
                        # Update file operations working directory
                        if file_ops:
                            file_ops.working_dir = working_dir
                        console.print(f"[green]📁 Changed working directory to: {working_dir}[/]")
                    else:
                        console.print(f"[red]❌ Directory not found: {new_dir}[/]")
                except Exception as e:
                    console.print(f"[red]❌ Invalid directory path: {e}[/]")
                continue
            
            if not user_input:
                continue
            
            # Send message and get response
            response = chat_manager.send_message(user_input, system_prompt)
            
            # Handle file operations
            if file_ops and response:
                code_blocks = file_ops.parse_code_blocks(response)
                if code_blocks:
                    file_ops.apply_code_changes(code_blocks, args.auto_confirm)
            
        except KeyboardInterrupt:
            console.print(f"\n[dim]Goodbye! 👋[/]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")
    
    return True


def run_speech_session(session_config, args):
    """Run speech session with voice interaction."""
    assistant_name = session_config["assistant"]["name"]
    model = session_config["assistant"]["model"]
    working_dir = Path(session_config["session"]["working_dir"])
    context_dir = Path(session_config["paths"]["context_dir"])
    
    # Get speech configuration
    voice_config = session_config.get("voice", {})
    speech_backend = voice_config.get("speech_backend", "google")
    voice_id = voice_config.get("voice_id", "")
    voice_name = voice_config.get("voice_name", "Default")
    speech_rate = voice_config.get("speech_rate", 200)
    
    console.print(Panel.fit(
        f"[bold cyan]{assistant_name}[/] - Speech Mode\n"
        f"Model: [green]{model}[/]\n"
        f"Working Directory: [yellow]{working_dir}[/]\n"
        f"Speech Backend: [cyan]{speech_backend}[/]\n"
        f"Voice: [magenta]{voice_name} (ID: {voice_id})[/]\n"
        f"Speech Rate: [blue]{speech_rate}[/]\n"
        f"Commands: 'exit', 'reset', or speak naturally",
        title="🎤 Speech Assistant"
    ))
    
    # Check if speech dependencies are available
    try:
        import speech_recognition as sr
        import pyttsx3
    except ImportError:
        console.print("[red]❌ Speech dependencies not installed[/]")
        console.print("[dim]Install with: python3 install.py --install-speech[/]")
        return False
    
    # Initialize speech components
    from library.conversation.chat_manager import ChatManager
    
    # Initialize chat manager
    chat_manager = ChatManager(model, assistant_name, context_dir)
    
    # Initialize speech recognition
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Check if configured speech backend is available
    if speech_backend == "whisper":
        try:
            import whisper
            whisper_model = whisper.load_model("base")
            console.print(f"[green]✅ Using Whisper model for speech recognition[/]")
        except ImportError:
            console.print(f"[yellow]⚠️ Whisper not installed, falling back to Google[/]")
            console.print("[dim]Install with: pip install openai-whisper[/]")
            speech_backend = "google"
    elif speech_backend == "vosk":
        try:
            import vosk
            console.print(f"[green]✅ Using Vosk for speech recognition[/]")
        except ImportError:
            console.print(f"[yellow]⚠️ Vosk not installed, falling back to Google[/]")
            console.print("[dim]Install with: pip install vosk[/]")
            speech_backend = "google"
    
    # Initialize text-to-speech
    tts_engine = pyttsx3.init()
    
    # Configure voice properly
    if voice_id:
        voices = tts_engine.getProperty('voices')
        target_voice = None
        
        # Try to find voice by ID (number) or by voice object ID
        try:
            voice_index = int(voice_id)
            if 0 <= voice_index < len(voices):
                target_voice = voices[voice_index]
        except ValueError:
            # Try as voice ID string
            for voice in voices:
                if voice.id == voice_id:
                    target_voice = voice
                    break
        
        if target_voice:
            tts_engine.setProperty('voice', target_voice.id)
            console.print(f"[green]✅ Using voice: {target_voice.name}[/]")
        else:
            console.print(f"[yellow]⚠️ Voice ID '{voice_id}' not found, using default[/]")
    
    # Set speech rate  
    tts_engine.setProperty('rate', speech_rate)
    console.print(f"[green]✅ Speech rate set to {speech_rate} WPM[/]")
    
    console.print("[green]🎤 Speech mode ready! Start speaking...[/]")
    console.print("[dim]Say 'exit', 'quit', or 'reset' for commands[/]")
    
    # Calibrate microphone for ambient noise
    with microphone as source:
        console.print("[dim]Calibrating for ambient noise...[/]", end="")
        recognizer.adjust_for_ambient_noise(source)
        console.print(" Done!")
    
    system_prompt = session_config.get("personality", {}).get("system_prompt", "")
    user_name = session_config["assistant"]["user_name"]
    
    while True:
        try:
            # Listen for speech
            console.print(f"\n🎤 [{user_name}] Listening...", end="")
            
            with microphone as source:
                # Listen for audio with timeout
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
            console.print(" Processing...", end="")
            
            # Recognize speech using configured backend
            try:
                if speech_backend == "whisper" and 'whisper_model' in locals():
                    # Use Whisper for recognition
                    import tempfile
                    import wave
                    
                    # Save audio to temporary file for Whisper
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        with wave.open(tmp_file.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(audio.sample_width)
                            wav_file.setframerate(audio.sample_rate)
                            wav_file.writeframes(audio.frame_data)
                        
                        # Transcribe with Whisper
                        result = whisper_model.transcribe(tmp_file.name)
                        user_input = result["text"].strip()
                elif speech_backend == "vosk" and 'vosk' in locals():
                    # Use Vosk for recognition
                    user_input = recognizer.recognize_vosk(audio)
                else:
                    # Fall back to Google
                    user_input = recognizer.recognize_google(audio)
                
                console.print(f" Heard: '{user_input}'")
            except sr.UnknownValueError:
                console.print(" Could not understand audio")
                continue
            except sr.RequestError as e:
                console.print(f" Error with speech service: {e}")
                continue
            except Exception as e:
                console.print(f" Speech recognition error: {e}")
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                console.print("Goodbye! 👋")
                break
            
            if user_input.lower() == 'reset':
                chat_manager.clear_conversation()
                console.print("Conversation reset")
                continue
            
            # Get AI response
            response = chat_manager.send_message(user_input, system_prompt)
            
            if response:
                console.print(f"🤖 [{assistant_name}]: {response}")
                
                # Filter out thinking tags for TTS
                import re
                tts_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
                tts_response = tts_response.strip()
                
                # Only speak if there's content after filtering
                if tts_response:
                    tts_engine.say(tts_response)
                    tts_engine.runAndWait()
                else:
                    console.print("[dim](Response contained only thinking, not spoken)[/]")
            
        except sr.WaitTimeoutError:
            console.print(" Timeout - no speech detected")
            continue
        except KeyboardInterrupt:
            console.print("\nGoodbye! 👋")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")
    
    # Clean up
    try:
        tts_engine.stop()
    except:
        pass
    
    return True


def run_image_session(session_config, args):
    """Run image generation session."""
    if not args.prompt:
        console.print("[red]❌ Image prompt required for image mode[/]")
        console.print("Use: --prompt 'Your image description'")
        return False
    
    from library.image_generation.generation import ImageGenerator
    
    working_dir = Path(session_config["session"]["working_dir"])
    images_dir = working_dir / session_config["image"]["output_subdir"]
    models = session_config["image"]["models"]
    
    console.print(f"[cyan]🎨 Generating image: '{args.prompt}'[/]")
    console.print(f"[dim]Output directory: {images_dir}[/]")
    
    # Initialize image generator
    generator = ImageGenerator(models, images_dir)
    
    # Generate image
    result = generator.generate_image(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        model_name=models[0] if models else None
    )
    
    if result:
        console.print(f"[green]✅ Image generated successfully![/]")
        return True
    else:
        console.print(f"[red]❌ Image generation failed[/]")
        return False


def run_setup():
    """Run initial setup."""
    console.print(Panel.fit(
        "[bold cyan]🚀 AI Assistant Setup[/]\n\n"
        "This will guide you through the initial setup process.",
        title="Setup Wizard"
    ))
    
    # Create default directories
    Path("knowledge").mkdir(exist_ok=True)
    Path("configuration").mkdir(exist_ok=True)
    
    console.print("[green]✅ Created default directories[/]")
    
    # Ask if user wants to create first assistant
    from rich.prompt import Confirm
    if Confirm.ask("Would you like to create your first assistant?"):
        agent_manager = AgentManager()
        agent_manager.create_agent_interactive()
    
    console.print(Panel.fit(
        "[bold green]🎉 Setup Complete![/]\n\n"
        "You can now use the AI assistant system.\n"
        "Try: coder --list-agents",
        title="Setup Complete"
    ))
    
    return True


def main():
    """Main entry point."""
    parser = create_parser()
    
    # Handle command shortcuts (legacy compatibility)
    if len(sys.argv) >= 2:
        command = sys.argv[1]
        
        # Management command shortcuts
        shortcuts = {
            "create-agent": ["--create-agent"],
            "list-agents": ["--list-agents"],
            "list-models": ["--list-models"],
            "models": ["--list-models"],
            "list-voices": ["--list-voices"],
            "list-knowledge": ["--list-knowledge"],
            "list-personalities": ["--list-personalities"],
            "setup": ["--setup"],
            "version": ["--version"]
        }
        
        if command in shortcuts:
            sys.argv = [sys.argv[0]] + shortcuts[command] + sys.argv[2:]
    
    args = parser.parse_args()
    
    try:
        # Handle management commands first
        if handle_management_commands(args):
            return
        
        # If no management command, run assistant session
        if args.assistant_name:
            run_assistant_session(args)
        else:
            # No assistant name and no management command - show help
            parser.print_help()
    
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted by user[/]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/]")
        console.print("[yellow]💡 Make sure all dependencies are installed and Ollama is running[/]")


if __name__ == "__main__":
    main()