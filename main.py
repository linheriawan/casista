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
    from library.session_manager import SessionManager
    
    # Initialize session manager with all configurations
    session = SessionManager(session_config)
    model_info = session.get_model_info()
    
    console.print(Panel.fit(
        f"[bold cyan]{session.assistant_name}[/] - Chat Mode\n"
        f"Model: [green]{model_info['model']}[/]\n"
        f"Temperature: [blue]{model_info['temperature']}[/]\n"
        f"Max Tokens: [blue]{model_info['max_tokens']}[/]\n"
        f"Personality: [magenta]{model_info['personality']}[/]\n"
        f"Working Directory: [yellow]{session.working_dir}[/]\n"
        f"Commands: 'exit', 'reset', '/dir=<path>' to change directory",
        title="🤖 AI Assistant"
    ))
    
    # Get properly configured components
    chat_manager = session.get_chat_manager()
    file_ops = session.get_file_operations(enable=not args.no_file_ops)
    
    # One-shot query mode
    if args.query:
        response_data = chat_manager.send_message(args.query)
        
        if response_data.get("error"):
            console.print(f"[red]❌ {response_data['content']}[/]")
            return False
        
        # Check if we should display the parsed response (for reasoning models with placeholder)
        streaming_config = session.get_streaming_config()
        show_placeholder = streaming_config.get("show_placeholder_for_reasoning", False)
        show_raw_stream = streaming_config.get("show_raw_stream", True)
        
        # Only display parsed response if we used placeholder streaming
        if show_placeholder and not show_raw_stream:
            if response_data.get("has_reasoning"):
                console.print(f"🤖 [{session.assistant_name}]: {response_data['clean_answer']}")
                if response_data.get("reasoning"):
                    console.print(f"[dim]💭 Reasoning: {response_data['reasoning']}[/]")
            else:
                console.print(f"🤖 [{session.assistant_name}]: {response_data['content']}")
        
        # Handle file operations
        if file_ops and response_data['content']:
            code_blocks = file_ops.parse_code_blocks(response_data['content'])
            if code_blocks:
                file_ops.apply_code_changes(code_blocks, args.auto_confirm)
        
        return True
    
    # Interactive mode
    user_name = session.get_user_name()
    
    while True:
        try:
            user_input = input(f"\n💬 [{user_name}] ({session.working_dir.name}): ").strip()
            
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
                        new_dir = session.working_dir / new_dir_str
                    elif new_dir_str == '.':
                        # Stay in current directory
                        new_dir = session.working_dir
                    elif new_dir_str == '~':
                        # Go to home directory
                        new_dir = Path.home()
                    else:
                        # Absolute path or relative without ./
                        new_dir = Path(new_dir_str)
                        if not new_dir.is_absolute():
                            # Make relative to current working directory
                            new_dir = session.working_dir / new_dir_str
                    
                    new_dir = new_dir.resolve()
                    
                    if new_dir.exists() and new_dir.is_dir():
                        session.update_working_directory(new_dir)
                    else:
                        console.print(f"[red]❌ Directory not found: {new_dir}[/]")
                except Exception as e:
                    console.print(f"[red]❌ Invalid directory path: {e}[/]")
                continue
            
            if not user_input:
                continue
            
            # Send message and get response
            response_data = chat_manager.send_message(user_input)
            
            if response_data.get("error"):
                console.print(f"[red]❌ {response_data['content']}[/]")
                continue
            
            # Check if we should display the parsed response (for reasoning models with placeholder)
            streaming_config = session.get_streaming_config()
            show_placeholder = streaming_config.get("show_placeholder_for_reasoning", False)
            show_raw_stream = streaming_config.get("show_raw_stream", True)
            
            # Only display parsed response if we used placeholder streaming
            if show_placeholder and not show_raw_stream:
                if response_data.get("has_reasoning"):
                    console.print(f"🤖 [{session.assistant_name}]: {response_data['clean_answer']}")
                    if response_data.get("reasoning"):
                        console.print(f"[dim]💭 Reasoning: {response_data['reasoning']}[/]")
                else:
                    console.print(f"🤖 [{session.assistant_name}]: {response_data['content']}")
            
            # Handle file operations
            if file_ops and response_data['content']:
                code_blocks = file_ops.parse_code_blocks(response_data['content'])
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
    from library.session_manager import SessionManager
    
    # Initialize session manager
    session = SessionManager(session_config)
    model_info = session.get_model_info()
    voice_config = session.get_voice_config()
    
    # Extract voice configuration
    speech_backend = voice_config["speech_backend"]
    voice_id = voice_config["voice_id"]
    voice_name = voice_config["voice_name"]
    speech_rate = voice_config["speech_rate"]
    
    console.print(Panel.fit(
        f"[bold cyan]{session.assistant_name}[/] - Speech Mode\n"
        f"Model: [green]{model_info['model']}[/]\n"
        f"Temperature: [blue]{model_info['temperature']}[/]\n"
        f"Working Directory: [yellow]{session.working_dir}[/]\n"
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
    
    # Get properly configured chat manager
    chat_manager = session.get_chat_manager()
    
    # Initialize speech recognition
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Check if configured speech backend is available
    if speech_backend == "whisper":
        try:
            import whisper
            whisper_model = whisper.load_model("base", device="cpu", download_root=None)
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
    
    system_prompt = session.get_system_prompt()
    user_name = session.get_user_name()
    
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
            response_data = chat_manager.send_message(user_input)
            
            if response_data.get("error"):
                console.print(f"[red]❌ {response_data['content']}[/]")
                continue
            
            # Check streaming configuration for display
            streaming_config = session.get_streaming_config()
            show_placeholder = streaming_config.get("show_placeholder_for_reasoning", False)
            show_raw_stream = streaming_config.get("show_raw_stream", True)
            
            # Determine TTS content and display
            if response_data.get("has_reasoning"):
                # Always use clean answer for TTS
                tts_response = response_data['clean_answer']
                
                # Display based on streaming config
                if show_placeholder and not show_raw_stream:
                    console.print(f"🤖 [{session.assistant_name}]: {response_data['clean_answer']}")
                    if response_data.get("reasoning"):
                        console.print(f"[dim]💭 Reasoning: {response_data['reasoning']}[/]")
            else:
                tts_response = response_data['content']
                
                # Display based on streaming config  
                if show_placeholder and not show_raw_stream:
                    console.print(f"🤖 [{session.assistant_name}]: {response_data['content']}")
            
            # Speak the clean response (reasoning-free)
            if tts_response and tts_response.strip():
                tts_engine.say(tts_response)
                tts_engine.runAndWait()
            else:
                console.print("[dim](No speakable content)[/]")
            
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
    """Run interactive image generation session with conversational context."""
    
    from library.session_manager import SessionManager
    
    # Initialize session manager
    session = SessionManager(session_config)
    model_info = session.get_model_info()
    
    # Get properly configured components
    chat_manager = session.get_chat_manager()
    generator = session.get_image_generator()
    
    if not generator:
        console.print("[red]❌ Image generation not enabled for this assistant[/]")
        console.print("[dim]Enable with: coder configure <assistant> and set image_generation = true[/]")
        return False
    
    console.print(Panel.fit(
        f"[bold cyan]{session.assistant_name}[/] - Image Generation Mode\n"
        f"Chat Model: [green]{model_info['model']}[/] (for conversation & prompt enhancement)\n"
        f"Image Models: [magenta]{', '.join(session.config.get('image', {}).get('models', ['stable-diffusion']))}[/]\n"
        f"Working Directory: [yellow]{session.working_dir}[/]\n"
        f"Commands: 'exit', 'reset', 'list models', 'switch model <name>', 'refine last', 'variations <n>', 'pipeline <steps>'",
        title="🎨 Interactive Image Studio"
    ))
    
    # Enhanced system prompt for interactive image generation
    image_system_prompt = session.get_mode_specific_system_prompt("image")
    
    # Interactive mode or one-shot mode
    if args.prompt:
        # One-shot generation mode
        return generate_single_image(session, chat_manager, generator, args.prompt, args)
    
    # Interactive conversational image generation mode
    user_name = session.get_user_name()
    last_generated_image = None
    current_model = session.config.get("image", {}).get("models", ["hakurei/waifu-diffusion"])[0]
    
    console.print("[green]🎨 Interactive Image Studio ready! Start describing what you want to create...[/]")
    console.print("[dim]Try commands like: 'list models', 'switch model <name>', 'refine last', 'variations 4', 'pipeline <steps>'[/]")
    
    while True:
        try:
            user_input = input(f"\n🎨 [{user_name}] ({session.working_dir.name}): ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print(f"[dim]Happy creating! 🎨[/]")
                break
            
            if user_input.lower() == 'reset':
                chat_manager.clear_conversation()
                last_generated_image = None
                console.print("[green]✅ Conversation and image history reset[/]")
                continue
            
            # Handle image-specific commands
            if user_input.lower() == 'list models':
                available_models = generator.list_available_models()
                console.print("[cyan]📋 Model Configuration:[/]")
                console.print(f"[green]🤖 Chat Model:[/] {model_info['model']} (for conversation & prompt enhancement)")
                console.print(f"[magenta]🎨 Image Models:[/] (for actual image generation)")
                cached_models = generator.cached_models
                for model in available_models:
                    if model == current_model:
                        marker = "🟢"
                        status = "(current)"
                    elif model in cached_models:
                        marker = "🔄"
                        status = "(cached)"
                    else:
                        marker = "⚪"
                        status = "(download)"
                    console.print(f"  {marker} {model} {status}")
                
                console.print("\n[dim]💡 The chat model enhances your prompts, then the image model generates visuals[/]")
                continue
            
            if user_input.lower().startswith('switch model '):
                new_model = user_input[13:].strip()
                if generator.validate_model(new_model):
                    current_model = new_model
                    console.print(f"[green]✅ Switched to model: {current_model}[/]")
                else:
                    console.print(f"[red]❌ Model not available: {new_model}[/]")
                    console.print("[dim]Use 'list models' to see available options[/]")
                continue
            
            if user_input.lower() == 'refine last' and last_generated_image:
                # Use img2img to refine the last generated image
                console.print(f"[cyan]🆔 Refining last image: {last_generated_image['path']}[/]")
                
                # Get refinement prompt from AI
                refinement_prompt = f"Please provide a refined prompt to improve this image: {last_generated_image['prompt']}"
                response_data = chat_manager.send_message(refinement_prompt)
                
                if response_data.get("error"):
                    enhanced_prompt = f"enhanced version of: {last_generated_image['prompt']}"
                else:
                    if response_data.get("has_reasoning"):
                        console.print(f"🤖 [{session.assistant_name}]: {response_data['clean_answer']}")
                        enhanced_prompt = response_data['clean_answer']
                    else:
                        console.print(f"🤖 [{session.assistant_name}]: {response_data['content']}")
                        enhanced_prompt = response_data['content']
                
                # Refine the image
                refined_path = generator.generate_image_to_image(
                    prompt=enhanced_prompt,
                    base_image_path=Path(last_generated_image['path']),
                    model_name=current_model,
                    strength=0.7
                )
                
                if refined_path:
                    last_generated_image = {
                        'success': True,
                        'path': refined_path,
                        'prompt': enhanced_prompt,
                        'model': current_model,
                        'type': 'refinement'
                    }
                    console.print(f"[green]✅ Refined image saved: {refined_path}[/]")
                else:
                    console.print(f"[red]❌ Refinement failed[/]")
                continue
            
            if user_input.lower().startswith('variations '):
                if last_generated_image:
                    try:
                        num_variations = int(user_input.split()[1]) if len(user_input.split()) > 1 else 4
                        console.print(f"[cyan]🎭 Creating {num_variations} variations...[/]")
                        
                        variations = generator.create_variations(
                            Path(last_generated_image['path']),
                            num_variations=num_variations
                        )
                        
                        successful_variations = [v for v in variations if v is not None]
                        if successful_variations:
                            console.print(f"[green]✅ Created {len(successful_variations)} variations[/]")
                            # Update last image to the first variation
                            last_generated_image = {
                                'success': True,
                                'path': successful_variations[0],
                                'prompt': f"variation of: {last_generated_image['prompt']}",
                                'model': current_model,
                                'type': 'variation'
                            }
                        else:
                            console.print(f"[red]❌ No variations created successfully[/]")
                    except ValueError:
                        console.print(f"[red]❌ Invalid number of variations. Use: variations <number>[/]")
                else:
                    console.print(f"[yellow]⚠️ No previous image to create variations from[/]")
                continue
            
            if user_input.lower().startswith('pipeline '):
                pipeline_definition = user_input[9:].strip()
                result = run_enhanced_image_pipeline(session, chat_manager, generator, pipeline_definition, current_model, last_generated_image)
                if result:
                    last_generated_image = result
                continue
            
            if not user_input:
                continue
            
            # Generate image with conversational AI assistance
            result_data = generate_conversational_image(
                session, chat_manager, generator, user_input, current_model, args, last_generated_image
            )
            
            if result_data and result_data.get('success'):
                last_generated_image = result_data
                console.print(f"[green]✅ Image saved: {result_data['path']}[/]")
            
        except KeyboardInterrupt:
            console.print(f"\n[dim]Happy creating! 🎨[/]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")
    
    return True


def generate_single_image(session, chat_manager, generator, prompt, args):
    """Generate a single image (one-shot mode)."""
    console.print(f"[cyan]🎨 Processing image request: '{prompt}'[/]")
    
    # Get AI assistance for prompt enhancement
    response_data = chat_manager.send_message(
        f"Please help me create an image: {prompt}",
        system_prompt=session.get_mode_specific_system_prompt("image")
    )
    
    if response_data.get("error"):
        console.print(f"[yellow]⚠️ AI assistance failed, using original prompt[/]")
        enhanced_prompt = prompt
    else:
        # Display AI's response and suggestions
        if response_data.get("has_reasoning"):
            console.print(f"🤖 [{session.assistant_name}]: {response_data['clean_answer']}")
            if response_data.get("reasoning"):
                console.print(f"[dim]💭 Analysis: {response_data['reasoning']}[/]")
        else:
            console.print(f"🤖 [{session.assistant_name}]: {response_data['content']}")
        
        enhanced_prompt = prompt  # Could extract enhanced prompt from AI response
    
    # Generate image
    image_config = session.config.get("image", {})
    models = image_config.get("models", [])
    
    result = generator.generate_image(
        prompt=enhanced_prompt,
        width=args.width,
        height=args.height,
        model_name=models[0] if models else None
    )
    
    if result:
        console.print(f"[green]✅ Image generated successfully![/]")
        console.print(f"[dim]Saved to: {result}[/]")
        
        # Add to conversation context
        image_context = f"Generated image: '{enhanced_prompt}' → {result}"
        chat_manager.context_manager.add_message("assistant", image_context)
        return True
    else:
        console.print(f"[red]❌ Image generation failed[/]")
        return False


def generate_conversational_image(session, chat_manager, generator, user_input, current_model, args, last_image=None):
    """Generate image with conversational AI assistance."""
    
    # Build context for the AI assistant
    context_info = ""
    if last_image:
        context_info = f"\\nPrevious image: '{last_image['prompt']}' → {last_image['path']}"
    
    # Get AI assistance for understanding the request
    ai_prompt = f"""User request: {user_input}{context_info}

Please help with this image generation request. Provide:
1. Enhanced prompt for better image quality
2. Technical suggestions (style, composition, etc.)
3. Any refinements based on conversation history

Respond with the enhanced prompt and brief explanation."""
    
    response_data = chat_manager.send_message(ai_prompt)
    
    if response_data.get("error"):
        console.print(f"[yellow]⚠️ AI assistance failed, using original request[/]")
        enhanced_prompt = user_input
        explanation = "Using original request as provided."
    else:
        # Display AI's response
        if response_data.get("has_reasoning"):
            console.print(f"🤖 [{session.assistant_name}]: {response_data['clean_answer']}")
            if response_data.get("reasoning"):
                console.print(f"[dim]💭 Analysis: {response_data['reasoning']}[/]")
        else:
            console.print(f"🤖 [{session.assistant_name}]: {response_data['content']}")
        
        # Extract enhanced prompt from AI response (simplified for now)
        enhanced_prompt = user_input
        explanation = response_data.get('clean_answer', response_data.get('content', ''))
    
    # Generate the image
    console.print(f"[cyan]🎨 Generating with model: {current_model}[/]")
    
    result = generator.generate_image(
        prompt=enhanced_prompt,
        width=getattr(args, 'width', 512),
        height=getattr(args, 'height', 512),
        model_name=current_model
    )
    
    if result:
        # Add to conversation context
        image_context = f"Generated image: '{enhanced_prompt}' using {current_model} → {result}"
        chat_manager.context_manager.add_message("assistant", image_context)
        
        return {
            'success': True,
            'path': result,
            'prompt': enhanced_prompt,
            'model': current_model,
            'explanation': explanation
        }
    else:
        # Add failure to context
        failure_context = f"Failed to generate image: '{enhanced_prompt}' using {current_model}"
        chat_manager.context_manager.add_message("assistant", failure_context)
        
        return {
            'success': False,
            'prompt': enhanced_prompt,
            'model': current_model,
            'error': 'Generation failed'
        }


def run_enhanced_image_pipeline(session, chat_manager, generator, pipeline_definition, current_model, last_image=None):
    """Run an enhanced image pipeline with multiple step types."""
    console.print(f"[cyan]🔗 Running enhanced pipeline: {pipeline_definition}[/]")
    
    # Parse pipeline definition
    # Format: "step1 -> step2 -> step3" or "generate:prompt -> refine:new prompt -> upscale:2x"
    steps = [step.strip() for step in pipeline_definition.split('->')]
    
    console.print(f"[dim]Pipeline steps ({len(steps)}): {' → '.join(steps)}[/]")
    
    current_image = last_image
    pipeline_steps = []
    
    # Parse each step
    for step_str in steps:
        if ':' in step_str:
            step_type, step_content = step_str.split(':', 1)
            step_type = step_type.strip().lower()
            step_content = step_content.strip()
        else:
            step_type = 'generate'
            step_content = step_str
        
        step_config = {
            'type': step_type,
            'prompt': step_content,
            'model': current_model
        }
        
        # Add step-specific parameters
        if step_type == 'upscale':
            try:
                scale_factor = int(step_content.replace('x', ''))
                step_config['scale_factor'] = scale_factor
                step_config['prompt'] = ''
            except:
                step_config['scale_factor'] = 2
        elif step_type == 'refine':
            step_config['strength'] = 0.7
        
        pipeline_steps.append(step_config)
    
    # Execute pipeline using generator's run_pipeline method
    base_prompt = pipeline_steps[0]['prompt'] if pipeline_steps else "abstract art"
    
    # If we have a starting image, modify the first step to be a refinement
    if current_image and current_image.get('path'):
        if pipeline_steps and pipeline_steps[0]['type'] == 'generate':
            pipeline_steps[0]['type'] = 'refine'
            pipeline_steps[0]['base_image'] = current_image['path']
    
    result_path = generator.run_pipeline(base_prompt, pipeline_steps)
    
    if result_path:
        result_data = {
            'success': True,
            'path': result_path,
            'prompt': f"pipeline result: {pipeline_definition}",
            'model': current_model,
            'type': 'pipeline'
        }
        
        # Add to conversation context
        context_msg = f"Completed image pipeline: {pipeline_definition} → {result_path}"
        chat_manager.context_manager.add_message("assistant", context_msg)
        
        console.print(f"[green]🎉 Pipeline completed! Result: {result_path}[/]")
        return result_data
    else:
        console.print(f"[red]❌ Pipeline failed[/]")
        return None


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