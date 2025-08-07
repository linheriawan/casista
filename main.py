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
from library.config_loader import ConfigLoader

console = Console()
SysCfg=ConfigLoader().load_toml("sys.definition.toml")

def create_parser():
    """Create argument parser with all commands."""
    # for key,data in SysCfg.items():
    #     console.print(f"[orange]SYSCON '{key}' -> {data}[/]")
    _epilog=SysCfg["main"]["epilog"]
    parser = argparse.ArgumentParser(
        prog="coder",
        description="AI Assistant with RAG, Voice, and Image Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog
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
        from library.operation_handler import OperationHandler
        operation_handler = OperationHandler()
        return operation_handler.run_setup()
    
    if args.version:
        console.print(f"[bold cyan]AI Assistant System {SysCfg['main']['version']}[/]")
        console.print(SysCfg["main"]["description"])
        return True
    
    # Direct image generation (one-shot mode)
    if args.prompt and args.assistant_name:
        from library.operation_handler import OperationHandler
        operation_handler = OperationHandler()
        return operation_handler.generate_single_image(args)
    
    # Direct one-shot query mode
    if args.query and args.assistant_name:
        from library.operation_handler import OperationHandler
        operation_handler = OperationHandler()
        return operation_handler.handle_one_shot_query(args)
    
    return False

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
        
        # If no management command, run session
        if args.assistant_name:
            from library.operation_handler import OperationHandler
            operation_handler = OperationHandler()
            return operation_handler.run_session(args)
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