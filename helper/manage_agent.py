"""
Agent management utilities.

Handles creating, configuring, and managing AI assistants.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from library.assistant_cfg import AssistantConfig
from library.personality_cfg import PersonalityConfig
from library.model_cfg import ModelConfig
from .manage_model import ModelManager

console = Console()


class AgentManager:
    """Manages AI assistant agents and their configurations."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize agent manager."""
        self.config_dir = config_dir or Path("configuration")
        
        # Initialize configuration managers
        self.assistant_config = AssistantConfig(self.config_dir)
        self.personality_config = PersonalityConfig(self.config_dir)
        self.model_config = ModelConfig(self.config_dir)
        self.model_manager = ModelManager(self.config_dir)
    
    def create_agent_interactive(self) -> bool:
        """Create a new agent interactively."""
        console.print(Panel.fit(
            "[bold cyan]🤖 Create New AI Assistant[/]",
            title="Agent Creation Wizard"
        ))
        
        # Get agent name
        agent_name = Prompt.ask(
            "[cyan]Agent name[/]",
            default="mycoder"
        )
        
        # Check if agent already exists
        if self.assistant_config.get_assistant(agent_name):
            if not Confirm.ask(f"[yellow]Agent '{agent_name}' already exists. Overwrite?[/]"):
                console.print("[yellow]❌ Agent creation cancelled[/]")
                return False
        
        # Show available models
        console.print("\n[bold]Available Models:[/]")
        ollama_models = self.model_manager.list_ollama_models()
        if not ollama_models:
            console.print("[yellow]⚠️ No Ollama models downloaded. Using default model.[/]")
            console.print("[dim]Download models with: ollama pull qwen2.5-coder:3b[/]")
        
        # Get model selection
        if ollama_models:
            default_model = ollama_models[0]  # Use first available model as default
            console.print(f"[dim]Available: {', '.join(ollama_models)}[/]")
        else:
            default_model = "qwen2.5-coder:3b"  # Fallback default
            
        model = Prompt.ask(
            "[cyan]Model to use[/]",
            default=default_model
        )
        
        # Validate model selection
        if ollama_models and model not in ollama_models:
            console.print(f"[yellow]⚠️ Model '{model}' not found in downloaded models. Using anyway.[/]")
        
        # Show available personalities
        console.print("\n[bold]Available Personalities:[/]")
        personalities = self.personality_config.list_personalities()
        if personalities:
            personality_table = Table()
            personality_table.add_column("ID", style="cyan")
            personality_table.add_column("Name", style="green")
            personality_table.add_column("Description", style="dim")
            
            for personality in personalities:
                personality_table.add_row(
                    personality["id"],
                    personality["name"],
                    personality["description"][:40] + "..." if len(personality["description"]) > 40 else personality["description"]
                )
            
            console.print(personality_table)
        
        # Get personality selection
        personality = Prompt.ask(
            "[cyan]Personality to use[/]",
            default="coder"
        )
        
        # Get working directory
        working_dir = Prompt.ask(
            "[cyan]Default working directory[/]",
            default=str(Path.cwd())
        )
        
        # Get user name
        import getpass
        try:
            default_user = getpass.getuser()
        except:
            default_user = "user"
        
        user_name = Prompt.ask(
            "[cyan]Your name[/]",
            default=default_user
        )
        
        # Create the agent
        console.print(f"\n[cyan]🔧 Creating agent '{agent_name}'...[/]")
        
        success = self.assistant_config.create_assistant(
            assistant_name=agent_name,
            model=model,
            personality=personality,
            working_dir=Path(working_dir),
            user_name=user_name
        )
        
        if success:
            console.print(Panel.fit(
                f"[bold green]✅ Agent '{agent_name}' created successfully![/]\n\n"
                f"Model: {model}\n"
                f"Personality: {personality}\n"
                f"Working Directory: {working_dir}\n"
                f"User: {user_name}",
                title="Agent Created"
            ))
        
        return success
    
    def list_agents(self):
        """List all available agents."""
        agents = self.assistant_config.list_assistants()
        
        if not agents:
            console.print("[yellow]⚠️ No agents found[/]")
            return
        
        table = Table(title="Available AI Assistants")
        table.add_column("Name", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Personality", style="yellow")
        table.add_column("Working Dir", style="dim")
        table.add_column("User", style="magenta")
        table.add_column("Created", style="dim")
        
        for agent in agents:
            # Truncate long paths
            working_dir = agent["working_dir"]
            if len(working_dir) > 30:
                working_dir = "..." + working_dir[-27:]
            
            table.add_row(
                agent["name"],
                agent["model"],
                agent["personality"],
                working_dir,
                agent["user_name"],
                agent["created_at"][:10]  # Just date part
            )
        
        console.print(table)
    
    def show_agent_info(self, agent_name: str):
        """Show detailed information about an agent."""
        config = self.assistant_config.get_assistant(agent_name)
        
        if not config:
            console.print(f"[red]❌ Agent '{agent_name}' not found[/]")
            return
        
        # Basic info
        assistant_info = config.get("assistant", {})
        personality_info = config.get("personality", {})
        capabilities = config.get("capabilities", {})
        rag_info = config.get("rag", {})
        
        info_text = f"""[bold cyan]{assistant_info.get('name', agent_name)}[/]

[bold]Model & Configuration:[/]
• Model: {assistant_info.get('model', 'Unknown')}
• Temperature: {personality_info.get('temperature', 0.1)}
• User: {assistant_info.get('user_name', 'Unknown')}
• Working Directory: {assistant_info.get('working_dir', '.')}

[bold]Personality:[/]
• ID: {personality_info.get('personality_id', 'Unknown')}
• Traits: {', '.join(personality_info.get('traits', []))}
• Specialties: {', '.join(personality_info.get('specialties', []))}

[bold]Capabilities:[/]
• File Operations: {'✅' if capabilities.get('file_operations') else '❌'}
• RAG Enabled: {'✅' if capabilities.get('rag_enabled') else '❌'}
• Image Generation: {'✅' if capabilities.get('image_generation') else '❌'}
• Voice Enabled: {'✅' if capabilities.get('voice_enabled') else '❌'}

[bold]RAG Knowledge:[/]
• Enabled: {'✅' if rag_info.get('enabled') else '❌'}
• RAG Files: {', '.join(rag_info.get('rag_files', [])) or 'None'}
• Knowledge Directory: {rag_info.get('knowledge_dir', 'knowledge')}

[bold]Timestamps:[/]
• Created: {assistant_info.get('created_at', 'Unknown')}
• Updated: {assistant_info.get('updated_at', 'Never')}
"""
        
        console.print(Panel(info_text, title=f"Agent: {agent_name}"))
    
    def set_agent_model(self, agent_name: str, model: str) -> bool:
        """Set AI model for an agent."""
        # Check if model exists in downloaded Ollama models (if any are available)
        ollama_models = self.model_manager.list_ollama_models()
        if ollama_models and model not in ollama_models:
            console.print(f"[yellow]⚠️ Model '{model}' not found in downloaded models[/]")
            console.print(f"[dim]Available models: {', '.join(ollama_models)}[/]")
            console.print("[dim]Download with: ollama pull <model>[/]")
            
            from rich.prompt import Confirm
            if not Confirm.ask("Continue anyway?", default=False):
                return False
            console.print("[dim]Setting model anyway (you can download it later)[/]")
        
        # Update agent configuration
        success = self.assistant_config.update_assistant_setting(
            agent_name, "assistant", "model", model
        )
        
        if success:
            console.print(f"[green]✅ Set model for '{agent_name}': {model}[/]")
        
        return success
    
    def set_agent_personality(self, agent_name: str, personality: str) -> bool:
        """Set personality for an agent."""
        # Check if personality exists
        personality_data = self.personality_config.get_personality(personality)
        if not personality_data:
            console.print(f"[red]❌ Personality '{personality}' not found[/]")
            console.print("[dim]Use 'list-personalities' to see available personalities[/]")
            return False
        
        # Apply personality to agent
        success = self.assistant_config.apply_personality(agent_name, personality)
        
        if success:
            console.print(f"[green]✅ Applied personality '{personality}' to '{agent_name}'[/]")
        
        return success
    
    def set_agent_rag(self, agent_name: str, rag_files: List[str]) -> bool:
        """Set RAG files for an agent."""
        # Validate RAG files exist
        from .rag_knowledge import RAGKnowledgeManager
        rag_manager = RAGKnowledgeManager()
        
        for rag_file in rag_files:
            if not rag_manager.load_ragfile(rag_file):
                console.print(f"[red]❌ RAG file not found: {rag_file}[/]")
                return False
        
        # Set RAG files for agent
        success = self.assistant_config.set_rag_files(agent_name, rag_files)
        
        if success:
            # Enable RAG if not already enabled
            self.assistant_config.enable_rag(agent_name)
            console.print(f"[green]✅ Set RAG files for '{agent_name}': {', '.join(rag_files)}[/]")
        
        return success
    
    def clone_agent(self, source_name: str, target_name: str, 
                   new_working_dir: str = None) -> bool:
        """Clone an existing agent."""
        working_dir = Path(new_working_dir) if new_working_dir else None
        
        success = self.assistant_config.clone_assistant(
            source_name, target_name, working_dir
        )
        
        if success:
            console.print(f"[green]✅ Cloned agent '{source_name}' to '{target_name}'[/]")
            if new_working_dir:
                console.print(f"[dim]New working directory: {new_working_dir}[/]")
        
        return success
    
    def delete_agent(self, agent_name: str) -> bool:
        """Delete an agent."""
        if not Confirm.ask(f"[red]Are you sure you want to delete agent '{agent_name}'?[/]"):
            console.print("[yellow]❌ Agent deletion cancelled[/]")
            return False
        
        success = self.assistant_config.delete_assistant(agent_name)
        
        if success:
            console.print(f"[green]✅ Deleted agent '{agent_name}'[/]")
        
        return success
    
    def configure_agent_interactive(self, agent_name: str) -> bool:
        """Configure an agent interactively."""
        config = self.assistant_config.get_assistant(agent_name)
        
        if not config:
            console.print(f"[red]❌ Agent '{agent_name}' not found[/]")
            return False
        
        console.print(Panel.fit(
            f"[bold cyan]🔧 Configure Agent: {agent_name}[/]",
            title="Agent Configuration"
        ))
        
        while True:
            console.print("\n[bold]Configuration Options:[/]")
            console.print("1. Change model")
            console.print("2. Change personality")
            console.print("3. Set RAG knowledge files")
            console.print("4. Enable/disable capabilities")
            console.print("5. View current configuration")
            console.print("6. Exit configuration")
            
            choice = Prompt.ask(
                "[cyan]Select option[/]",
                choices=["1", "2", "3", "4", "5", "6"],
                default="6"
            )
            
            if choice == "1":
                console.print("\n[bold]Available Models:[/]")
                ollama_models = self.model_manager.list_ollama_models()
                if ollama_models:
                    console.print(f"[green]Choose from {len(ollama_models)} downloaded Ollama models above[/]")
                    new_model = Prompt.ask("[cyan]New model[/]")
                    if new_model in ollama_models:
                        self.set_agent_model(agent_name, new_model)
                    else:
                        console.print(f"[red]❌ Model '{new_model}' not found in downloaded models[/]")
                else:
                    console.print("[yellow]⚠️ No Ollama models downloaded. Download a model first:[/]")
                    console.print("[dim]Example: ollama pull qwen2.5-coder:3b[/]")
                
            elif choice == "2":
                self.personality_config.display_personalities()
                new_personality = Prompt.ask("[cyan]New personality[/]")
                self.set_agent_personality(agent_name, new_personality)
                
            elif choice == "3":
                from .rag_knowledge import RAGKnowledgeManager
                rag_manager = RAGKnowledgeManager()
                ragfiles = rag_manager.list_ragfiles()
                
                if ragfiles:
                    console.print("\n[bold]Available RAG Files:[/]")
                    for ragfile in ragfiles:
                        console.print(f"• {ragfile['name']}")
                    
                    rag_files_str = Prompt.ask(
                        "[cyan]RAG files (comma-separated)[/]",
                        default=""
                    )
                    
                    if rag_files_str:
                        rag_files = [f.strip() for f in rag_files_str.split(",")]
                        self.set_agent_rag(agent_name, rag_files)
                else:
                    console.print("[yellow]⚠️ No RAG files available[/]")
                
            elif choice == "4":
                self._configure_capabilities(agent_name)
                
            elif choice == "5":
                self.show_agent_info(agent_name)
                
            elif choice == "6":
                break
        
        console.print(f"[green]✅ Configuration complete for '{agent_name}'[/]")
        return True
    
    def _configure_capabilities(self, agent_name: str):
        """Configure agent capabilities."""
        config = self.assistant_config.get_assistant(agent_name)
        capabilities = config.get("capabilities", {})
        
        console.print("\n[bold]Current Capabilities:[/]")
        for cap, enabled in capabilities.items():
            status = "✅" if enabled else "❌"
            console.print(f"• {cap.replace('_', ' ').title()}: {status}")
        
        # Toggle capabilities
        for cap, current_value in capabilities.items():
            new_value = Confirm.ask(
                f"Enable {cap.replace('_', ' ').title()}?",
                default=current_value
            )
            
            if new_value != current_value:
                self.assistant_config.update_assistant_setting(
                    agent_name, "capabilities", cap, new_value
                )
    
    def export_agent_config(self, agent_name: str, output_file: str = None) -> bool:
        """Export agent configuration to file."""
        config = self.assistant_config.get_assistant(agent_name)
        
        if not config:
            console.print(f"[red]❌ Agent '{agent_name}' not found[/]")
            return False
        
        if output_file is None:
            output_file = f"{agent_name}_config_export.toml"
        
        try:
            output_path = Path(output_file)
            
            # Use the assistant config's save method
            success = self.assistant_config.save_toml(output_path.name, config)
            
            if success:
                console.print(f"[green]✅ Exported agent config to: {output_file}[/]")
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Error exporting config: {e}[/]")
            return False
    
    def import_agent_config(self, import_file: str, agent_name: str = None) -> bool:
        """Import agent configuration from file."""
        import_path = Path(import_file)
        
        if not import_path.exists():
            console.print(f"[red]❌ Import file not found: {import_file}[/]")
            return False
        
        try:
            # Load configuration
            config = self.assistant_config.load_toml(import_path.name)
            
            if not config:
                console.print(f"[red]❌ Could not load configuration from: {import_file}[/]")
                return False
            
            # Get agent name
            if agent_name is None:
                agent_name = config.get("assistant", {}).get("name", import_path.stem)
            
            # Save as new agent
            config_file = f"{agent_name}.assistant.toml"
            success = self.assistant_config.save_toml(config_file, config)
            
            if success:
                console.print(f"[green]✅ Imported agent config as: {agent_name}[/]")
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Error importing config: {e}[/]")
            return False