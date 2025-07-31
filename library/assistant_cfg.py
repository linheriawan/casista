"""
Assistant configuration management.

Handles complete assistant configurations including working directory,
personality, model settings, and per-assistant RAG knowledge.
"""

import getpass
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console

from .config_loader import ConfigLoader
from .personality_cfg import PersonalityConfig
from .model_cfg import ModelConfig
from .prompt_cfg import PromptConfig

console = Console()


class AssistantConfig(ConfigLoader):
    """Manages complete assistant configurations."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize assistant configuration manager."""
        super().__init__(config_dir)
        
        # Initialize sub-managers
        self.personality_config = PersonalityConfig(config_dir)
        self.model_config = ModelConfig(config_dir)
        self.prompt_config = PromptConfig(config_dir)
    
    def create_assistant(self, assistant_name: str, model: str = "qwen2.5-coder:3b",
                        personality: str = "coder", working_dir: Path = None,
                        user_name: str = None) -> bool:
        """Create a new assistant configuration."""
        
        # Auto-detect user if not provided
        if user_name is None:
            try:
                user_name = getpass.getuser()
            except:
                user_name = "user"
        
        # Set working directory
        if working_dir is None:
            working_dir = Path.cwd()
        else:
            working_dir = Path(working_dir)
        
        # Create assistant configuration
        config = {
            "assistant": {
                "name": assistant_name,
                "model": model,
                "user_name": user_name,
                "working_dir": str(working_dir.absolute()),
                "created_at": self._get_timestamp()
            },
            
            "personality": {
                "personality_id": personality,
                "system_prompt": "",  # Will be populated from personality
                "temperature": 0.1,
                "custom_traits": []
            },
            
            "capabilities": {
                "file_operations": True,
                "rag_enabled": False,
                "image_generation": False,
                "voice_enabled": False
            },
            
            "voice": {
                "voice_id": "",
                "voice_name": "Default",
                "speech_backend": "google",
                "speech_rate": 200,
                "noise_level": "normal"
            },
            
            "rag": {
                "enabled": False,
                "rag_files": [],  # List of .ragfile paths to load from global knowledge/
                "knowledge_dir": "knowledge",  # Global .ragfile storage at project root
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "max_context_length": 4000
            },
            
            "image": {
                "enabled": False,
                "models": ["runwayml/stable-diffusion-v1-5"],
                "output_subdir": "generated_images",  # Subdirectory in working_dir for images
                "default_width": 512,
                "default_height": 512
            },
            
            "coding": {
                "primary_language": "Python",
                "auto_format": True,
                "include_tests": True,
                "documentation_style": "google"
            },
            
            "research": {
                "domains": [],
                "sources": ["web", "documentation"],
                "citation_style": "apa"
            },
            
            "advanced": {
                "custom_instructions": "",
                "context_window": 8192,
                "memory_limit": 100,
                "auto_save": True
            }
        }
        
        # Apply personality settings
        self._apply_personality(config, personality)
        
        # Create assistant's context directory structure
        self._create_assistant_directories(assistant_name)
        
        # Save configuration
        config_file = f"{assistant_name}.assistant.toml"
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Created assistant '{assistant_name}' with model '{model}' and personality '{personality}'[/]")
            console.print(f"[dim]Working directory: {working_dir}[/]")
            console.print(f"[dim]Assistant context: .ai_context/{assistant_name}/[/]")
            console.print(f"[dim]Configuration saved as: {config_file}[/]")
        
        return success
    
    def get_assistant(self, assistant_name: str) -> Optional[Dict[str, Any]]:
        """Get assistant configuration."""
        config_file = f"{assistant_name}.assistant.toml"
        return self.load_toml(config_file)
    
    def list_assistants(self) -> List[Dict[str, str]]:
        """List all available assistants."""
        config_files = self.list_configs("*.assistant.toml")
        assistants = []
        
        for config_file in config_files:
            assistant_name = config_file.replace(".assistant.toml", "")
            config = self.load_toml(config_file)
            
            if config and "assistant" in config:
                assistant_info = config["assistant"]
                assistants.append({
                    "name": assistant_name,
                    "model": assistant_info.get("model", "unknown"),
                    "personality": config.get("personality", {}).get("personality_id", "unknown"),
                    "working_dir": assistant_info.get("working_dir", "."),
                    "user_name": assistant_info.get("user_name", "user"),
                    "created_at": assistant_info.get("created_at", "unknown")
                })
        
        return sorted(assistants, key=lambda x: x["name"])
    
    def set_session_working_dir(self, assistant_name: str, working_dir: Path) -> Path:
        """Set working directory for current session (not saved to config)."""
        working_dir = Path(working_dir).absolute()
        console.print(f"[cyan]📁 Session working directory: {working_dir}[/]")
        
        # Return the absolute path for use in current session
        return working_dir
    
    def get_default_working_dir(self, assistant_name: str) -> Optional[Path]:
        """Get default working directory from assistant config."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            return None
        
        working_dir_str = config.get("assistant", {}).get("working_dir", ".")
        return Path(working_dir_str)
    
    def set_rag_files(self, assistant_name: str, rag_files: List[str]) -> bool:
        """Set RAG files that this assistant should load."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Update RAG files list
        config["rag"]["rag_files"] = rag_files
        config["assistant"]["updated_at"] = self._get_timestamp()
        
        # Save configuration
        config_file = f"{assistant_name}.assistant.toml"
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Updated RAG files for '{assistant_name}': {', '.join(rag_files)}[/]")
        
        return success
    
    def get_rag_files(self, assistant_name: str) -> List[str]:
        """Get RAG files that this assistant should load."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            return []
        
        return config.get("rag", {}).get("rag_files", [])
    
    def add_rag_file(self, assistant_name: str, rag_file: str) -> bool:
        """Add a RAG file to the assistant's knowledge base."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Get current RAG files
        rag_files = config.get("rag", {}).get("rag_files", [])
        
        # Add new file if not already present
        if rag_file not in rag_files:
            rag_files.append(rag_file)
            config["rag"]["rag_files"] = rag_files
            config["assistant"]["updated_at"] = self._get_timestamp()
            
            # Save configuration
            config_file = f"{assistant_name}.assistant.toml"
            success = self.save_toml(config_file, config)
            
            if success:
                console.print(f"[green]✅ Added RAG file to '{assistant_name}': {rag_file}[/]")
            
            return success
        else:
            console.print(f"[yellow]⚠️ RAG file already exists for '{assistant_name}': {rag_file}[/]")
            return True
    
    def remove_rag_file(self, assistant_name: str, rag_file: str) -> bool:
        """Remove a RAG file from the assistant's knowledge base."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Get current RAG files
        rag_files = config.get("rag", {}).get("rag_files", [])
        
        # Remove file if present
        if rag_file in rag_files:
            rag_files.remove(rag_file)
            config["rag"]["rag_files"] = rag_files
            config["assistant"]["updated_at"] = self._get_timestamp()
            
            # Save configuration
            config_file = f"{assistant_name}.assistant.toml"
            success = self.save_toml(config_file, config)
            
            if success:
                console.print(f"[green]✅ Removed RAG file from '{assistant_name}': {rag_file}[/]")
            
            return success
        else:
            console.print(f"[yellow]⚠️ RAG file not found for '{assistant_name}': {rag_file}[/]")
            return True
    
    def enable_rag(self, assistant_name: str, knowledge_path: str = None) -> bool:
        """Enable RAG for an assistant."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Enable RAG
        config["rag"]["enabled"] = True
        config["capabilities"]["rag_enabled"] = True
        
        # Set knowledge path if provided
        if knowledge_path:
            config["rag"]["knowledge_base_path"] = knowledge_path
        
        config["assistant"]["updated_at"] = self._get_timestamp()
        
        # Save configuration
        config_file = f"{assistant_name}.assistant.toml"
        success = self.save_toml(config_file, config)
        
        if success:
            knowledge_path = config["rag"]["knowledge_base_path"]
            console.print(f"[green]✅ Enabled RAG for '{assistant_name}'[/]")
            console.print(f"[dim]Knowledge base path: {knowledge_path}[/]")
        
        return success
    
    def disable_rag(self, assistant_name: str) -> bool:
        """Disable RAG for an assistant."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Disable RAG
        config["rag"]["enabled"] = False
        config["capabilities"]["rag_enabled"] = False
        config["assistant"]["updated_at"] = self._get_timestamp()
        
        # Save configuration
        config_file = f"{assistant_name}.assistant.toml"
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Disabled RAG for '{assistant_name}'[/]")
        
        return success
    
    def update_rag_stats(self, assistant_name: str, document_count: int = 0,
                        domains: List[str] = None) -> bool:
        """Update RAG statistics for an assistant."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Update RAG stats
        config["rag"]["document_count"] = document_count
        config["rag"]["last_updated"] = self._get_timestamp()
        
        if domains:
            config["rag"]["domains"] = domains
        
        config["assistant"]["updated_at"] = self._get_timestamp()
        
        # Save configuration
        config_file = f"{assistant_name}.assistant.toml"
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Updated RAG stats for '{assistant_name}': {document_count} documents[/]")
        
        return success
    
    def apply_personality(self, assistant_name: str, personality_id: str) -> bool:
        """Apply a personality to an assistant."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Apply personality
        success = self._apply_personality(config, personality_id)
        
        if success:
            config["assistant"]["updated_at"] = self._get_timestamp()
            
            # Save configuration
            config_file = f"{assistant_name}.assistant.toml"
            success = self.save_toml(config_file, config)
            
            if success:
                console.print(f"[green]✅ Applied personality '{personality_id}' to '{assistant_name}'[/]")
        
        return success
    
    def _apply_personality(self, config: Dict[str, Any], personality_id: str) -> bool:
        """Apply personality settings to configuration."""
        personality_data = self.personality_config.get_personality(personality_id)
        
        if not personality_data:
            console.print(f"[red]❌ Personality '{personality_id}' not found[/]")
            return False
        
        # Update personality settings
        config["personality"]["personality_id"] = personality_id
        config["personality"]["system_prompt"] = personality_data.get("system_prompt", "")
        config["personality"]["temperature"] = personality_data.get("temperature", 0.3)
        config["personality"]["traits"] = personality_data.get("traits", [])
        config["personality"]["specialties"] = personality_data.get("specialties", [])
        
        # Update assistant temperature to match personality
        if "assistant" not in config:
            config["assistant"] = {}
        config["assistant"]["temperature"] = personality_data.get("temperature", 0.3)
        
        return True
    
    def generate_system_prompt(self, assistant_name: str, template_id: str = "base_assistant") -> Optional[str]:
        """Generate system prompt for an assistant."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return None
        
        return self.prompt_config.generate_system_prompt(config, template_id)
    
    def update_assistant_setting(self, assistant_name: str, section: str, 
                                key: str, value: Any) -> bool:
        """Update a specific assistant setting."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Navigate to the section
        if section not in config:
            config[section] = {}
        
        # Update the value
        config[section][key] = value
        config["assistant"]["updated_at"] = self._get_timestamp()
        
        # Save configuration
        config_file = f"{assistant_name}.assistant.toml"
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Updated {section}.{key} for '{assistant_name}': {value}[/]")
        
        return success
    
    def get_assistant_setting(self, assistant_name: str, section: str, 
                             key: str, default: Any = None) -> Any:
        """Get a specific assistant setting."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            return default
        
        return config.get(section, {}).get(key, default)
    
    def clone_assistant(self, source_name: str, target_name: str, 
                       new_working_dir: Path = None) -> bool:
        """Clone an assistant configuration."""
        source_config = self.get_assistant(source_name)
        
        if not source_config:
            console.print(f"[red]❌ Source assistant '{source_name}' not found[/]")
            return False
        
        # Clone configuration
        cloned_config = source_config.copy()
        
        # Update assistant-specific settings
        cloned_config["assistant"]["name"] = target_name
        cloned_config["assistant"]["created_at"] = self._get_timestamp()
        
        if "updated_at" in cloned_config["assistant"]:
            del cloned_config["assistant"]["updated_at"]
        
        # Update working directory if provided
        if new_working_dir:
            working_dir = Path(new_working_dir).absolute()
            cloned_config["assistant"]["working_dir"] = str(working_dir)
            cloned_config["image"]["output_dir"] = f"{working_dir}/generated_images"
        
        # Copy RAG files but keep them independent
        # Each assistant can have different RAG files loaded
        
        # Save cloned configuration
        config_file = f"{target_name}.assistant.toml"
        success = self.save_toml(config_file, cloned_config)
        
        if success:
            console.print(f"[green]✅ Cloned assistant '{source_name}' to '{target_name}'[/]")
        
        return success
    
    def delete_assistant(self, assistant_name: str) -> bool:
        """Delete an assistant configuration."""
        config_file = f"{assistant_name}.assistant.toml"
        success = self.delete_config(config_file)
        
        if success:
            console.print(f"[green]✅ Deleted assistant '{assistant_name}'[/]")
        
        return success
    
    def _create_assistant_directories(self, assistant_name: str):
        """Create directory structure for an assistant."""
        base_dir = Path(f".ai_context/{assistant_name}")
        
        # Create assistant-specific directories
        directories = [
            base_dir,
            base_dir / "voice",      # For voice recordings/cache
            base_dir / "context",    # For conversation context
            base_dir / "models",     # For assistant-specific model cache
            base_dir / "logs"        # For assistant logs
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create global knowledge directory at project root (shared across assistants)
        knowledge_dir = Path("knowledge")
        knowledge_dir.mkdir(exist_ok=True)
        
        console.print(f"[dim]📁 Created assistant directories in .ai_context/{assistant_name}/[/]")
        console.print(f"[dim]📚 Ensured global knowledge directory: knowledge/[/]")
    
    def get_assistant_context_dir(self, assistant_name: str) -> Path:
        """Get the context directory for an assistant."""
        return Path(f".ai_context/{assistant_name}")
    
    def get_global_knowledge_dir(self) -> Path:
        """Get the global knowledge directory (shared across all assistants)."""
        return Path("knowledge")
    
    def get_session_images_dir(self, session_working_dir: Path, subdir: str = "generated_images") -> Path:
        """Get the images directory for current session (in working directory)."""
        return Path(session_working_dir) / subdir
    
    def get_assistant_voice_dir(self, assistant_name: str) -> Path:
        """Get the voice directory for an assistant."""
        return self.get_assistant_context_dir(assistant_name) / "voice"
    
    def get_assistant_models_dir(self, assistant_name: str) -> Path:
        """Get the models directory for an assistant."""
        return self.get_assistant_context_dir(assistant_name) / "models"
    
    def set_assistant_model_preference(self, assistant_name: str, model_type: str, models: List[str]) -> bool:
        """Set model preferences for an assistant in their context directory."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return False
        
        # Update model preferences based on type
        if model_type == "chat":
            config["assistant"]["model"] = models[0] if models else config["assistant"]["model"]
        elif model_type == "image":
            config["image"]["models"] = models
        elif model_type == "voice":
            if models:
                config["voice"]["voice_id"] = models[0]
        
        config["assistant"]["updated_at"] = self._get_timestamp()
        
        # Save configuration
        config_file = f"{assistant_name}.assistant.toml"
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Updated {model_type} models for '{assistant_name}': {', '.join(models)}[/]")
        
        return success
    
    def get_assistant_model_preferences(self, assistant_name: str) -> Dict[str, Any]:
        """Get model preferences for an assistant."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            return {}
        
        return {
            "chat_model": config.get("assistant", {}).get("model", ""),
            "image_models": config.get("image", {}).get("models", []),
            "voice_id": config.get("voice", {}).get("voice_id", ""),
            "speech_backend": config.get("voice", {}).get("speech_backend", "google"),
            "rag_files": config.get("rag", {}).get("rag_files", [])
        }
    
    def load_assistant_for_session(self, assistant_name: str, session_working_dir: Path = None) -> Dict[str, Any]:
        """Load assistant configuration for a session with optional working directory override."""
        config = self.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return {}
        
        # Create session context
        session_config = config.copy()
        
        # Override working directory for this session if provided
        if session_working_dir:
            session_working_dir = Path(session_working_dir).absolute()
            session_config["session"] = {
                "working_dir": str(session_working_dir),
                "started_at": self._get_timestamp()
            }
            console.print(f"[cyan]📁 Session working directory: {session_working_dir}[/]")
        else:
            # Use default working directory
            default_working_dir = Path(config["assistant"]["working_dir"])
            session_config["session"] = {
                "working_dir": str(default_working_dir),
                "started_at": self._get_timestamp()
            }
        
        # Set assistant context directories
        session_working_dir = Path(session_config["session"]["working_dir"])
        session_config["paths"] = {
            "context_dir": str(self.get_assistant_context_dir(assistant_name)),
            "knowledge_dir": str(self.get_global_knowledge_dir()),  # Global knowledge directory
            "images_dir": str(self.get_session_images_dir(session_working_dir)),  # Images in working dir
            "voice_dir": str(self.get_assistant_voice_dir(assistant_name)),
            "models_dir": str(self.get_assistant_models_dir(assistant_name))
        }
        
        console.print(f"[green]🤖 Loaded assistant '{assistant_name}' for session[/]")
        console.print(f"[dim]Context: {session_config['paths']['context_dir']}[/]")
        
        return session_config
    
    def cleanup_assistant_data(self, assistant_name: str, keep_config: bool = True) -> bool:
        """Clean up assistant data (context, cache, etc.) but optionally keep configuration."""
        context_dir = self.get_assistant_context_dir(assistant_name)
        
        if not context_dir.exists():
            console.print(f"[yellow]⚠️ No context directory found for '{assistant_name}'[/]")
            return True
        
        try:
            import shutil
            
            # Remove specific subdirectories but keep structure
            cleanup_dirs = ["context", "logs", "voice", "images"]
            
            for cleanup_dir in cleanup_dirs:
                dir_path = context_dir / cleanup_dir
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(exist_ok=True)  # Recreate empty directory
            
            # Optionally remove configuration
            if not keep_config:
                config_file = self.config_dir / f"{assistant_name}.assistant.toml"
                if config_file.exists():
                    config_file.unlink()
            
            console.print(f"[green]✅ Cleaned up assistant data for '{assistant_name}'[/]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error cleaning up assistant data: {e}[/]")
            return False
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()