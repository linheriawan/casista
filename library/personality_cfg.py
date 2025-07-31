"""
Personality configuration management.

Handles predefined personalities and custom personality creation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table

from .config_loader import ConfigLoader

console = Console()


class PersonalityConfig(ConfigLoader):
    """Manages AI assistant personalities and traits."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize personality configuration manager."""
        super().__init__(config_dir)
        
        # Check if default personalities exist
        if not self.config_exists("default.personality.toml"):
            console.print(f"[yellow]⚠️ Default personality config not found, please ensure default.personality.toml exists[/]")
    
    def get_personality(self, personality_id: str, config_file: str = "default.personality.toml") -> Optional[Dict[str, Any]]:
        """Get a personality by ID."""
        config = self.load_toml(config_file)
        return config.get(personality_id)
    
    def list_personalities(self, config_file: str = "default.personality.toml") -> List[Dict[str, str]]:
        """List all available personalities."""
        config = self.load_toml(config_file)
        personalities = []
        
        for personality_id, personality_data in config.items():
            if isinstance(personality_data, dict):
                personalities.append({
                    "id": personality_id,
                    "name": personality_data.get("name", personality_id),
                    "description": personality_data.get("description", "No description"),
                    "temperature": personality_data.get("temperature", 0.3),
                    "traits": personality_data.get("traits", []),
                    "specialties": personality_data.get("specialties", [])
                })
        
        return sorted(personalities, key=lambda x: x["name"])
    
    def display_personalities(self, config_file: str = "default.personality.toml"):
        """Display available personalities in a table."""
        personalities = self.list_personalities(config_file)
        
        if not personalities:
            console.print("[yellow]⚠️ No personalities found[/]")
            return True
        
        table = Table(title="Available Personalities")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description", style="dim")
        table.add_column("Specialties", style="yellow")
        table.add_column("Temp", style="magenta")
        
        for personality in personalities:
            specialties = ", ".join(personality["specialties"][:3])  # Show first 3
            if len(personality["specialties"]) > 3:
                specialties += "..."
            
            table.add_row(
                personality["id"],
                personality["name"],
                personality["description"][:40] + "..." if len(personality["description"]) > 40 else personality["description"],
                specialties,
                str(personality["temperature"])
            )
        
        console.print(table)
        return True
    
    def create_personality(self, personality_id: str, name: str, description: str,
                          system_prompt: str, traits: List[str] = None,
                          specialties: List[str] = None, temperature: float = 0.3,
                          config_file: str = "custom.personality.toml") -> bool:
        """Create a new personality."""
        # Load existing config or create new
        config = self.load_toml(config_file) if self.config_exists(config_file) else {}
        
        personality_data = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "traits": traits or [],
            "specialties": specialties or [],
            "temperature": temperature,
            "created_at": self._get_timestamp()
        }
        
        config[personality_id] = personality_data
        
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Personality '{personality_id}' created in {config_file}[/]")
        
        return success
    
    def update_personality(self, personality_id: str, updates: Dict[str, Any],
                          config_file: str = "custom.personality.toml") -> bool:
        """Update an existing personality."""
        config = self.load_toml(config_file)
        
        if personality_id not in config:
            console.print(f"[red]❌ Personality '{personality_id}' not found in {config_file}[/]")
            return False
        
        # Merge updates
        for key, value in updates.items():
            config[personality_id][key] = value
        
        config[personality_id]["updated_at"] = self._get_timestamp()
        
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Personality '{personality_id}' updated in {config_file}[/]")
        
        return success
    
    def delete_personality(self, personality_id: str, config_file: str = "custom.personality.toml") -> bool:
        """Delete a personality."""
        config = self.load_toml(config_file)
        
        if personality_id not in config:
            console.print(f"[red]❌ Personality '{personality_id}' not found in {config_file}[/]")
            return False
        
        del config[personality_id]
        
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Personality '{personality_id}' deleted from {config_file}[/]")
        
        return success
    
    def copy_personality(self, source_id: str, target_id: str, new_name: str = None,
                        source_file: str = "default.personality.toml",
                        target_file: str = "custom.personality.toml") -> bool:
        """Copy a personality to create a new one."""
        source_data = self.get_personality(source_id, source_file)
        
        if not source_data:
            console.print(f"[red]❌ Source personality '{source_id}' not found in {source_file}[/]")
            return False
        
        # Load target config
        target_config = self.load_toml(target_file) if self.config_exists(target_file) else {}
        
        # Update metadata for the copy
        copied_data = source_data.copy()
        copied_data["name"] = new_name or f"{source_data.get('name', source_id)} (Copy)"
        copied_data["created_at"] = self._get_timestamp()
        
        # Remove update timestamp if it exists
        if "updated_at" in copied_data:
            del copied_data["updated_at"]
        
        target_config[target_id] = copied_data
        success = self.save_toml(target_file, target_config)
        
        if success:
            console.print(f"[green]✅ Personality copied from '{source_id}' to '{target_id}' in {target_file}[/]")
        
        return success
    
    def list_personality_configs(self) -> List[str]:
        """List all personality configuration files."""
        return self.list_configs("*.personality.toml")
    
    def get_personalities_by_trait(self, trait: str) -> List[Dict[str, Any]]:
        """Get personalities that have a specific trait."""
        personalities = self.list_personalities()
        matching = []
        
        for personality in personalities:
            traits = personality.get("traits", [])
            if trait in traits:
                matching.append(personality)
        
        return matching
    
    def get_personalities_by_specialty(self, specialty: str) -> List[Dict[str, Any]]:
        """Get personalities that have a specific specialty."""
        personalities = self.list_personalities()
        matching = []
        
        for personality in personalities:
            specialties = personality.get("specialties", [])
            if specialty in specialties:
                matching.append(personality)
        
        return matching
    
    def validate_personality(self, personality_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate personality configuration."""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["name", "description", "system_prompt"]
        for field in required_fields:
            if field not in personality_data or not personality_data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate system prompt
        if "system_prompt" in personality_data:
            prompt = personality_data["system_prompt"]
            if len(prompt) < 50:
                warnings.append("System prompt seems too short for a personality")
            elif len(prompt) > 2000:
                warnings.append("System prompt is very long, may be truncated")
        
        # Validate temperature
        if "temperature" in personality_data:
            temp = personality_data["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                errors.append("Temperature must be between 0 and 2")
        
        # Validate traits and specialties
        for field in ["traits", "specialties"]:
            if field in personality_data:
                if not isinstance(personality_data[field], list):
                    errors.append(f"{field} must be a list")
                elif len(personality_data[field]) > 10:
                    warnings.append(f"Large number of {field}, consider reducing")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def create_custom_personality_config(self, filename: str, personalities_data: Dict[str, Any]) -> bool:
        """Create a custom personality configuration file."""
        if not filename.endswith('.personality.toml'):
            filename += '.personality.toml'
        
        success = self.save_toml(filename, personalities_data)
        
        if success:
            console.print(f"[green]✅ Created custom personality config: {filename}[/]")
        
        return success
        
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()