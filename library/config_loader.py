"""
Base TOML configuration loader.

Handles generic TOML file operations for all configuration types.
"""

import toml
from pathlib import Path
from typing import Dict, Any, Optional, List
from rich.console import Console

console = Console()


class ConfigLoader:
    """Generic TOML configuration file loader and manager."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize config loader."""
        self.config_dir = config_dir or Path("configuration")
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_toml(self, filename: str) -> Dict[str, Any]:
        """Load a TOML configuration file."""
        config_file = self.config_dir / filename
        
        if not config_file.exists():
            console.print(f"[yellow]⚠️ Config file not found: {filename}[/]")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = toml.load(f)
            return config
        except Exception as e:
            console.print(f"[red]❌ Error loading {filename}: {e}[/]")
            return {}
    
    def save_toml(self, filename: str, config: Dict[str, Any]) -> bool:
        """Save configuration to TOML file."""
        config_file = self.config_dir / filename
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                toml.dump(config, f)
            return True
        except Exception as e:
            console.print(f"[red]❌ Error saving {filename}: {e}[/]")
            return False
    
    def list_configs(self, pattern: str = "*.toml") -> List[str]:
        """List all configuration files matching pattern."""
        try:
            return [f.name for f in self.config_dir.glob(pattern)]
        except Exception as e:
            console.print(f"[red]❌ Error listing configs: {e}[/]")
            return []
    
    def config_exists(self, filename: str) -> bool:
        """Check if configuration file exists."""
        return (self.config_dir / filename).exists()
    
    def delete_config(self, filename: str) -> bool:
        """Delete a configuration file."""
        config_file = self.config_dir / filename
        
        if not config_file.exists():
            console.print(f"[yellow]⚠️ Config file not found: {filename}[/]")
            return False
        
        try:
            config_file.unlink()
            console.print(f"[green]✅ Deleted config: {filename}[/]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Error deleting {filename}: {e}[/]")
            return False
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def validate_toml_structure(self, config: Dict[str, Any], 
                               required_keys: List[str] = None) -> Dict[str, List[str]]:
        """Validate TOML configuration structure."""
        errors = []
        warnings = []
        
        if required_keys:
            for key in required_keys:
                if key not in config:
                    errors.append(f"Missing required key: {key}")
        
        # Check for empty config
        if not config:
            warnings.append("Configuration is empty")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def backup_config(self, filename: str) -> Optional[str]:
        """Backup a configuration file."""
        config_file = self.config_dir / filename
        
        if not config_file.exists():
            return None
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filename}.backup_{timestamp}"
            backup_file = self.config_dir / backup_name
            
            config = self.load_toml(filename)
            self.save_toml(backup_name, config)
            
            console.print(f"[green]✅ Backup created: {backup_name}[/]")
            return backup_name
            
        except Exception as e:
            console.print(f"[red]❌ Error creating backup: {e}[/]")
            return None
    
    def restore_config(self, backup_filename: str, target_filename: str) -> bool:
        """Restore configuration from backup."""
        backup_config = self.load_toml(backup_filename)
        
        if not backup_config:
            console.print(f"[red]❌ Backup file empty or not found: {backup_filename}[/]")
            return False
        
        success = self.save_toml(target_filename, backup_config)
        
        if success:
            console.print(f"[green]✅ Restored {target_filename} from {backup_filename}[/]")
        
        return success
    
    def get_config_info(self, filename: str) -> Dict[str, Any]:
        """Get information about a configuration file."""
        config_file = self.config_dir / filename
        
        info = {
            "filename": filename,
            "exists": config_file.exists(),
            "path": str(config_file)
        }
        
        if config_file.exists():
            try:
                stat = config_file.stat()
                info.update({
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "readable": True
                })
                
                # Try to load to check validity
                config = self.load_toml(filename)
                info["valid"] = bool(config is not None)
                info["keys"] = list(config.keys()) if config else []
                
            except Exception as e:
                info.update({
                    "readable": False,
                    "error": str(e)
                })
        
        return info