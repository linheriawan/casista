"""
Model configuration management.

Handles AI model settings, parameters, and model-specific configurations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table

from .config_loader import ConfigLoader

console = Console()


class ModelConfig(ConfigLoader):
    """Manages AI model configurations and settings."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize model configuration manager."""
        super().__init__(config_dir)
        
        # Initialize default model configurations if they don't exist
        if not self.config_exists("default.model.toml"):
            console.print(f"[yellow]⚠️ Default model config not found, please ensure default.model.toml exists[/]")
    
    def get_model_config(self, model_id: str, model_type: str = "models") -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        config = self.load_toml("default.model.toml")
        return config.get(model_type, {}).get(model_id)
    
    def list_models(self, model_type: str = "models") -> List[Dict[str, Any]]:
        """List all models of a specific type."""
        config = self.load_toml("default.model.toml")
        models = []
        
        for model_id, model_config in config.get(model_type, {}).items():
            model_info = {
                "id": model_id,
                **model_config
            }
            models.append(model_info)
        
        return sorted(models, key=lambda x: x.get("name", x["id"]))
    
    def display_models(self, model_type: str = "models"):
        """Display models in a formatted table."""
        models = self.list_models(model_type)
        
        if not models:
            console.print(f"[yellow]⚠️ No {model_type} found[/]")
            return
        
        table = Table(title=f"Available {model_type.title()}")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Provider", style="yellow")
        table.add_column("Description", style="dim")
        table.add_column("Memory", style="magenta")
        
        for model in models:
            description = model.get("description", "No description")
            if len(description) > 50:
                description = description[:47] + "..."
            
            table.add_row(
                model["id"],
                model.get("name", model["id"]),
                model.get("provider", "unknown"),
                description,
                model.get("memory_usage", "unknown")
            )
        
        console.print(table)
    
    def add_model(self, model_id: str, model_config: Dict[str, Any],
                  model_type: str = "models", config_file: str = "default.model.toml") -> bool:
        """Add a new model configuration."""
        config = self.load_toml(config_file)
        
        if model_type not in config:
            config[model_type] = {}
        
        config[model_type][model_id] = model_config
        
        success = self.save_toml(config_file, config)
        if success:
            console.print(f"[green]✅ Added model '{model_id}' to {model_type}[/]")
        
        return success
    
    def update_model(self, model_id: str, updates: Dict[str, Any],
                    model_type: str = "models", config_file: str = "default.model.toml") -> bool:
        """Update an existing model configuration."""
        config = self.load_toml(config_file)
        
        if model_type not in config or model_id not in config[model_type]:
            console.print(f"[red]❌ Model '{model_id}' not found in {model_type}[/]")
            return False
        
        # Merge updates
        for key, value in updates.items():
            if key == "parameters" and isinstance(value, dict):
                # Merge parameters instead of replacing
                if "parameters" not in config[model_type][model_id]:
                    config[model_type][model_id]["parameters"] = {}
                config[model_type][model_id]["parameters"].update(value)
            else:
                config[model_type][model_id][key] = value
        
        success = self.save_toml(config_file, config)
        if success:
            console.print(f"[green]✅ Updated model '{model_id}' in {model_type}[/]")
        
        return success
    
    def remove_model(self, model_id: str, model_type: str = "models",
                    config_file: str = "default.model.toml") -> bool:
        """Remove a model configuration."""
        config = self.load_toml(config_file)
        
        if model_type not in config or model_id not in config[model_type]:
            console.print(f"[red]❌ Model '{model_id}' not found in {model_type}[/]")
            return False
        
        del config[model_type][model_id]
        
        success = self.save_toml(config_file, config)
        if success:
            console.print(f"[green]✅ Removed model '{model_id}' from {model_type}[/]")
        
        return success
    
    def get_recommended_models(self, task: str) -> List[Dict[str, Any]]:
        """Get models recommended for a specific task."""
        models = self.list_models()
        recommended = []
        
        for model in models:
            recommended_for = model.get("recommended_for", [])
            if task in recommended_for or any(task in rec for rec in recommended_for):
                recommended.append(model)
        
        return sorted(recommended, key=lambda x: x.get("memory_usage", "high"))
    
    def get_model_parameters(self, model_id: str, model_type: str = "models") -> Dict[str, Any]:
        """Get default parameters for a model."""
        model_config = self.get_model_config(model_id, model_type)
        
        if not model_config:
            return {}
        
        return model_config.get("parameters", {})
    
    def validate_model_config(self, model_config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate a model configuration."""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["name", "type", "provider"]
        for field in required_fields:
            if field not in model_config:
                errors.append(f"Missing required field: {field}")
        
        # Validate type
        valid_types = ["chat", "image_generation", "speech_to_text", "text_to_speech"]
        if "type" in model_config and model_config["type"] not in valid_types:
            errors.append(f"Invalid model type. Must be one of: {', '.join(valid_types)}")
        
        # Validate parameters
        if "parameters" in model_config:
            params = model_config["parameters"]
            
            # Check temperature
            if "temperature" in params:
                temp = params["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    errors.append("Temperature must be between 0 and 2")
            
            # Check max_tokens
            if "max_tokens" in params:
                tokens = params["max_tokens"]
                if not isinstance(tokens, int) or tokens < 1:
                    errors.append("max_tokens must be a positive integer")
        
        # Check memory usage
        if "memory_usage" in model_config:
            valid_memory = ["low", "medium", "high", "very_high"]
            if model_config["memory_usage"] not in valid_memory:
                warnings.append(f"Unknown memory usage level: {model_config['memory_usage']}")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def create_custom_model_config(self, filename: str, models_data: Dict[str, Any]) -> bool:
        """Create a custom model configuration file."""
        if not filename.endswith('.model.toml'):
            filename += '.model.toml'
        
        success = self.save_toml(filename, models_data)
        
        if success:
            console.print(f"[green]✅ Created custom model config: {filename}[/]")
        
        return success
    
    def load_custom_model_config(self, filename: str) -> Dict[str, Any]:
        """Load a custom model configuration file."""
        if not filename.endswith('.model.toml'):
            filename += '.model.toml'
        
        return self.load_toml(filename)
    
    def list_model_configs(self) -> List[str]:
        """List all model configuration files."""
        return self.list_configs("*.model.toml")
    
    def get_model_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get models that have a specific capability."""
        models = self.list_models()
        matching = []
        
        for model in models:
            capabilities = model.get("capabilities", [])
            if capability in capabilities:
                matching.append(model)
        
        return matching
    
    def get_models_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Get all models from a specific provider."""
        models = self.list_models()
        return [model for model in models if model.get("provider") == provider]