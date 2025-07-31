"""
Model management utilities.

Handles AI model installation, configuration, and management.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table

from library.model_cfg import ModelConfig

console = Console()


class ModelManager:
    """Manages AI models and model configurations."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize model manager."""
        self.config_dir = config_dir or Path("configuration")
        self.model_config = ModelConfig(self.config_dir)
    
    def list_models(self, model_type: str = "models"):
        """List available models."""
        self.model_config.display_models(model_type)
    
    def list_ollama_models(self):
        """List actually downloaded Ollama models."""
        try:
            import ollama
            models = ollama.list()
            
            if not models.get('models'):
                console.print("[yellow]⚠️ No Ollama models found. Download models first.[/]")
                console.print("[dim]Example: ollama pull qwen2.5-coder:3b[/]")
                return []
            
            from rich.table import Table
            table = Table(title="Downloaded Ollama Models")
            table.add_column("Model", style="cyan", width=30)
            table.add_column("Size", style="green", width=12)
            table.add_column("Modified", style="dim", width=15)
            
            for model in models['models']:
                # Handle different possible model object structures
                if hasattr(model, 'name'):
                    name = model.name
                elif hasattr(model, 'model'):
                    name = model.model
                elif isinstance(model, dict):
                    name = model.get('name') or model.get('model', 'Unknown')
                else:
                    name = str(model)
                
                # Handle size
                if hasattr(model, 'size'):
                    size_bytes = model.size
                elif isinstance(model, dict):
                    size_bytes = model.get('size', 0)
                else:
                    size_bytes = 0
                    
                size = f"{size_bytes / (1024**3):.1f}GB" if size_bytes else 'Unknown'
                
                # Handle modified date
                modified_at = None
                if hasattr(model, 'modified_at'):
                    modified_at = model.modified_at
                elif isinstance(model, dict):
                    modified_at = model.get('modified_at')
                
                if modified_at:
                    modified = str(modified_at)[:10]  # Just date part
                else:
                    modified = 'Unknown'
                    
                table.add_row(name, size, modified)
            
            console.print(table)
            
            # Return list of model names
            model_names = []
            for model in models['models']:
                if hasattr(model, 'name'):
                    model_names.append(model.name)
                elif hasattr(model, 'model'):
                    model_names.append(model.model)
                elif isinstance(model, dict):
                    model_names.append(model.get('name') or model.get('model', ''))
                else:
                    model_names.append(str(model))
            return model_names
            
        except ImportError:
            console.print("[red]❌ Ollama not installed or not accessible[/]")
            return []
        except Exception as e:
            console.print(f"[red]❌ Error listing Ollama models: {e}[/]")
            return []
    
    def list_huggingface_models(self):
        """List cached HuggingFace models."""
        try:
            from pathlib import Path
            import os
            
            # Common HuggingFace cache locations
            cache_dirs = [
                Path.home() / ".cache" / "huggingface" / "hub",
                Path.home() / ".cache" / "huggingface" / "transformers",
                Path(os.environ.get('HF_HOME', '')) / "hub" if os.environ.get('HF_HOME') else None
            ]
            
            models = []
            
            for cache_dir in cache_dirs:
                if not cache_dir or not cache_dir.exists():
                    continue
                    
                # Look for model directories
                for item in cache_dir.iterdir():
                    if not item.is_dir():
                        continue
                    
                    # Skip if already processed
                    if any(m['path'] == item for m in models):
                        continue
                    
                    # Parse model name from directory structure
                    model_name = "Unknown"
                    if "models--" in item.name:
                        # Format: models--organizationname--modelname
                        parts = item.name.split("models--", 1)
                        if len(parts) > 1:
                            model_name = parts[1].replace("--", "/")
                    elif "snapshots" not in item.name and "blobs" not in item.name:
                        model_name = item.name
                    else:
                        continue  # Skip non-model directories
                    
                    # Calculate directory size
                    total_size = 0
                    try:
                        for file_path in item.rglob("*"):
                            if file_path.is_file():
                                total_size += file_path.stat().st_size
                    except (OSError, PermissionError):
                        total_size = 0
                    
                    # Get modification time
                    try:
                        modified = item.stat().st_mtime
                        import datetime
                        modified_str = datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d')
                    except:
                        modified_str = "Unknown"
                    
                    models.append({
                        'name': model_name,
                        'path': item,
                        'size': total_size,
                        'modified': modified_str,
                        'type': self._guess_model_type(model_name)
                    })
            
            if not models:
                console.print("[yellow]⚠️ No HuggingFace models found in cache[/]")
                return []
            
            # Sort by size (largest first)
            models.sort(key=lambda x: x['size'], reverse=True)
            
            from rich.table import Table
            table = Table(title="Cached HuggingFace Models")
            table.add_column("Model", style="cyan", width=35)
            table.add_column("Type", style="magenta", width=12)
            table.add_column("Size", style="green", width=10)
            table.add_column("Modified", style="dim", width=12)
            
            for model in models:
                size_str = self._format_size(model['size'])
                table.add_row(
                    model['name'][:34] + "..." if len(model['name']) > 34 else model['name'],
                    model['type'],
                    size_str,
                    model['modified']
                )
            
            console.print(table)
            return [model['name'] for model in models]
            
        except Exception as e:
            console.print(f"[red]❌ Error scanning HuggingFace cache: {e}[/]")
            return []
    
    def _guess_model_type(self, model_name: str) -> str:
        """Guess model type from name."""
        name_lower = model_name.lower()
        if any(term in name_lower for term in ['stable-diffusion', 'sd-', 'diffusion', 'dalle']):
            return "Image"
        elif any(term in name_lower for term in ['whisper', 'wav2vec', 'speech']):
            return "Speech"  
        elif any(term in name_lower for term in ['bert', 'roberta', 'gpt', 'llama', 'mistral', 'qwen']):
            return "Text"
        elif any(term in name_lower for term in ['clip', 'blip', 'vision']):
            return "Vision"
        else:
            return "Other"
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes to human readable size."""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}PB"
    
    def list_all_models(self):
        """List downloaded Ollama models, HuggingFace cached models."""
        console.print("[bold cyan]🤖 AI Models[/]\n")
        
        # First show actually downloaded Ollama models
        console.print("[bold green]Downloaded Models (Ollama):[/]")
        ollama_models = self.list_ollama_models()
        
        if ollama_models:
            console.print()
            
        # Show HuggingFace cached models
        console.print("[bold magenta]Cached Models (HuggingFace):[/]")
        hf_models = self.list_huggingface_models()
        
        if hf_models:
            console.print()
        
        # Show download suggestions or summary
        total_models = len(ollama_models) + len(hf_models)
        
        if total_models == 0:
            console.print("[bold yellow]💡 Popular Models to Download:[/]")
            console.print("[dim]Ollama models (use 'ollama pull <model>'):[/]")
            console.print("  • [cyan]qwen2.5-coder:3b[/] - Great for coding tasks (~1.9GB)")
            console.print("  • [cyan]llama3.2:3b[/] - General purpose model (~2.0GB)")
            console.print("  • [cyan]codellama:7b[/] - Specialized for code (~3.8GB)")
            console.print("  • [cyan]mistral:7b[/] - Fast and efficient (~4.1GB)")
            console.print()
            console.print("[dim]HuggingFace models (auto-downloaded when needed):[/]")
            console.print("  • [magenta]Stable Diffusion models[/] - For image generation")
            console.print("  • [magenta]Whisper models[/] - For speech recognition")
            console.print()
            console.print("[dim]Example: [/][yellow]ollama pull qwen2.5-coder:3b[/]")
        else:
            ollama_count = len(ollama_models)
            hf_count = len(hf_models)
            console.print(f"[dim]✅ You have {total_models} model(s) cached:[/]")
            if ollama_count > 0:
                console.print(f"[dim]   • {ollama_count} Ollama model(s) ready to use[/]")
            if hf_count > 0:
                console.print(f"[dim]   • {hf_count} HuggingFace model(s) cached[/]")
        
        return True
    
    def clean_hf_cache(self) -> bool:
        """Clean HuggingFace model cache."""
        try:
            from pathlib import Path
            import shutil
            import os
            
            cache_dirs = [
                Path.home() / ".cache" / "huggingface",
                Path(os.environ.get('HF_HOME', '')) if os.environ.get('HF_HOME') else None
            ]
            
            total_freed = 0
            cleaned_dirs = []
            
            for cache_dir in cache_dirs:
                if not cache_dir or not cache_dir.exists():
                    continue
                
                # Calculate size before deletion
                cache_size = 0
                try:
                    for file_path in cache_dir.rglob("*"):
                        if file_path.is_file():
                            cache_size += file_path.stat().st_size
                except (OSError, PermissionError):
                    pass
                
                if cache_size > 0:
                    from rich.prompt import Confirm
                    size_str = self._format_size(cache_size)
                    if Confirm.ask(f"Delete HuggingFace cache at {cache_dir} ({size_str})?"):
                        try:
                            shutil.rmtree(cache_dir)
                            total_freed += cache_size
                            cleaned_dirs.append(str(cache_dir))
                            console.print(f"[green]✅ Deleted {cache_dir}[/]")
                        except Exception as e:
                            console.print(f"[red]❌ Failed to delete {cache_dir}: {e}[/]")
            
            if total_freed > 0:
                console.print(f"[green]🗑️ Freed {self._format_size(total_freed)} of disk space[/]")
            else:
                console.print("[yellow]⚠️ No HuggingFace cache found to clean[/]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error cleaning HuggingFace cache: {e}[/]")
            return False
    
    def download_model(self, model_id: str) -> bool:
        """Download/pull a model."""
        try:
            import ollama
            
            console.print(f"[cyan]📥 Downloading model: {model_id}[/]")
            
            # Pull the model
            ollama.pull(model_id)
            
            console.print(f"[green]✅ Successfully downloaded {model_id}[/]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to download {model_id}: {e}[/]")
            return False
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model."""
        try:
            import ollama
            
            console.print(f"[yellow]🗑️ Removing model: {model_id}[/]")
            
            # Delete the model
            ollama.delete(model_id)
            
            console.print(f"[green]✅ Successfully removed {model_id}[/]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to remove {model_id}: {e}[/]")
            return False
    
    def set_cache_dir(self, cache_dir: str) -> bool:
        """Set HuggingFace cache directory."""
        import os
        
        cache_path = Path(cache_dir).absolute()
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable
        os.environ['HF_HOME'] = str(cache_path)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_path)
        
        console.print(f"[green]✅ Set HuggingFace cache directory: {cache_path}[/]")
        console.print("[dim]Note: Restart the application for changes to take effect[/]")
        
        return True
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        # Check in different model types
        for model_type in ["models", "image_models", "speech_models"]:
            model_config = self.model_config.get_model_config(model_id, model_type)
            if model_config:
                model_config["model_type"] = model_type
                return model_config
        
        console.print(f"[red]❌ Model '{model_id}' not found in configuration[/]")
        return None
    
    def show_model_info(self, model_id: str):
        """Display detailed model information."""
        model_info = self.get_model_info(model_id)
        
        if not model_info:
            return
        
        info_text = f"""[bold cyan]{model_info.get('name', model_id)}[/]

[bold]Basic Information:[/]
• ID: {model_id}
• Type: {model_info.get('type', 'Unknown')}
• Provider: {model_info.get('provider', 'Unknown')}
• Model Type: {model_info.get('model_type', 'Unknown')}
• Description: {model_info.get('description', 'No description')}

[bold]Specifications:[/]
• Memory Usage: {model_info.get('memory_usage', 'Unknown')}
• Context Window: {model_info.get('context_window', 'Unknown')}
• Requires GPU: {'Yes' if model_info.get('requires_gpu') else 'No'}
• Requires Internet: {'Yes' if model_info.get('requires_internet') else 'No'}

[bold]Capabilities:[/]
• {', '.join(model_info.get('capabilities', []))}

[bold]Recommended For:[/]
• {', '.join(model_info.get('recommended_for', []))}
"""
        
        # Add parameters if available
        if 'parameters' in model_info:
            info_text += "\n[bold]Default Parameters:[/]\n"
            for param, value in model_info['parameters'].items():
                info_text += f"• {param}: {value}\n"
        
        from rich.panel import Panel
        console.print(Panel(info_text, title=f"Model Information: {model_id}"))
    
    def add_model_config(self, model_id: str, model_data: Dict[str, Any], 
                        model_type: str = "models") -> bool:
        """Add a new model configuration."""
        return self.model_config.add_model(model_id, model_data, model_type)
    
    def update_model_config(self, model_id: str, updates: Dict[str, Any],
                           model_type: str = "models") -> bool:
        """Update an existing model configuration."""
        return self.model_config.update_model(model_id, updates, model_type)
    
    def get_recommended_models(self, task: str) -> List[Dict[str, Any]]:
        """Get models recommended for a specific task."""
        return self.model_config.get_recommended_models(task)
    
    def show_recommended_models(self, task: str):
        """Display models recommended for a task."""
        models = self.get_recommended_models(task)
        
        if not models:
            console.print(f"[yellow]⚠️ No models found for task: {task}[/]")
            return
        
        table = Table(title=f"Recommended Models for: {task}")
        table.add_column("Model ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Memory", style="yellow")
        table.add_column("Provider", style="magenta")
        table.add_column("Description", style="dim")
        
        for model in models:
            table.add_row(
                model["id"],
                model.get("name", model["id"])[:20],
                model.get("memory_usage", "Unknown"),
                model.get("provider", "Unknown"),
                model.get("description", "No description")[:40] + "..."
            )
        
        console.print(table)