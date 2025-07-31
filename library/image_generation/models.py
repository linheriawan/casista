"""
Image model management functionality.

Handles downloading, caching, and managing image generation models.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from rich.console import Console
from rich.table import Table

console = Console()


class ImageModelManager:
    """Manages image generation models and cache."""
    
    def __init__(self, cache_dir: Path = None):
        """Initialize model manager."""
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "huggingface" / "hub")
    
    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List all cached image generation models."""
        if not self.cache_dir.exists():
            console.print("[yellow]⚠️ HuggingFace cache directory not found[/]")
            return []
        
        models = []
        
        # Look for model directories
        for item in self.cache_dir.iterdir():
            if item.is_dir() and self._is_image_model(item.name):
                # Calculate size
                total_size = 0
                try:
                    for file in item.rglob("*"):
                        if file.is_file():
                            total_size += file.stat().st_size
                except:
                    total_size = 0
                
                models.append({
                    'name': item.name,
                    'path': item,
                    'size': total_size,
                    'display_name': self._extract_model_name(item.name)
                })
        
        return sorted(models, key=lambda x: x['display_name'])
    
    def _is_image_model(self, model_name: str) -> bool:
        """Check if a model is likely an image generation model."""
        image_keywords = [
            "stable-diffusion", "tiny-sd", "openjourney", "dreamlike",
            "waifu", "anything-v", "deliberate", "realistic-vision",
            "counterfeit", "chilloutmix", "protogen", "midjourney",
            "dall-e", "imagen", "kandinsky", "flux"
        ]
        
        model_lower = model_name.lower()
        return any(keyword in model_lower for keyword in image_keywords)
    
    def _extract_model_name(self, full_name: str) -> str:
        """Extract readable model name from full path name."""
        if 'models--' in full_name:
            parts = full_name.split('models--')[1].replace('--', '/')
            return parts
        return full_name
    
    def format_size(self, size_bytes: int) -> str:
        """Format bytes to human readable."""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def display_models(self):
        """Display cached models in a table."""
        models = self.list_cached_models()
        
        if not models:
            console.print("[yellow]⚠️ No image generation models found in cache[/]")
            console.print("[dim]Models will be downloaded automatically when first used[/]")
            return
        
        table = Table(title="Cached Image Generation Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Status", style="yellow")
        
        for model in models:
            table.add_row(
                model['display_name'],
                self.format_size(model['size']),
                "✅ Cached"
            )
        
        console.print(table)
    
    def remove_model(self, model_name: str) -> bool:
        """Remove image model from cache."""
        models = self.list_cached_models()
        
        # Find matching models
        matches = [m for m in models if model_name.lower() in m['name'].lower() or 
                  model_name.lower() in m['display_name'].lower()]
        
        if not matches:
            console.print(f"[red]❌ No models found matching '{model_name}'[/]")
            return False
        
        success = True
        for model in matches:
            display_name = model['display_name']
            console.print(f"[yellow]Removing:[/] {display_name}")
            console.print(f"[dim]Size: {self.format_size(model['size'])}[/]")
            
            try:
                shutil.rmtree(model['path'])
                console.print(f"[green]✅ Removed {display_name} ({self.format_size(model['size'])} freed)[/]")
            except Exception as e:
                console.print(f"[red]❌ Failed to remove {display_name}: {e}[/]")
                success = False
        
        return success
    
    def preload_model(self, model_name: str) -> bool:
        """Preload/download image model."""
        console.print(f"[cyan]📥 Downloading model: {model_name}[/]")
        
        try:
            from diffusers import StableDiffusionPipeline
            import torch
            
            # Load the model (this will download and cache it)
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True
            )
            
            console.print(f"[green]✅ Successfully downloaded {model_name}[/]")
            console.print(f"[dim]Model is now cached and ready for use[/]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to download {model_name}: {e}[/]")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        models = self.list_cached_models()
        
        for model in models:
            if (model_name.lower() in model['name'].lower() or 
                model_name.lower() in model['display_name'].lower()):
                return model
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        models = self.list_cached_models()
        
        total_size = sum(model['size'] for model in models)
        
        return {
            'total_models': len(models),
            'total_size': total_size,
            'total_size_formatted': self.format_size(total_size),
            'cache_dir': str(self.cache_dir),
            'cache_exists': self.cache_dir.exists()
        }
    
    def cleanup_cache(self, max_size_gb: float = None) -> bool:
        """Cleanup cache by removing oldest models if size exceeds limit."""
        if max_size_gb is None:
            return False
        
        models = self.list_cached_models()
        total_size = sum(model['size'] for model in models)
        max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        if total_size <= max_size_bytes:
            console.print(f"[green]✅ Cache size ({self.format_size(total_size)}) is within limit[/]")
            return True
        
        # Sort by modification time (oldest first)
        models_with_mtime = []
        for model in models:
            try:
                mtime = model['path'].stat().st_mtime
                models_with_mtime.append((model, mtime))
            except:
                models_with_mtime.append((model, 0))
        
        models_with_mtime.sort(key=lambda x: x[1])
        
        # Remove oldest models until under limit
        current_size = total_size
        removed_count = 0
        
        for model, _ in models_with_mtime:
            if current_size <= max_size_bytes:
                break
            
            console.print(f"[yellow]Removing old model: {model['display_name']}[/]")
            try:
                shutil.rmtree(model['path'])
                current_size -= model['size']
                removed_count += 1
                console.print(f"[green]✅ Removed {model['display_name']}[/]")
            except Exception as e:
                console.print(f"[red]❌ Failed to remove {model['display_name']}: {e}[/]")
        
        console.print(f"[green]✅ Cleaned up {removed_count} models[/]")
        console.print(f"[green]Cache size reduced from {self.format_size(total_size)} to {self.format_size(current_size)}[/]")
        
        return True
    
    def verify_model(self, model_name: str) -> bool:
        """Verify that a model is properly cached and loadable."""
        try:
            from diffusers import StableDiffusionPipeline
            
            # Try to load the model
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            console.print(f"[green]✅ Model {model_name} is valid and loadable[/]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Model {model_name} verification failed: {e}[/]")
            return False