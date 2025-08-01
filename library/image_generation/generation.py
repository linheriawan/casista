"""
Image generation functionality.

Handles image generation using various diffusion models.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import os

console = Console()

# Check for required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    console.print("[yellow]⚠️ PyTorch not installed. Image generation disabled.[/]")

try:
    from huggingface_hub import scan_cache_dir
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    console.print("[yellow]⚠️ HuggingFace Hub not installed. Cache scanning disabled.[/]")

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    console.print("[yellow]⚠️ Diffusers not installed. Image generation disabled.[/]")


class ImageGenerator:
    """Handles image generation using diffusion models."""
    
    def __init__(self, default_models: List[str] = None, output_dir: Path = None):
        """Initialize image generator."""
        self.dependencies_available = TORCH_AVAILABLE and DIFFUSERS_AVAILABLE
        
        if not self.dependencies_available:
            console.print("[red]❌ Image generation dependencies not available[/]")
            console.print("[dim]Install with: pip install torch diffusers transformers[/]")
            self.default_models = []
            self.cached_models = []
            self.loaded_models = {}
            self.device = "cpu"
            self.torch_dtype = None
            return
        
        self.default_models = default_models or [
            "hakurei/waifu-diffusion",
            "stabilityai/stable-diffusion-2-1",
            "CompVis/stable-diffusion-v1-4"
        ]
        self.output_dir = output_dir or Path.cwd() / "generated_images"
        self.output_dir.mkdir(exist_ok=True)
        
        # Model cache
        self.loaded_models = {}
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Scan HuggingFace cache for available models
        self.cached_models = self._scan_hf_cache()
        
        console.print(f"[cyan]🖼️ Image Generator initialized on {self.device}[/]")
        console.print(f"[dim]Found {len(self.cached_models)} models in HuggingFace cache[/]")
    
    def generate_image(self, prompt: str, model_name: str = None, 
                      negative_prompt: str = None, width: int = 512, 
                      height: int = 512, num_inference_steps: int = 20,
                      guidance_scale: float = 7.5, seed: int = None,
                      save_path: str = None, base_image: str = None) -> Optional[Path]:
        """Generate an image from a text prompt."""
        
        if not self.dependencies_available:
            console.print("[red]❌ Image generation dependencies not available[/]")
            return None
        
        # Select model
        if model_name is None:
            model_name = self.default_models[0]
        
        try:
            # Load model
            pipe = self._load_model(model_name)
            if pipe is None:
                return None
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            console.print(f"[cyan]🎨 Generating image: '{prompt[:50]}...'[/]")
            
            # Generate image with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Generating image...", total=num_inference_steps)
                
                # Modern callback approach (no deprecation warning)
                def progress_callback(pipe, step_index, timestep, callback_kwargs):
                    progress.update(task, completed=step_index + 1)
                    return callback_kwargs
                
                # Generate image (suppress deprecation warnings for callback)
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*callback.*")
                    
                    # Try modern callback first, fallback to legacy
                    try:
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            callback_on_step_end=progress_callback
                        )
                    except (TypeError, ValueError):
                        # Fallback to legacy callback
                        def legacy_callback(step: int, timestep: int, latents):
                            progress.update(task, completed=step)
                            
                        result = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            callback=legacy_callback,
                            callback_steps=1
                        )
                
                progress.update(task, completed=num_inference_steps)
            
            # Get the generated image
            image = result.images[0]
            
            # Save image
            if save_path is None:
                # Generate filename
                safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_prompt = safe_prompt.replace(' ', '_')
                timestamp = self._get_timestamp()
                save_path = self.output_dir / f"{safe_prompt}_{timestamp}.png"
            else:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
            
            image.save(save_path)
            
            console.print(f"[green]✅ Image saved to: {save_path}[/]")
            return save_path
            
        except Exception as e:
            console.print(f"[red]❌ Error generating image: {e}[/]")
            return None
    
    def generate_batch(self, prompts: List[str], model_name: str = None,
                      **kwargs) -> List[Optional[Path]]:
        """Generate multiple images from a list of prompts."""
        results = []
        
        console.print(f"[cyan]🎨 Generating {len(prompts)} images...[/]")
        
        for i, prompt in enumerate(prompts):
            console.print(f"[dim]Progress: {i+1}/{len(prompts)}[/]")
            result = self.generate_image(prompt, model_name, **kwargs)
            results.append(result)
        
        successful = len([r for r in results if r is not None])
        console.print(f"[green]✅ Generated {successful}/{len(prompts)} images successfully[/]")
        
        return results
    
    def _load_model(self, model_name: str):
        """Load a diffusion model."""
        if not self.dependencies_available:
            console.print("[red]❌ Cannot load model: dependencies not available[/]")
            return None
            
        if model_name in self.loaded_models:
            console.print(f"[green]♻️ Using cached model: {model_name}[/]")
            return self.loaded_models[model_name]
        
        try:
            from diffusers import StableDiffusionPipeline
            
            # Check if model is in HuggingFace cache
            cache_status = "from cache" if model_name in self.cached_models else "downloading"
            console.print(f"[cyan]📥 Loading model: {model_name} ({cache_status})[/]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}")
            ) as progress:
                task = progress.add_task(f"Loading {model_name}...", total=None)
                
                # Load with optimized settings
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True,
                    variant="fp16" if self.device == "cuda" else None
                )
                
                pipe = pipe.to(self.device)
                
                # Enable memory optimizations
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                if hasattr(pipe, 'enable_model_cpu_offload') and self.device == "cuda":
                    # Only enable CPU offload if we have limited GPU memory
                    try:
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        if gpu_memory < 8:  # Less than 8GB VRAM
                            pipe.enable_model_cpu_offload()
                    except:
                        pass
                
                progress.update(task, description="Model loaded successfully!")
            
            # Cache the model
            self.loaded_models[model_name] = pipe
            
            console.print(f"[green]✅ Model {model_name} loaded successfully[/]")
            return pipe
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load model {model_name}: {e}[/]")
            
            # Try fallback models if available
            available_models = self.list_available_models()
            fallback_models = [m for m in available_models if m != model_name and m in self.cached_models]
            
            if fallback_models:
                console.print(f"[yellow]⚠️ Trying fallback model: {fallback_models[0]}[/]")
                try:
                    return self._load_model(fallback_models[0])
                except:
                    pass
            
            return None
    
    def _scan_hf_cache(self) -> List[str]:
        """Scan HuggingFace cache for available models."""
        if not HF_HUB_AVAILABLE:
            return []
            
        cached_models = []
        try:
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                # Filter for diffusion models
                repo_id = repo.repo_id
                if any(keyword in repo_id.lower() for keyword in ['stable-diffusion', 'diffusion', 'sd-', 'sdxl']):
                    cached_models.append(repo_id)
        except Exception as e:
            console.print(f"[dim]⚠️ Could not scan HF cache: {e}[/]")
        
        return cached_models
    
    def list_available_models(self) -> List[str]:
        """List all available models (cached + default)."""
        all_models = list(set(self.default_models + self.cached_models))
        return sorted(all_models)
    
    def validate_model(self, model_name: str) -> bool:
        """Check if a model is available (cached or downloadable)."""
        available_models = self.list_available_models()
        return model_name in available_models or self._is_valid_hf_model(model_name)
    
    def _is_valid_hf_model(self, model_name: str) -> bool:
        """Check if model exists on HuggingFace hub."""
        try:
            from huggingface_hub import model_info
            info = model_info(model_name)
            return True
        except:
            return False
    
    def unload_model(self, model_name: str = None):
        """Unload a model from memory."""
        if model_name is None:
            # Unload all models
            self.loaded_models.clear()
            console.print("[green]✅ All models unloaded[/]")
        elif model_name in self.loaded_models:
            del self.loaded_models[model_name]
            console.print(f"[green]✅ Model {model_name} unloaded[/]")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def list_loaded_models(self) -> List[str]:
        """List currently loaded models."""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        info = {
            "name": model_name,
            "loaded": model_name in self.loaded_models,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype)
        }
        
        if info["loaded"]:
            pipe = self.loaded_models[model_name]
            info["memory_usage"] = self._estimate_memory_usage(pipe)
        
        return info
    
    def _estimate_memory_usage(self, pipe) -> str:
        """Estimate memory usage of a loaded model."""
        try:
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                return f"{memory_mb:.1f} MB"
            else:
                return "Unknown (CPU)"
        except:
            return "Unknown"
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate and analyze a prompt."""
        validation = {
            "valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        # Length check
        if len(prompt) < 3:
            validation["valid"] = False
            validation["warnings"].append("Prompt too short")
        
        if len(prompt) > 500:
            validation["warnings"].append("Very long prompt may be truncated")
        
        # Content analysis
        if not any(c.isalpha() for c in prompt):
            validation["valid"] = False
            validation["warnings"].append("Prompt should contain descriptive text")
        
        # Suggest improvements
        common_quality_terms = ["high quality", "detailed", "8k", "masterpiece"]
        if not any(term in prompt.lower() for term in common_quality_terms):
            validation["suggestions"].append("Consider adding quality descriptors like 'high quality' or 'detailed'")
        
        if "," not in prompt:
            validation["suggestions"].append("Consider using commas to separate different aspects of your description")
        
        return validation
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for filename."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_image_grid(self, image_paths: List[Path], output_path: Path = None,
                         grid_size: tuple = None) -> Optional[Path]:
        """Create a grid from multiple generated images."""
        try:
            from PIL import Image
            import math
            
            if not image_paths:
                console.print("[yellow]⚠️ No images to create grid[/]")
                return None
            
            # Load images
            images = []
            for path in image_paths:
                if path and path.exists():
                    images.append(Image.open(path))
            
            if not images:
                console.print("[yellow]⚠️ No valid images found[/]")
                return None
            
            # Determine grid size
            if grid_size is None:
                num_images = len(images)
                cols = math.ceil(math.sqrt(num_images))
                rows = math.ceil(num_images / cols)
                grid_size = (cols, rows)
            
            cols, rows = grid_size
            
            # Assume all images are the same size
            img_width, img_height = images[0].size
            
            # Create grid
            grid_width = cols * img_width
            grid_height = rows * img_height
            grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
            
            # Place images
            for i, img in enumerate(images):
                if i >= cols * rows:
                    break
                
                col = i % cols
                row = i // cols
                x = col * img_width
                y = row * img_height
                
                grid_image.paste(img, (x, y))
            
            # Save grid
            if output_path is None:
                timestamp = self._get_timestamp()
                output_path = self.output_dir / f"image_grid_{timestamp}.png"
            
            grid_image.save(output_path)
            console.print(f"[green]✅ Image grid saved to: {output_path}[/]")
            
            return output_path
            
        except Exception as e:
            console.print(f"[red]❌ Error creating image grid: {e}[/]")
            return None
    
    def upscale_image(self, image_path: Path, scale_factor: int = 2) -> Optional[Path]:
        """Upscale an image using simple interpolation."""
        try:
            from PIL import Image
            
            image = Image.open(image_path)
            original_size = image.size
            new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
            
            upscaled = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save upscaled image
            stem = image_path.stem
            suffix = image_path.suffix
            output_path = image_path.parent / f"{stem}_upscaled_{scale_factor}x{suffix}"
            
            upscaled.save(output_path)
            console.print(f"[green]✅ Upscaled image saved to: {output_path}[/]")
            
            return output_path
            
        except Exception as e:
            console.print(f"[red]❌ Error upscaling image: {e}[/]")
            return None
    
    def generate_image_to_image(self, prompt: str, base_image_path: Path, 
                               model_name: str = None, strength: float = 0.8,
                               negative_prompt: str = None, guidance_scale: float = 7.5,
                               num_inference_steps: int = 20, seed: int = None) -> Optional[Path]:
        """Generate image from existing image (img2img)."""
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            from PIL import Image
            
            # Load base image
            if not base_image_path.exists():
                console.print(f"[red]❌ Base image not found: {base_image_path}[/]")
                return None
            
            base_image = Image.open(base_image_path).convert("RGB")
            
            # Select model
            if model_name is None:
                model_name = self.default_models[0]
            
            # Load img2img pipeline
            img2img_key = f"{model_name}_img2img"
            if img2img_key not in self.loaded_models:
                console.print(f"[cyan]📥 Loading img2img pipeline: {model_name}[/]")
                
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_name,
                    torch_dtype=self.torch_dtype,
                    use_safetensors=True
                )
                
                pipe = pipe.to(self.device)
                
                if hasattr(pipe, 'enable_attention_slicing'):
                    pipe.enable_attention_slicing()
                
                self.loaded_models[img2img_key] = pipe
            else:
                pipe = self.loaded_models[img2img_key]
            
            # Set seed
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            console.print(f"[cyan]🔄 Refining image: '{prompt[:50]}...'[/]")
            
            # Generate refined image
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task("Refining image...", total=num_inference_steps)
                
                def progress_callback(pipe, step_index, timestep, callback_kwargs):
                    progress.update(task, completed=step_index + 1)
                    return callback_kwargs
                
                # Generate image (suppress deprecation warnings for callback)
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*callback.*")
                    
                    # Try modern callback first, fallback to legacy
                    try:
                        result = pipe(
                            prompt=prompt,
                            image=base_image,
                            strength=strength,
                            negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            callback_on_step_end=progress_callback
                        )
                    except (TypeError, ValueError):
                        # Fallback to legacy callback
                        def legacy_callback(step: int, timestep: int, latents):
                            progress.update(task, completed=step)
                            
                        result = pipe(
                            prompt=prompt,
                            image=base_image,
                            strength=strength,
                            negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            callback=legacy_callback,
                            callback_steps=1
                        )
                
                progress.update(task, completed=num_inference_steps)
            
            # Save refined image
            refined_image = result.images[0]
            
            # Generate filename
            base_stem = base_image_path.stem
            timestamp = self._get_timestamp()
            save_path = self.output_dir / f"{base_stem}_refined_{timestamp}.png"
            
            refined_image.save(save_path)
            
            console.print(f"[green]✅ Refined image saved to: {save_path}[/]")
            return save_path
            
        except Exception as e:
            console.print(f"[red]❌ Error refining image: {e}[/]")
            return None
    
    def create_variations(self, base_image_path: Path, num_variations: int = 4,
                         variation_strength: float = 0.6) -> List[Optional[Path]]:
        """Create variations of an existing image."""
        variations = []
        
        console.print(f"[cyan]🎨 Creating {num_variations} variations...[/]")
        
        for i in range(num_variations):
            console.print(f"[dim]Variation {i+1}/{num_variations}[/]")
            
            # Use different seeds for variations
            seed = torch.randint(0, 1000000, (1,)).item()
            
            variation = self.generate_image_to_image(
                prompt="create artistic variation, same style and composition",
                base_image_path=base_image_path,
                strength=variation_strength,
                seed=seed
            )
            
            variations.append(variation)
        
        successful = len([v for v in variations if v is not None])
        console.print(f"[green]✅ Created {successful}/{num_variations} variations[/]")
        
        return variations
    
    def run_pipeline(self, base_prompt: str, pipeline_steps: List[Dict[str, Any]]) -> Optional[Path]:
        """Run a pipeline of image generation steps."""
        console.print(f"[cyan]🔗 Running image pipeline: {len(pipeline_steps)} steps[/]")
        
        current_image = None
        current_prompt = base_prompt
        
        for i, step in enumerate(pipeline_steps):
            step_type = step.get('type', 'generate')
            step_prompt = step.get('prompt', current_prompt)
            step_model = step.get('model', self.default_models[0])
            
            console.print(f"[cyan]Step {i+1}/{len(pipeline_steps)}: {step_type} - {step_prompt[:50]}...[/]")
            
            if step_type == 'generate':
                # Generate new image
                current_image = self.generate_image(
                    prompt=step_prompt,
                    model_name=step_model,
                    **{k: v for k, v in step.items() if k not in ['type', 'prompt', 'model']}
                )
            elif step_type == 'refine' and current_image:
                # Refine existing image
                current_image = self.generate_image_to_image(
                    prompt=step_prompt,
                    base_image_path=current_image,
                    model_name=step_model,
                    **{k: v for k, v in step.items() if k not in ['type', 'prompt', 'model']}
                )
            elif step_type == 'upscale' and current_image:
                # Upscale image
                scale_factor = step.get('scale_factor', 2)
                current_image = self.upscale_image(current_image, scale_factor)
            else:
                console.print(f"[yellow]⚠️ Skipping step {i+1}: {step_type} (unsupported or no base image)[/]")
                continue
            
            if current_image is None:
                console.print(f"[red]❌ Pipeline failed at step {i+1}[/]")
                return None
            
            console.print(f"[green]✅ Step {i+1} completed: {current_image}[/]")
            current_prompt = step_prompt  # Update prompt for next step
        
        console.print(f"[green]🎉 Pipeline completed! Final result: {current_image}[/]")
        return current_image