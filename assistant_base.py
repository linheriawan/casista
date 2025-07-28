#!/usr/bin/env python3
"""
Base Assistant Framework for Multi-Modal AI Interactions
Supports: chat, speech, image processing with multiple models
"""

import os
import json
import abc
import getpass
from pathlib import Path
from typing import List, Dict, Optional, Any
import ollama
from rich.console import Console
from rich.panel import Panel

console = Console()

class BaseAssistant(abc.ABC):
    """Base class for all AI assistants"""
    
    def __init__(self, model: str, name: str, working_dir: Path = None):
        self.model = model
        self.name = name
        self.working_dir = working_dir or Path.cwd()
        self.context_dir = self.working_dir / ".ai_context" / name
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.context_file = self.context_dir / "context.json"
        self.config_file = self.context_dir / "config.json"
        
        # Load or create config
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load assistant configuration"""
        # Detect system username
        try:
            system_user = getpass.getuser()
        except:
            system_user = "user"
        
        default_config = {
            "model": self.model,
            "name": self.name,
            "user_name": system_user,  # Auto-detected user
            "temperature": 0.1,
            "max_tokens": 2048,
            "system_prompt": f"You are {self.name}, a helpful AI assistant talking to {system_user}.",
            "capabilities": []
        }
        
        if self.config_file.exists():
            try:
                config = json.loads(self.config_file.read_text())
                # Merge with defaults, but preserve user_name if set
                merged = {**default_config, **config}
                # Update system prompt with current user name
                merged["system_prompt"] = f"You are {self.name}, a helpful AI assistant talking to {merged['user_name']}."
                return merged
            except:
                pass
        
        self._save_config(default_config)
        return default_config
    
    def _save_config(self, config: Dict):
        """Save assistant configuration"""
        self.config_file.write_text(json.dumps(config, indent=2))
        self.config = config
    
    def _load_context(self) -> List[Dict]:
        """Load conversation context"""
        if self.context_file.exists():
            try:
                return json.loads(self.context_file.read_text()).get("messages", [])
            except:
                pass
        return []
    
    def _save_context(self, messages: List[Dict]):
        """Save conversation context"""
        self.context_file.write_text(json.dumps({"messages": messages}, indent=2))
    
    def reset_context(self):
        """Reset conversation context"""
        self._save_context([])
        console.print(f"[green]✅ Context reset for {self.name}![/]")
    
    def set_user(self, user_name: str):
        """Set the user name for this assistant"""
        self.config["user_name"] = user_name
        self.config["system_prompt"] = f"You are {self.name}, a helpful AI assistant talking to {user_name}."
        self._save_config(self.config)
        console.print(f"[green]✅ I'll now call you {user_name}![/]")
    
    def get_user_name(self) -> str:
        """Get the current user name"""
        return self.config.get("user_name", "user")
    
    def handle_special_commands(self, user_input: str) -> bool:
        """Handle special commands like 'set user', 'who am i', returns True if handled"""
        input_lower = user_input.lower().strip()
        
        if input_lower == "who am i":
            user_name = self.get_user_name()
            console.print(f"[cyan]You are {user_name}[/]")
            return True
        
        if input_lower.startswith("set user "):
            new_name = user_input[9:].strip()  # Remove "set user "
            if new_name:
                self.set_user(new_name)
            else:
                console.print("[yellow]Usage: set user [your_name][/]")
            return True
        
        return False
    
    @abc.abstractmethod
    def process_input(self, input_data: Any, **kwargs) -> str:
        """Process input and return response"""
        pass
    
    @abc.abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for this assistant type"""
        pass


class ChatAssistant(BaseAssistant):
    """Text-based chat assistant"""
    
    def __init__(self, model: str, name: str, working_dir: Path = None, file_ops: bool = True):
        super().__init__(model, name, working_dir)
        self.file_ops = file_ops
        self.config["capabilities"] = ["text_chat"]
        if file_ops:
            self.config["capabilities"].append("file_operations")
    
    def get_system_prompt(self) -> str:
        user_name = self.get_user_name()
        base_prompt = f"You are {self.name}, a helpful AI assistant talking to {user_name}. Working directory: {self.working_dir}"
        
        if self.file_ops:
            base_prompt += """

File operations format:
- Create file: ```create:filename.ext
actual content here
```
- Update file: ```update:filename.ext
full new content
```
- Read file: ```read:filename.ext
```
- Delete file: ```delete:filename.ext
```
- Make directory: ```mkdir:dirname
```

Always provide real, complete file content. Be concise."""
        
        return base_prompt
    
    def process_input(self, text: str, auto_confirm: bool = False, **kwargs) -> str:
        """Process text input and return response"""
        # Handle special commands first
        if self.handle_special_commands(text):
            return ""
        
        messages = self._load_context()
        
        # Handle read blocks in user input
        processed_text = self._handle_read_blocks(text)
        
        # Prepare messages
        system_prompt = self.get_system_prompt()
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": processed_text}]
        
        try:
            console.print(f"\n[bold cyan]🤖 {self.name}:[/]")
            
            # Stream response from Ollama
            stream = ollama.chat(
                model=self.model,
                messages=messages_with_system,
                stream=True,
                options={
                    'temperature': self.config.get('temperature', 0.1),
                    'num_predict': self.config.get('max_tokens', 2048)
                }
            )
            
            full_response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    full_response += content
                    
                if len(full_response) > 15000:  # Safety limit
                    print("\n⚠️ Response truncated (too long)")
                    break
            
            print("\n")
            
            # Handle file operations if enabled
            if self.file_ops and full_response:
                self._handle_file_operations(full_response, auto_confirm)
            
            # Save context
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": full_response})
            self._save_context(messages)
            
            return full_response
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")
            return ""
    
    def _handle_read_blocks(self, text: str) -> str:
        """Replace ```read:file``` blocks with actual file content."""
        import re
        lines = text.splitlines(keepends=True)
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            match = re.match(r"^```read:([^\n]+)", line.strip())
            if match:
                file_path = self.working_dir / match.group(1).strip()
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    i += 1
                i += 1  # skip closing ```
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8')
                    result.append(f"```text:read:{file_path.relative_to(self.working_dir)}\n")
                    result.append(content + "\n")
                    result.append("```\n\n")
                else:
                    result.append(f"[File not found: {file_path.relative_to(self.working_dir)}]\n")
            else:
                result.append(line)
                i += 1
        return "".join(result)
    
    def _handle_file_operations(self, response: str, auto_confirm: bool = False):
        """Parse and execute file operations from response"""
        from coder import parse_code_blocks, apply_code_changes
        
        # Change to working directory for file operations
        original_cwd = os.getcwd()
        try:
            os.chdir(self.working_dir)
            code_blocks = parse_code_blocks(response)
            if code_blocks:
                console.print(f"\n[yellow]📋 Found {len(code_blocks)} file operation(s)[/]\n")
                apply_code_changes(code_blocks, auto_confirm)
        finally:
            os.chdir(original_cwd)


class SpeechAssistant(BaseAssistant):
    """Speech-based assistant with voice recognition and text-to-speech"""
    
    def __init__(self, model: str, name: str, working_dir: Path = None):
        super().__init__(model, name, working_dir)
        self.config["capabilities"] = ["text_to_speech", "speech_to_text"]
        
        # Initialize speech components
        self._init_speech_components()
    
    def _init_speech_components(self):
        """Initialize speech recognition and TTS components"""
        try:
            import speech_recognition as sr
            import pyttsx3
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Initialize text-to-speech
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS voice from config or use default
            voices = self.tts_engine.getProperty('voices')
            preferred_voice = self.config.get('tts_voice', None)
            
            if preferred_voice and voices:
                # Try to find the preferred voice
                for voice in voices:
                    if voice.id == preferred_voice or voice.name == preferred_voice:
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            elif voices:
                # Default: prefer female voice or first available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'samantha' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate from config or default
            speech_rate = self.config.get('speech_rate', 200)
            self.tts_engine.setProperty('rate', speech_rate)
            
            console.print("[green]✅ Speech components initialized[/]")
            
        except ImportError as e:
            console.print(f"[red]❌ Speech libraries not installed: {e}[/]")
            console.print("[yellow]💡 Install with: pip install speechrecognition pyttsx3 pyaudio[/]")
            self.recognizer = None
            self.tts_engine = None
    
    def get_system_prompt(self) -> str:
        user_name = self.get_user_name()
        return f"You are {self.name}, a voice assistant talking to {user_name}. Provide clear, concise spoken responses. Avoid long explanations since this is voice conversation."
    
    def speak(self, text: str):
        """Convert text to speech"""
        if not self.tts_engine:
            console.print(f"[dim]🤖 {self.name} would say: {text}[/]")
            return
        
        try:
            console.print(f"[cyan]🔊 {self.name} speaking...[/]")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            console.print(f"[red]❌ TTS Error: {e}[/]")
            console.print(f"[dim]🤖 {self.name}: {text}[/]")
    
    def listen(self) -> Optional[str]:
        """Listen for speech and convert to text"""
        if not self.recognizer:
            print("\r⚠️ Speech recognition not available", end="", flush=True)
            return None
        
        try:
            print("\r🎤 Listening... (speak now)", end="", flush=True)
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for audio
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
            print("\r🔄 Processing speech...    ", end="", flush=True)
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            print(f"\r✅ You said: {text}                    ")  # Clear line with spaces
            return text
            
        except sr.WaitTimeoutError:
            print("\r⏰ Listening timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("\r❓ Could not understand audio          ")
            return None
        except sr.RequestError as e:
            print(f"\r❌ Speech recognition error: {e}      ")
            return None
        except Exception as e:
            print(f"\r❌ Unexpected error: {e}              ")
            return None
    
    def handle_voice_commands(self, text: str) -> bool:
        """Handle voice-specific commands, returns True if handled"""
        text_lower = text.lower().strip()
        
        if text_lower == "change voice":
            current_voice = self.config.get('tts_voice_name', 'Unknown')
            response = f"I'm currently using {current_voice}. To change my voice, use the voice selector tool on your computer."
            self.speak(response)
            return True
        
        if text_lower == "what voice am i using" or text_lower == "what voice are you using":
            current_voice = self.config.get('tts_voice_name', 'the default system voice')
            response = f"I'm using {current_voice}."
            self.speak(response)
            return True
        
        if text_lower.startswith("speak slower"):
            new_rate = max(100, self.config.get('speech_rate', 200) - 50)
            self.config['speech_rate'] = new_rate
            self._save_config(self.config)
            if self.tts_engine:
                self.tts_engine.setProperty('rate', new_rate)
            self.speak("I'll speak slower now.")
            return True
        
        if text_lower.startswith("speak faster"):
            new_rate = min(300, self.config.get('speech_rate', 200) + 50)
            self.config['speech_rate'] = new_rate
            self._save_config(self.config)
            if self.tts_engine:
                self.tts_engine.setProperty('rate', new_rate)
            self.speak("I'll speak faster now.")
            return True
        
        return False
    
    def process_input(self, input_data: Any = None, auto_confirm: bool = False, **kwargs) -> str:
        """Process voice input and provide voice response"""
        # If input_data is provided (text), use it directly
        if input_data and isinstance(input_data, str):
            text = input_data
        else:
            # Listen for voice input
            text = self.listen()
            if not text:
                return ""
        
        # Handle voice-specific commands first
        if self.handle_voice_commands(text):
            return ""
        
        # Handle general special commands
        if self.handle_special_commands(text):
            return ""
        
        messages = self._load_context()
        
        # Prepare messages for AI
        system_prompt = self.get_system_prompt()
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": text}]
        
        try:
            console.print(f"\n[bold cyan]🤖 {self.name} thinking...[/]")
            
            # Get response from Ollama (non-streaming for voice)
            response = ollama.chat(
                model=self.model,
                messages=messages_with_system,
                stream=False,
                options={
                    'temperature': self.config.get('temperature', 0.1),
                    'num_predict': 300  # Shorter responses for voice
                }
            )
            
            full_response = response['message']['content']
            
            # Speak the response
            self.speak(full_response)
            
            # Save context
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": full_response})
            self._save_context(messages)
            
            return full_response
            
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            console.print(f"[red]❌ Error: {e}[/]")
            self.speak(error_msg)
            return ""


class ImageAssistant(BaseAssistant):
    """Image generation and processing assistant"""
    
    def __init__(self, model: str, name: str, working_dir: Path = None):
        super().__init__(model, name, working_dir)
        self.config["capabilities"] = ["image_analysis", "image_generation"]
        self._init_image_components()
    
    def _init_image_components(self):
        """Initialize image generation components"""
        try:
            # Try different image generation backends
            self.image_backend = self._detect_image_backend()
            if self.image_backend:
                console.print(f"[green]✅ Image backend initialized: {self.image_backend}[/]")
            else:
                console.print("[yellow]⚠️ No image generation backend available[/]")
                console.print("[dim]Install: pip install diffusers torch transformers[/]")
        except Exception as e:
            console.print(f"[red]❌ Image initialization error: {e}[/]")
            self.image_backend = None
    
    def _detect_image_backend(self):
        """Detect available image generation backend"""
        # Try Ollama LLaVA first
        try:
            import ollama
            models = ollama.list()
            for model in models.get('models', []):
                model_name = getattr(model, 'model', '')
                if 'llava' in model_name.lower():
                    return f"llava:{model_name}"
        except:
            pass
        
        # Try local Stable Diffusion
        try:
            from diffusers import StableDiffusionPipeline
            return "diffusers"
        except ImportError:
            pass
        
        # Try OpenAI-compatible API (if configured)
        try:
            import requests
            # Could check for local APIs like Automatic1111
            return None
        except:
            pass
        
        return None
    
    def get_system_prompt(self) -> str:
        user_name = self.get_user_name()
        backend_info = ""
        
        if self.image_backend and "dreamlike" in self.image_backend.lower():
            backend_info = """
IMPORTANT: You're using DreamLike Anime model optimized for:
- Anime/artistic style images
- Maximum 77 tokens in descriptions (keep prompts SHORT)
- Works best with: anime characters, artistic scenes, vibrant colors
- Default size: 512x512 (specify if user wants different: 768x768, 1024x1024)"""
        elif self.image_backend and "tiny" in self.image_backend.lower():
            backend_info = """
IMPORTANT: You're using TinySD model optimized for:
- Fast generation with good quality
- Keep descriptions concise (under 50 tokens)
- Works well with: realistic scenes, landscapes, portraits"""
        else:
            backend_info = """
IMPORTANT: Keep image descriptions concise and focused for best results."""
        
        return f"""You are {self.name}, an image generation assistant talking to {user_name}.{backend_info}

When {user_name} asks you to create, generate, or make an image, respond with:
```generate:descriptive_filename.png
SHORT, focused description optimized for the current model
```

Extract size requests from prompts (512x512, 768x768, 1024x1024) and use appropriate dimensions.
Use descriptive filenames that match the content.
For anime characters/scenes, emphasize: character names, actions, art style, colors."""
    
    def generate_image(self, prompt: str, filename: str, width: int = 512, height: int = 512) -> bool:
        """Generate image from text prompt with custom dimensions"""
        if not self.image_backend:
            console.print("[yellow]⚠️ No image generation backend available[/]")
            return False
        
        try:
            console.print(f"[cyan]🎨 Generating image: {filename}[/]")
            console.print(f"[dim]Prompt: {prompt}[/]")
            
            if self.image_backend.startswith("llava"):
                return self._generate_with_llava(prompt, filename, width, height)
            elif self.image_backend == "diffusers":
                return self._generate_with_diffusers(prompt, filename, width, height)
            else:
                console.print("[red]❌ Unknown image backend[/]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Image generation error: {e}[/]")
            return False
    
    def _generate_with_llava(self, prompt: str, filename: str, width: int = 512, height: int = 512) -> bool:
        """Generate image using LLaVA model (if it supports generation)"""
        console.print("[yellow]ℹ️ LLaVA is primarily for image analysis, not generation[/]")
        console.print("[dim]Consider using diffusers backend for generation[/]")
        return False
    
    def _generate_with_diffusers(self, prompt: str, filename: str, width: int = 512, height: int = 512) -> bool:
        """Generate image using Stable Diffusion with lightweight models"""
        try:
            from diffusers import StableDiffusionPipeline, DiffusionPipeline
            import torch
            
            # Lightweight model options (in order of preference)
            lightweight_models = [
                ("segmind/tiny-sd", "TinySD (~800MB)"),
                ("dreamlike-art/dreamlike-anime-1.0", "DreamLike Anime (~2GB)"),
                ("prompthero/openjourney-v4", "OpenJourney v4 (~2GB)"),
                ("hakurei/waifu-diffusion", "Waifu Diffusion (~2GB)"),
                ("runwayml/stable-diffusion-v1-5", "SD v1.5 (~4GB)")
            ]
            
            console.print("[cyan]🔍 Trying lightweight models...[/]")
            
            pipe = None
            used_model = None
            
            for model_id, model_name in lightweight_models:
                try:
                    console.print(f"[dim]Attempting: {model_name}[/]")
                    
                    # Special handling for TinySD
                    if "tiny-sd" in model_id:
                        pipe = DiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            use_safetensors=True
                        )
                    else:
                        pipe = StableDiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            use_safetensors=True
                        )
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        pipe = pipe.to("cuda")
                    
                    used_model = model_name
                    console.print(f"[green]✅ Using: {model_name}[/]")
                    break
                    
                except Exception as e:
                    console.print(f"[dim]❌ {model_name} failed: {e}[/]")
                    continue
            
            if not pipe:
                console.print("[red]❌ No compatible models available[/]")
                return False
            
            # Optimize settings for lightweight models
            if "tiny" in used_model.lower():
                # TinySD optimized settings
                num_steps = 10
                guidance = 5.0
            elif "anime" in used_model.lower() or "waifu" in used_model.lower():
                # Anime models optimized settings
                num_steps = 15
                guidance = 6.0
            else:
                # Standard settings
                num_steps = 20
                guidance = 7.5
            
            console.print(f"[cyan]🎨 Generating with {used_model} ({num_steps} steps)...[/]")
            
            # Generate image
            image = pipe(
                prompt, 
                num_inference_steps=num_steps, 
                guidance_scale=guidance,
                height=height,
                width=width
            ).images[0]
            
            # Save to working directory
            image_path = self.working_dir / filename
            image.save(image_path)
            
            console.print(f"[green]✅ Image saved: {image_path}[/]")
            console.print(f"[dim]Model used: {used_model}[/]")
            return True
            
        except ImportError:
            console.print("[red]❌ Diffusers not installed: pip install diffusers torch transformers accelerate[/]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Generation failed: {e}[/]")
            return False
    
    def process_input(self, input_data: Any = None, auto_confirm: bool = False, **kwargs) -> str:
        """Process image generation or analysis requests"""
        if isinstance(input_data, str):
            text = input_data
        else:
            text = input("🎨 Describe the image you want to create: ").strip()
        
        if not text:
            return ""
        
        # Handle special commands
        if self.handle_special_commands(text):
            return ""
        
        messages = self._load_context()
        
        # Get AI response for image description/generation
        system_prompt = self.get_system_prompt()
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": text}]
        
        try:
            console.print(f"\n[bold cyan]🤖 {self.name}:[/]")
            
            # Get response from Ollama
            response = ollama.chat(
                model=self.model,
                messages=messages_with_system,
                stream=False,
                options={
                    'temperature': self.config.get('temperature', 0.7),  # Higher for creativity
                    'num_predict': 500
                }
            )
            
            full_response = response['message']['content']
            console.print(full_response)
            
            # Parse for image generation commands
            import re
            pattern = r"```generate:([^\n]+)\n(.*?)\n```"
            matches = re.findall(pattern, full_response, re.DOTALL)
            
            for filename, description in matches:
                filename = filename.strip()
                description = description.strip()
                
                # Parse size from description or filename
                size_match = re.search(r'(\d+)x(\d+)', description + " " + filename)
                width, height = 512, 512  # default
                
                if size_match:
                    width, height = int(size_match.group(1)), int(size_match.group(2))
                    # Remove size from description to avoid confusion
                    description = re.sub(r'\b\d+x\d+\b', '', description).strip()
                    console.print(f"[dim]📐 Size detected: {width}x{height}[/]")
                else:
                    console.print(f"[dim]📐 No size found, using default: {width}x{height}[/]")
                    console.print(f"[dim]Debug - description: '{description}'[/]")
                    console.print(f"[dim]Debug - filename: '{filename}'[/]")
                
                console.print(f"\n[yellow]🎨 Generating image: {filename} ({width}x{height})[/]")
                
                if auto_confirm or input(f"Generate image '{filename}'? (y/n): ").lower().startswith('y'):
                    self.generate_image(description, filename, width, height)
            
            # Save context
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": full_response})
            self._save_context(messages)
            
            return full_response
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/]")
            return ""


# Assistant factory
def create_assistant(model: str, name: str, mode: str, working_dir: Path = None) -> BaseAssistant:
    """Factory function to create appropriate assistant"""
    mode = mode.lower()
    
    if mode == "chat":
        return ChatAssistant(model, name, working_dir)
    elif mode == "speech":
        return SpeechAssistant(model, name, working_dir)
    elif mode == "image":
        return ImageAssistant(model, name, working_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported: chat, speech, image")