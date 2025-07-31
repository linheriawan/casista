"""
Voice management utilities.

Handles TTS voices, speech recognition models, and voice configuration.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, NamedTuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()


class VoiceInfo(NamedTuple):
    """Voice information structure."""
    id: str
    name: str
    language: str
    gender: str
    engine: str


class VoiceManager:
    """Manages TTS voices and speech recognition configuration."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize voice manager."""
        self.config_dir = config_dir or Path("configuration")
        self.voice_config_file = self.config_dir / "voice_settings.toml"
    
    def list_tts_voices(self) -> List[VoiceInfo]:
        """List all available TTS voices."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            if not voices:
                console.print("[red]❌ No TTS voices found[/]")
                return []
            
            voice_list = []
            for voice in voices:
                # Parse voice information
                voice_info = self._parse_voice_info(voice)
                voice_list.append(voice_info)
            
            engine.stop()
            return voice_list
            
        except ImportError:
            console.print("[red]❌ pyttsx3 not installed. Install with: pip install pyttsx3[/]")
            return []
        except Exception as e:
            console.print(f"[red]❌ Error listing TTS voices: {e}[/]")
            return []
    
    def display_tts_voices(self):
        """Display available TTS voices in a table."""
        voices = self.list_tts_voices()
        
        if not voices:
            console.print("[yellow]⚠️ No TTS voices found[/]")
            return True
        
        console.print(f"[green]Found {len(voices)} TTS voices:[/]\n")
        
        table = Table(title="Available TTS Voices")
        table.add_column("ID", style="dim", width=3)
        table.add_column("Name", style="cyan", min_width=20)
        table.add_column("Language", style="green", width=15)
        table.add_column("Gender", style="magenta", width=10)
        table.add_column("Engine", style="yellow", width=10)
        
        for i, voice in enumerate(voices):
            table.add_row(
                str(i),
                voice.name[:30] + "..." if len(voice.name) > 30 else voice.name,
                voice.language,
                voice.gender,
                voice.engine
            )
        
        console.print(table)
        return True
    
    def test_tts_voice(self, voice_id: str, test_text: str = "Hello, this is a voice test.") -> bool:
        """Test a TTS voice."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            # Find voice by ID or index
            target_voice = None
            
            # Try as index first
            try:
                voice_index = int(voice_id)
                if 0 <= voice_index < len(voices):
                    target_voice = voices[voice_index]
            except ValueError:
                # Try as voice ID
                for voice in voices:
                    if voice.id == voice_id:
                        target_voice = voice
                        break
            
            if not target_voice:
                console.print(f"[red]❌ Voice not found: {voice_id}[/]")
                return False
            
            console.print(f"[cyan]🔊 Testing voice: {target_voice.name}[/]")
            
            # Set voice and speak
            engine.setProperty('voice', target_voice.id)
            engine.say(test_text)
            engine.runAndWait()
            engine.stop()
            
            console.print("[green]✅ Voice test completed[/]")
            return True
            
        except ImportError:
            console.print("[red]❌ pyttsx3 not installed. Install with: pip install pyttsx3[/]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Error testing voice: {e}[/]")
            return False
    
    def set_assistant_voice(self, assistant_name: str, voice_id: str, voice_name: str = None) -> bool:
        """Set TTS voice for an assistant."""
        from library.assistant_cfg import AssistantConfig
        
        assistant_config = AssistantConfig(self.config_dir)
        
        # Get voice name if not provided
        if voice_name is None:
            voices = self.list_tts_voices()
            try:
                voice_index = int(voice_id)
                if 0 <= voice_index < len(voices):
                    voice_name = voices[voice_index].name
            except ValueError:
                voice_name = voice_id
        
        # Update assistant voice settings
        success = assistant_config.update_assistant_setting(
            assistant_name, "voice", "voice_id", voice_id
        )
        
        if success:
            assistant_config.update_assistant_setting(
                assistant_name, "voice", "voice_name", voice_name
            )
            console.print(f"[green]✅ Set voice for '{assistant_name}': {voice_name}[/]")
        
        return success
    
    def list_sr_models(self) -> List[Dict[str, Any]]:
        """List available speech recognition models/backends."""
        models = [
            {
                "id": "google",
                "name": "Google Speech Recognition",
                "type": "cloud",
                "status": self._check_sr_backend("google"),
                "description": "Cloud-based, requires internet",
                "accuracy": "High",
                "speed": "Fast"
            },
            {
                "id": "whisper",
                "name": "OpenAI Whisper",
                "type": "local",
                "status": self._check_sr_backend("whisper"),
                "description": "Local processing, works offline",
                "accuracy": "Very High",
                "speed": "Medium"
            },
            {
                "id": "vosk",
                "name": "Vosk",
                "type": "local",
                "status": self._check_sr_backend("vosk"),
                "description": "Lightweight, fast offline recognition",
                "accuracy": "Medium",
                "speed": "Very Fast"
            },
            {
                "id": "sphinx",
                "name": "CMU Sphinx",
                "type": "local",
                "status": self._check_sr_backend("sphinx"),
                "description": "Traditional speech recognition",
                "accuracy": "Medium",
                "speed": "Fast"
            }
        ]
        
        return models
    
    def display_sr_models(self):
        """Display available speech recognition models."""
        models = self.list_sr_models()
        
        console.print("[bold cyan]🎤 Speech Recognition Models[/]\n")
        
        table = Table(title="Available Speech Recognition Backends")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Status", style="magenta")
        table.add_column("Accuracy", style="blue")
        table.add_column("Speed", style="blue")
        table.add_column("Description", style="dim")
        
        for model in models:
            status = "✅ Available" if model["status"] else "❌ Not Installed"
            
            table.add_row(
                model["id"],
                model["name"],
                model["type"],
                status,
                model["accuracy"],
                model["speed"],
                model["description"]
            )
        
        console.print(table)
        
        # Show installation instructions
        console.print(f"\n[bold yellow]💡 Installation Instructions:[/]")
        console.print("• Whisper: pip install openai-whisper")
        console.print("• Vosk: pip install vosk")
        console.print("• Sphinx: pip install pocketsphinx")
        console.print("• Google: pip install speechrecognition (built-in)")
        
        return True
    
    def download_sr_model(self, model_id: str) -> bool:
        """Download/install a speech recognition model."""
        if model_id == "whisper":
            return self._install_whisper()
        elif model_id == "vosk":
            return self._install_vosk()
        elif model_id == "sphinx":
            return self._install_sphinx()
        elif model_id == "google":
            console.print("[green]✅ Google Speech Recognition is built-in with speechrecognition package[/]")
            return True
        else:
            console.print(f"[red]❌ Unknown model ID: {model_id}[/]")
            return False
    
    def remove_sr_model(self, model_id: str) -> bool:
        """Remove a speech recognition model."""
        console.print(f"[yellow]⚠️ Manual removal required for {model_id}[/]")
        
        if model_id == "whisper":
            console.print("Run: pip uninstall openai-whisper")
        elif model_id == "vosk":
            console.print("Run: pip uninstall vosk")
            console.print("Also remove model files from ~/.cache/vosk/")
        elif model_id == "sphinx":
            console.print("Run: pip uninstall pocketsphinx")
        elif model_id == "google":
            console.print("Google Speech Recognition cannot be removed (built-in)")
        
        return True
    
    def set_assistant_speech_backend(self, assistant_name: str, backend: str) -> bool:
        """Set speech recognition backend for an assistant."""
        from library.assistant_cfg import AssistantConfig
        
        valid_backends = ['google', 'whisper', 'vosk', 'sphinx']
        
        if backend not in valid_backends:
            console.print(f"[red]❌ Invalid backend. Choose from: {', '.join(valid_backends)}[/]")
            return False
        
        # Check if backend is available
        if not self._check_sr_backend(backend):
            console.print(f"[red]❌ Backend '{backend}' is not installed[/]")
            console.print(f"[dim]Install with: download-sr-model {backend}[/]")
            return False
        
        assistant_config = AssistantConfig(self.config_dir)
        
        # Update assistant speech settings
        success = assistant_config.update_assistant_setting(
            assistant_name, "voice", "speech_backend", backend
        )
        
        if success:
            console.print(f"[green]✅ Set speech backend for '{assistant_name}': {backend}[/]")
        
        return success
    
    def configure_voice_interactive(self, assistant_name: str) -> bool:
        """Configure voice settings for an assistant interactively."""
        console.print(Panel.fit(
            f"[bold cyan]🎤 Configure Voice for: {assistant_name}[/]",
            title="Voice Configuration"
        ))
        
        while True:
            console.print("\n[bold]Voice Configuration Options:[/]")
            console.print("1. Set TTS voice")
            console.print("2. Set speech recognition backend")
            console.print("3. Test current TTS voice")
            console.print("4. View current voice settings")
            console.print("5. Exit voice configuration")
            
            choice = Prompt.ask(
                "[cyan]Select option[/]",
                choices=["1", "2", "3", "4", "5"],
                default="5"
            )
            
            if choice == "1":
                voices = self.display_tts_voices()
                if voices:
                    voice_id = Prompt.ask("[cyan]Voice ID (number)[/]")
                    try:
                        voice_index = int(voice_id)
                        if 0 <= voice_index < len(voices):
                            self.set_assistant_voice(assistant_name, voices[voice_index].id, voices[voice_index].name)
                        else:
                            console.print(f"[red]❌ Invalid voice ID. Use 0-{len(voices)-1}[/]")
                    except ValueError:
                        console.print("[red]❌ Voice ID must be a number[/]")
            
            elif choice == "2":
                self.display_sr_models()
                backend = Prompt.ask(
                    "[cyan]Speech backend[/]",
                    choices=["google", "whisper", "vosk", "sphinx"],
                    default="google"
                )
                self.set_assistant_speech_backend(assistant_name, backend)
            
            elif choice == "3":
                from library.assistant_cfg import AssistantConfig
                assistant_config = AssistantConfig(self.config_dir)
                voice_id = assistant_config.get_assistant_setting(assistant_name, "voice", "voice_id")
                
                if voice_id:
                    test_text = Prompt.ask(
                        "[cyan]Test text[/]",
                        default="Hello, this is a voice test for the assistant."
                    )
                    self.test_tts_voice(voice_id, test_text)
                else:
                    console.print("[yellow]⚠️ No voice configured for this assistant[/]")
            
            elif choice == "4":
                self._show_assistant_voice_settings(assistant_name)
            
            elif choice == "5":
                break
        
        console.print(f"[green]✅ Voice configuration complete for '{assistant_name}'[/]")
        return True
    
    def _parse_voice_info(self, voice) -> VoiceInfo:
        """Parse voice information from pyttsx3 voice object."""
        name = voice.name if hasattr(voice, 'name') else "Unknown"
        voice_id = voice.id if hasattr(voice, 'id') else "unknown"
        
        # Detect language
        language = "Unknown"
        if any(lang in voice_id.lower() for lang in ['en_us', 'en-us', 'english']):
            language = "English (US)"
        elif any(lang in voice_id.lower() for lang in ['en_gb', 'en-gb']):
            language = "English (UK)"
        elif any(lang in voice_id.lower() for lang in ['es', 'spanish']):
            language = "Spanish"
        elif any(lang in voice_id.lower() for lang in ['fr', 'french']):
            language = "French"
        elif any(lang in voice_id.lower() for lang in ['de', 'german']):
            language = "German"
        elif any(lang in voice_id.lower() for lang in ['it', 'italian']):
            language = "Italian"
        elif any(lang in voice_id.lower() for lang in ['ja', 'japanese']):
            language = "Japanese"
        elif any(lang in voice_id.lower() for lang in ['zh', 'chinese']):
            language = "Chinese"
        
        # Detect gender
        gender = "Unknown"
        name_lower = name.lower()
        if any(term in name_lower for term in ['male', 'man', 'boy', 'mr']):
            gender = "Male"
        elif any(term in name_lower for term in ['female', 'woman', 'girl', 'ms', 'mrs']):
            gender = "Female"
        elif any(name in name_lower for name in ['alex', 'tom', 'daniel', 'david', 'james']):
            gender = "Male"
        elif any(name in name_lower for name in ['victoria', 'susan', 'karen', 'anna', 'sara']):
            gender = "Female"
        
        # Detect engine
        engine = "System"
        if 'sapi' in voice_id.lower():
            engine = "SAPI"
        elif 'espeak' in voice_id.lower():
            engine = "eSpeak"
        elif 'festival' in voice_id.lower():
            engine = "Festival"
        
        return VoiceInfo(voice_id, name, language, gender, engine)
    
    def _check_sr_backend(self, backend: str) -> bool:
        """Check if a speech recognition backend is available."""
        try:
            if backend == "google":
                import speech_recognition
                return True
            elif backend == "whisper":
                import whisper
                return True
            elif backend == "vosk":
                import vosk
                return True
            elif backend == "sphinx":
                import speech_recognition
                # Check if pocketsphinx is available
                sr = speech_recognition.Recognizer()
                return hasattr(sr, 'recognize_sphinx')
            return False
        except ImportError:
            return False
    
    def _install_whisper(self) -> bool:
        """Install OpenAI Whisper."""
        console.print("[cyan]📥 Installing OpenAI Whisper...[/]")
        
        try:
            import subprocess
            import sys
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "openai-whisper"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green]✅ OpenAI Whisper installed successfully[/]")
                return True
            else:
                console.print(f"[red]❌ Installation failed: {result.stderr}[/]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Installation error: {e}[/]")
            return False
    
    def _install_vosk(self) -> bool:
        """Install Vosk."""
        console.print("[cyan]📥 Installing Vosk...[/]")
        
        try:
            import subprocess
            import sys
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "vosk"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green]✅ Vosk installed successfully[/]")
                console.print("[dim]💡 Download language models from: https://alphacephei.com/vosk/models[/]")
                return True
            else:
                console.print(f"[red]❌ Installation failed: {result.stderr}[/]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Installation error: {e}[/]")
            return False
    
    def _install_sphinx(self) -> bool:
        """Install CMU Sphinx."""
        console.print("[cyan]📥 Installing CMU Sphinx...[/]")
        
        try:
            import subprocess
            import sys
            
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "pocketsphinx"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print("[green]✅ CMU Sphinx installed successfully[/]")
                return True
            else:
                console.print(f"[red]❌ Installation failed: {result.stderr}[/]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Installation error: {e}[/]")
            return False
    
    def _show_assistant_voice_settings(self, assistant_name: str):
        """Show current voice settings for an assistant."""
        from library.assistant_cfg import AssistantConfig
        
        assistant_config = AssistantConfig(self.config_dir)
        config = assistant_config.get_assistant(assistant_name)
        
        if not config:
            console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
            return
        
        voice_settings = config.get("voice", {})
        
        settings_text = f"""[bold]Voice Settings for {assistant_name}:[/]

[bold cyan]Text-to-Speech:[/]
• Voice ID: {voice_settings.get('voice_id', 'Not set')}
• Voice Name: {voice_settings.get('voice_name', 'Not set')}
• Speech Rate: {voice_settings.get('speech_rate', 200)}

[bold cyan]Speech Recognition:[/]
• Backend: {voice_settings.get('speech_backend', 'google')}
• Noise Level: {voice_settings.get('noise_level', 'normal')}

[bold cyan]Status:[/]
• Voice Enabled: {'✅' if config.get('capabilities', {}).get('voice_enabled') else '❌'}
"""
        
        console.print(Panel(settings_text, title="Voice Configuration"))