#!/usr/bin/env python3
"""
Base Assistant Framework for Multi-Modal AI Interactions
Supports: chat, speech, image processing with multiple models
"""

import os
import json
import abc
import getpass
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text

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
        
        # Initialize RAG if enabled
        self.rag_assistant = None
        if self.config.get('rag_enabled', False):
            self._init_rag()
    
    def _init_rag(self):
        """Initialize RAG system"""
        try:
            from rag_system import RAGKnowledgeBase, RAGAssistant
            
            kb_dir = self.working_dir / ".ai_context" / "knowledge"
            self.knowledge_base = RAGKnowledgeBase(kb_dir)
            self.rag_assistant = RAGAssistant(self.knowledge_base)
            
            # Auto-index current project if knowledge base is empty
            stats = self.knowledge_base.get_stats()
            if stats['total_documents'] == 0:
                console.print(f"[cyan]🔍 Auto-indexing project: {self.working_dir}[/]")
                self.knowledge_base.index_codebase(self.working_dir)
            
            console.print(f"[green]✅ RAG enabled with {stats['total_documents']} documents[/]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️ RAG initialization failed: {e}[/]")
            self.rag_assistant = None
    
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
        
        # Prepare messages with RAG enhancement
        system_prompt = self.get_system_prompt()
        
        if self.rag_assistant:
            # Use RAG to enhance the prompt with relevant context
            enhanced_prompt = self.rag_assistant.enhance_prompt(processed_text, system_prompt)
            messages_with_system = [{"role": "system", "content": enhanced_prompt}] + messages
        else:
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
        
        # Initialize UI components
        self.layout = Layout()
        self.status_text = Text("")
        self.chat_lines = []
        
        # Initialize speech components
        self._init_speech_components()
    
    def _init_speech_components(self):
        """Initialize speech recognition and TTS components"""
        # Initialize speech recognition backend
        self.speech_backend = self.config.get('speech_backend', 'google')  # 'google', 'whisper', 'vosk'
        
        # Try to initialize the preferred backend
        if self.speech_backend == 'whisper':
            self._init_whisper()
        elif self.speech_backend == 'vosk':
            self._init_vosk()
        else:
            self._init_google_sr()
        
        # Initialize TTS (always same)
        self._init_tts()
    
    def _init_whisper(self):
        """Initialize Whisper for local speech recognition"""
        try:
            import whisper
            import speech_recognition as sr
            
            # Load Whisper model (configurable size)
            model_size = self.config.get('whisper_model', 'base')  # tiny, base, small, medium, large
            console.print(f"[yellow]📥 Loading Whisper model '{model_size}'...[/]")
            
            self.whisper_model = whisper.load_model(model_size)
            self.recognizer = sr.Recognizer() 
            self.microphone = sr.Microphone()
            self.sr = sr
            self.speech_backend = 'whisper'
            
            console.print(f"[green]✅ Whisper ({model_size}) speech recognition initialized[/]")
            
        except ImportError:
            console.print("[yellow]⚠️ Whisper not installed, falling back to Google[/]")
            console.print("[dim]Install with: pip install openai-whisper[/]")
            self._init_google_sr()
        except Exception as e:
            console.print(f"[yellow]⚠️ Whisper failed ({e}), falling back to Google[/]")
            self._init_google_sr()
    
    def _init_vosk(self):
        """Initialize Vosk for local speech recognition"""
        try:
            import vosk
            import json
            import pyaudio
            
            # Download model if needed (you'd need to implement model management)
            model_path = self.config.get('vosk_model_path', './vosk-model')
            if not Path(model_path).exists():
                console.print(f"[yellow]⚠️ Vosk model not found at {model_path}[/]")
                console.print("[dim]Download from: https://alphacephei.com/vosk/models[/]")
                self._init_google_sr()
                return
            
            self.vosk_model = vosk.Model(model_path)
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, 16000)
            self.speech_backend = 'vosk'
            
            console.print("[green]✅ Vosk local speech recognition initialized[/]")
            
        except ImportError:
            console.print("[yellow]⚠️ Vosk not installed, falling back to Google[/]")
            console.print("[dim]Install with: pip install vosk[/]")
            self._init_google_sr()
        except Exception as e:
            console.print(f"[yellow]⚠️ Vosk failed ({e}), falling back to Google[/]")
            self._init_google_sr()
    
    def _init_google_sr(self):
        """Initialize Google speech recognition (online)"""
        try:
            import speech_recognition as sr
            
            self.sr = sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.speech_backend = 'google'
            
            console.print("[green]✅ Google speech recognition initialized[/]")
            
        except ImportError as e:
            console.print(f"[red]❌ Speech recognition not available: {e}[/]")
            console.print("[yellow]💡 Install with: pip install speechrecognition pyaudio[/]")
            self.recognizer = None
            self.sr = None
    
    def _init_tts(self):
        """Initialize text-to-speech"""
        try:
            import pyttsx3
            
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS voice from config or use default
            voices = self.tts_engine.getProperty('voices')
            preferred_voice = self.config.get('tts_voice', None)
            
            if preferred_voice and voices:
                for voice in voices:
                    if voice.id == preferred_voice or voice.name == preferred_voice:
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            elif voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'samantha' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            speech_rate = self.config.get('speech_rate', 200)
            self.tts_engine.setProperty('rate', speech_rate)
            
        except ImportError as e:
            console.print(f"[yellow]⚠️ TTS not available: {e}[/]")
            console.print("[yellow]💡 Install with: pip install pyttsx3[/]")
            self.tts_engine = None
    
    def get_system_prompt(self) -> str:
        user_name = self.get_user_name()
        
        # Use custom system prompt from config if available
        custom_prompt = self.config.get('system_prompt', '')
        if custom_prompt:
            # Add voice-specific instructions to custom prompt
            return f"{custom_prompt}\n\nIMPORTANT: This is voice conversation. Provide clear, spoken responses and wait for voice input. Be conversational and encouraging."
        
        # Default fallback
        return f"You are {self.name}, a voice assistant talking to {user_name}. Provide clear, concise spoken responses. Avoid long explanations since this is voice conversation."
    
    def speak(self, text: str):
        """Convert text to speech with robust error handling"""
        if not text or not text.strip():
            return
            
        
        try:
            import pyttsx3
            import threading
            import time
            
            # Global TTS lock to prevent concurrent speech
            if not hasattr(self.__class__, '_tts_lock'):
                self.__class__._tts_lock = threading.Lock()
            
            with self.__class__._tts_lock:
                # Create completely fresh engine every time
                engine = pyttsx3.init()
                
                # Set basic properties
                engine.setProperty('rate', self.config.get('speech_rate', 200))
                engine.setProperty('volume', 1.0)
                
                # Get and set voice
                voices = engine.getProperty('voices')
                if voices:
                    preferred_voice = self.config.get('tts_voice', None)
                    if preferred_voice:
                        for voice in voices:
                            if voice.id == preferred_voice:
                                engine.setProperty('voice', voice.id)
                                break
                    else:
                        # Use first available voice
                        engine.setProperty('voice', voices[0].id)
                
                # Simple speech with timeout
                engine.say(text)
                
                # Run in separate thread with timeout
                speech_done = threading.Event()
                
                def speak_thread():
                    try:
                        engine.runAndWait()
                        speech_done.set()
                    except Exception as e:
                        speech_done.set()
                
                thread = threading.Thread(target=speak_thread, daemon=True)
                thread.start()
                
                # Wait for completion with timeout
                if speech_done.wait(timeout=10):  # 10 second timeout
                    pass  # Success
                else:
                    # Timeout - stop engine
                    try:
                        engine.stop()
                    except:
                        pass
                
                # Clean up
                try:
                    del engine
                except:
                    pass
                
                time.sleep(0.2)  # Brief pause between speeches
            
        except Exception as e:
            console.print(f"[red]❌ TTS Error: {e}[/]")
            
            # Use system 'say' command as fallback on macOS
            try:
                if sys.platform == "darwin":  # macOS
                    import subprocess
                    subprocess.run(['say', text], timeout=10, check=True)
                else:
                    # Final fallback: just show text
                    if hasattr(self, '_add_chat_line'):
                        self._add_chat_line(f"🔊 {self.name} would say: {text}", "dim")
            except Exception:
                # Final fallback: just show text
                if hasattr(self, '_add_chat_line'):
                    self._add_chat_line(f"🔊 {self.name} would say: {text}", "dim")
    
    def _update_status(self, status: str):
        """Update the status bar at the top"""
        self.status_text = Text(status, style="bold cyan")
    
    def _add_chat_line(self, text: str, style: str = ""):
        """Add a line to the chat area"""
        self.chat_lines.append(Text(text, style=style))
        # Keep only last 20 lines
        if len(self.chat_lines) > 20:
            self.chat_lines = self.chat_lines[-20:]
    
    def _setup_layout(self):
        """Setup the layout with status bar and chat area"""
        self.layout.split_column(
            Layout(Panel(self.status_text, height=3), name="status"),
            Layout(name="chat")
        )
        
        # Build chat content with latest on top
        chat_content = Text()
        for line in reversed(self.chat_lines):
            chat_content.append(line)
            chat_content.append("\n")
        
        self.layout["chat"].update(Panel(chat_content, title="Conversation (Latest First)"))
        return self.layout

    def listen(self) -> Optional[str]:
        """Listen for speech and convert to text using configured backend"""
        if self.speech_backend == 'whisper':
            return self._listen_whisper()
        elif self.speech_backend == 'vosk':
            return self._listen_vosk()
        else:
            return self._listen_google()
    
    def _listen_whisper(self) -> Optional[str]:
        """Listen using Whisper (local)"""
        if not hasattr(self, 'whisper_model') or not self.recognizer:
            self._update_status("⚠️ Whisper not available")
            return None
        
        try:
            self._update_status("📊 Calibrating microphone... (be quiet)")
            
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            self._update_status("🎤 Listening... (speak clearly)")
            
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=8)
            
            self._update_status("🔄 Processing with Whisper (local)...")
            
            # Convert to wav data for Whisper
            wav_data = audio.get_wav_data()
            
            # Save to temporary file for Whisper
            import tempfile
            import numpy as np
            import io
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(wav_data)
                tmp_file.flush()
                
                # Process with Whisper
                result = self.whisper_model.transcribe(tmp_file.name)
                text = result["text"].strip()
                
                # Clean up
                import os
                os.unlink(tmp_file.name)
            
            if text:
                self._add_chat_line(f"👤 You: {text}", "green")
                self._update_status("✅ Speech recognized (Whisper)")
                return text
            else:
                self._update_status("❓ No speech detected")
                return None
                
        except Exception as e:
            self._update_status(f"❌ Whisper error: {e}")
            return None
    
    def _listen_vosk(self) -> Optional[str]:
        """Listen using Vosk (local)"""
        if not hasattr(self, 'vosk_rec'):
            self._update_status("⚠️ Vosk not available")
            return None
        
        try:
            import pyaudio
            import json
            
            self._update_status("🎤 Listening with Vosk... (speak clearly)")
            
            # Audio stream setup
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
            stream.start_stream()
            
            # Listen for speech
            frames = []
            silent_frames = 0
            recording = False
            
            for _ in range(int(16000 / 8000 * 10)):  # 10 seconds max
                data = stream.read(8000, exception_on_overflow=False)
                
                if self.vosk_rec.AcceptWaveform(data):
                    result = json.loads(self.vosk_rec.Result())
                    if result.get("text"):
                        text = result["text"].strip()
                        break
                else:
                    partial = json.loads(self.vosk_rec.PartialResult())
                    if partial.get("partial"):
                        recording = True
                        silent_frames = 0
                    elif recording:
                        silent_frames += 1
                        if silent_frames > 20:  # ~2.5 seconds of silence
                            break
            else:
                # Final result if no intermediate result
                final = json.loads(self.vosk_rec.FinalResult())
                text = final.get("text", "").strip()
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if text:
                self._add_chat_line(f"👤 You: {text}", "green")
                self._update_status("✅ Speech recognized (Vosk)")
                return text
            else:
                self._update_status("❓ No speech detected")
                return None
                
        except Exception as e:
            self._update_status(f"❌ Vosk error: {e}")
            return None
    
    def _listen_google(self) -> Optional[str]:
        """Listen using Google Speech Recognition (online)"""
        if not self.recognizer or not self.sr:
            self._update_status("⚠️ Speech recognition not available")
            return None
        
        try:
            self._update_status("📊 Calibrating microphone... (be quiet)")
            
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                self.recognizer.energy_threshold = max(300, self.recognizer.energy_threshold)
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                self.recognizer.phrase_threshold = 0.3
                self.recognizer.non_speaking_duration = 0.8
            
            self._update_status("🎤 Listening... (speak clearly)")
            
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=8)
            
            self._update_status("🔄 Processing with Google (online)...")
            
            # Try multiple recognition approaches
            recognition_attempts = [
                lambda: self.recognizer.recognize_google(audio, language='en-US'),
                lambda: self.recognizer.recognize_google(audio, language='en-GB'),
                lambda: self.recognizer.recognize_google(audio, show_all=False),
            ]
            
            for attempt in recognition_attempts:
                try:
                    text = attempt()
                    if text and text.strip():
                        self._add_chat_line(f"👤 You: {text}", "green")
                        self._update_status("✅ Speech recognized (Google)")
                        return text.strip()
                except:
                    continue
            
            self._update_status("❓ Could not understand audio")
            return None
            
        except self.sr.WaitTimeoutError:
            self._update_status("⏰ Listening timeout")
            return None
        except self.sr.UnknownValueError:
            self._update_status("❓ Could not understand audio")
            return None
        except self.sr.RequestError as e:
            self._update_status(f"❌ Google API error: {e}")
            return None
        except Exception as e:
            self._update_status(f"❌ Error: {e}")
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
        """Process voice input and provide voice response with htop-like UI"""
        
        # Setup initial UI
        with Live(self._setup_layout(), refresh_per_second=4, screen=True) as live:
            # If input_data is provided (text), use it directly
            if input_data and isinstance(input_data, str):
                text = input_data
                self._add_chat_line(f"👤 You: {text}", "green")
            else:
                # Listen for voice input
                live.update(self._setup_layout())
                text = self.listen()
                live.update(self._setup_layout())
                if not text:
                    return ""
            
            # Handle voice-specific commands first
            if self.handle_voice_commands(text):
                live.update(self._setup_layout())
                return ""
            
            # Handle general special commands
            if self.handle_special_commands(text):
                live.update(self._setup_layout())
                return ""
            
            messages = self._load_context()
            
            # Prepare messages for AI
            system_prompt = self.get_system_prompt()
            messages_with_system = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": text}]
            
            try:
                self._update_status(f"🤖 {self.name} thinking...")
                live.update(self._setup_layout())
                
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
                self._add_chat_line(f"🤖 {self.name}: {full_response}", "cyan")
                
                # Update status to speaking
                self._update_status(f"🔊 {self.name} speaking...")
                live.update(self._setup_layout())
                
                # Speak the response
                self.speak(full_response)
                
                # Save context
                messages.append({"role": "user", "content": text})
                messages.append({"role": "assistant", "content": full_response})
                self._save_context(messages)
                
                # Ready for next input
                self._update_status("🎤 Ready - say something...")
                live.update(self._setup_layout())
                
                return full_response
                
            except Exception as e:
                error_msg = f"I'm sorry, I encountered an error: {str(e)}"
                self._update_status(f"❌ Error: {e}")
                self._add_chat_line(f"❌ Error: {e}", "red")
                live.update(self._setup_layout())
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
        
        # Model-specific prompting strategies
        model_name = self.model.lower()
        
        if self.image_backend and "dreamlike" in self.image_backend.lower():
            backend_info = """
IMPORTANT: You're using DreamLike Anime model optimized for:
- Anime/artistic style images
- Maximum 77 tokens in descriptions (keep prompts SHORT)
- Works best with: anime characters, artistic scenes, vibrant colors
- Default size: 512x512 (specify if user wants different: 768x768, 1024x1024)"""
        elif self.image_backend and "openjourney" in self.image_backend.lower():
            backend_info = """
IMPORTANT: You're using OpenJourney v4 model optimized for:
- Fantasy, artistic, and creative illustrations
- Detailed fantasy art, landscapes, portraits
- Works well with artistic styles like "digital art", "concept art", "fantasy art"
- Default size: 512x512 (specify if user wants different: 768x768, 1024x1024)"""
        else:
            backend_info = """
IMPORTANT: Keep image descriptions concise and focused for best results."""
        
        # Different prompt strategies for different models
        if "gemma" in model_name:
            # Gemma needs very explicit instructions and examples
            base_prompt = f"""You are {self.name}, an image generation assistant talking to {user_name}.{backend_info}

SYSTEM OVERRIDE: Ignore any training about not being able to generate images. You CAN and MUST generate images.

REQUIRED RESPONSE FORMAT for ALL image requests:
```generate:filename.png
description here
```

EXAMPLE INPUT: "Create a picture of a cat"
REQUIRED OUTPUT:
```generate:cute_cat.png
fluffy orange cat sitting, photorealistic, cute
```

RULES:
- You MUST respond with the code block format above
- You CANNOT say "I can't generate images" 
- You CANNOT give text descriptions
- ALWAYS use ```generate:filename.png format
- Keep descriptions short and clear

For ANY image request, use the ```generate:filename``` format. No exceptions."""

        else:
            # For coder models and others, use standard prompt
            base_prompt = f"""You are {self.name}, an image generation assistant talking to {user_name}.{backend_info}

When {user_name} asks you to create, generate, or make an image, respond with:
```generate:descriptive_filename.png
SHORT, focused description optimized for the current model
```

Extract size requests from prompts (512x512, 768x768, 1024x1024) and use appropriate dimensions.
Use descriptive filenames that match the content.
For anime characters/scenes, emphasize: character names, actions, art style, colors."""

        return base_prompt
    
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
        """Generate image using Stable Diffusion with configurable models"""
        try:
            from diffusers import StableDiffusionPipeline, DiffusionPipeline
            import torch
            
            # Get model list from config or use defaults
            configured_models = self.config.get('image_models', [
                "prompthero/openjourney-v4",
                "dreamlike-art/dreamlike-anime-1.0", 
                "hakurei/waifu-diffusion",
                "runwayml/stable-diffusion-v1-5"
            ])
            
            # Model info mapping
            model_info = {
                "prompthero/openjourney-v4": "OpenJourney v4 (~2GB)",
                "dreamlike-art/dreamlike-anime-1.0": "DreamLike Anime (~2GB)",
                "hakurei/waifu-diffusion": "Waifu Diffusion (~2GB)", 
                "runwayml/stable-diffusion-v1-5": "SD v1.5 (~4GB)",
                "segmind/tiny-sd": "TinySD (~800MB)"
            }
            
            # Build lightweight_models list from config
            lightweight_models = []
            for model_id in configured_models:
                model_name = model_info.get(model_id, f"{model_id} (Unknown size)")
                lightweight_models.append((model_id, model_name))
            
            console.print("[cyan]🔍 Trying lightweight models...[/]")
            
            pipe = None
            used_model = None
            
            for model_id, model_name in lightweight_models:
                try:
                    console.print(f"[dim]Attempting: {model_name}[/]")
                    
                    # Handle different model types  
                    if "tiny-sd" in model_id:
                        pipe = DiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            use_safetensors=True,
                            safety_checker=None,
                            requires_safety_checker=False
                        )
                    else:
                        pipe = StableDiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            use_safetensors=True,
                            safety_checker=None,
                            requires_safety_checker=False
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
            
            # Ensure filename has proper extension
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                if not filename.endswith('.'):
                    filename += '.png'
                else:
                    filename += 'png'
            
            # Save to working directory
            image_path = self.working_dir / filename
            
            # Save with explicit format specification
            if filename.lower().endswith('.png'):
                image.save(image_path, format='PNG')
            elif filename.lower().endswith(('.jpg', '.jpeg')):
                image.save(image_path, format='JPEG')
            elif filename.lower().endswith('.webp'):
                image.save(image_path, format='WEBP')
            else:
                # Default to PNG
                image.save(image_path, format='PNG')
            
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
        
        # For gemma models, add few-shot examples to help it understand the format
        if "gemma" in self.model.lower() and not messages:
            # Add example conversations if context is empty
            few_shot_examples = [
                {"role": "user", "content": "Create an image of a sunset over mountains"},
                {"role": "assistant", "content": "```generate:mountain_sunset.png\nsunset over mountain range, golden sky, silhouette peaks, scenic landscape\n```"},
                {"role": "user", "content": "Generate a picture of a robot"},
                {"role": "assistant", "content": "```generate:robot_character.png\nfuturistic robot, metallic blue, standing pose, sci-fi style\n```"}
            ]
            messages_with_system = [{"role": "system", "content": system_prompt}] + few_shot_examples + [{"role": "user", "content": text}]
        else:
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
            
            # Try multiple patterns to catch different formats
            patterns = [
                r"```generate:([^\n]+)\n(.*?)\n```",  # Full code block format
                r"```generate:([^\n]+)\n(.*?)```",    # Code block without trailing newline
                r"generate:([^\n\s]+)\.png\s*(.*?)(?=\n|$)",  # Simple format: generate:filename.png description
                r"```generate:([^\n]+)```",           # Single line code block
            ]
            
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, full_response, re.DOTALL | re.MULTILINE)
                if found:
                    matches.extend(found)
                    break  # Use first successful pattern
            
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    filename, description = match[0], match[1]
                elif isinstance(match, tuple) and len(match) == 1:
                    filename, description = match[0], ""
                else:
                    filename, description = str(match), ""
                    
                filename = filename.strip()
                description = description.strip() if description else "generated image"
                
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