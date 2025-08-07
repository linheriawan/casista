#!/usr/bin/env python3
"""
Speech Handler - Manages Text-to-Speech and Speech-to-Text capabilities.

Provides enhanced voice input/output functionality with support for multiple backends.
"""

import tempfile
import wave
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console

console = Console()

class SpeechHandler:
    """Handles speech recognition and text-to-speech functionality."""
    
    def __init__(self, voice_config: Dict[str, Any]):
        """Initialize speech handler with voice configuration."""
        self.voice_config = voice_config
        self.speech_backend = voice_config.get("speech_backend", "google")
        self.voice_id = voice_config.get("voice_id")
        self.voice_name = voice_config.get("voice_name", "Default")
        self.speech_rate = voice_config.get("speech_rate", 200)
        
        # Components that will be initialized
        self.recognizer = None
        self.microphone = None
        self.tts_engine = None
        self.whisper_model = None
        
        # State
        self.is_initialized = False
    
    def setup_tts(self) -> bool:
        """Setup text-to-speech engine."""
        try:
            import pyttsx3
            
            self.tts_engine = pyttsx3.init()
            
            # Configure voice if specified
            if self.voice_id:
                voices = self.tts_engine.getProperty('voices')
                target_voice = None
                
                # Try to find voice by ID (number) or by voice object ID
                try:
                    voice_index = int(self.voice_id)
                    if 0 <= voice_index < len(voices):
                        target_voice = voices[voice_index]
                except ValueError:
                    # Try as voice ID string
                    for voice in voices:
                        if voice.id == self.voice_id:
                            target_voice = voice
                            break
                
                if target_voice:
                    self.tts_engine.setProperty('voice', target_voice.id)
                    console.print(f"[green]✅ Using voice: {target_voice.name}[/]")
                else:
                    console.print(f"[yellow]⚠️ Voice ID '{self.voice_id}' not found, using default[/]")
            
            # Set speech rate
            self.tts_engine.setProperty('rate', self.speech_rate)
            console.print(f"[green]✅ TTS initialized with rate {self.speech_rate} WPM[/]")
            return True
            
        except ImportError:
            console.print("[red]❌ TTS dependencies not installed[/]")
            console.print("[dim]Install with: pip install pyttsx3[/]")
            return False
        except Exception as e:
            console.print(f"[red]❌ TTS setup failed: {e}[/]")
            return False
    
    def setup_stt(self) -> bool:
        """Setup speech-to-text recognition."""
        try:
            import speech_recognition as sr
            
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Setup backend-specific components
            if self.speech_backend == "whisper":
                if not self._setup_whisper():
                    console.print(f"[yellow]⚠️ Whisper setup failed, falling back to Google[/]")
                    self.speech_backend = "google"
            elif self.speech_backend == "vosk":
                if not self._setup_vosk():
                    console.print(f"[yellow]⚠️ Vosk setup failed, falling back to Google[/]")
                    self.speech_backend = "google"
            
            console.print(f"[green]✅ STT initialized with backend: {self.speech_backend}[/]")
            return True
            
        except ImportError:
            console.print("[red]❌ STT dependencies not installed[/]")
            console.print("[dim]Install with: pip install speechrecognition[/]")
            return False
        except Exception as e:
            console.print(f"[red]❌ STT setup failed: {e}[/]")
            return False
    
    def _setup_whisper(self) -> bool:
        """Setup Whisper speech recognition."""
        try:
            import whisper
            model_size = self.voice_config.get("whisper_model", "base")
            self.whisper_model = whisper.load_model(model_size, device="cpu", download_root=None)
            console.print(f"[green]✅ Whisper model '{model_size}' loaded[/]")
            return True
        except ImportError:
            console.print("[dim]Whisper not installed. Install with: pip install openai-whisper[/]")
            return False
        except Exception as e:
            console.print(f"[yellow]⚠️ Whisper setup error: {e}[/]")
            return False
    
    def _setup_vosk(self) -> bool:
        """Setup Vosk speech recognition."""
        try:
            import vosk
            # Vosk setup would be more complex - model path validation, etc.
            console.print(f"[green]✅ Vosk backend available[/]")
            return True
        except ImportError:
            console.print("[dim]Vosk not installed. Install with: pip install vosk[/]")
            return False
        except Exception as e:
            console.print(f"[yellow]⚠️ Vosk setup error: {e}[/]")
            return False
    
    def calibrate_microphone(self) -> bool:
        """Calibrate microphone for ambient noise."""
        if not self.microphone or not self.recognizer:
            console.print("[red]❌ Microphone not initialized[/]")
            return False
        
        try:
            with self.microphone as source:
                console.print("[dim]🎤 Calibrating for ambient noise...", end="")
                self.recognizer.adjust_for_ambient_noise(source)
                console.print(" Done!")
            return True
        except Exception as e:
            console.print(f"[yellow]⚠️ Microphone calibration failed: {e}[/]")
            return False
    
    def listen(self, timeout: int = 30, phrase_time_limit: int = 300) -> Optional[str]:
        """Listen for speech and return transcribed text."""
        if not self.recognizer or not self.microphone:
            console.print("[red]❌ Speech recognition not initialized[/]")
            return None
        
        try:
            console.print("🎤 Listening...", end="")
            
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            console.print(" Processing...", end="")
            
            # Recognize speech using configured backend
            if self.speech_backend == "whisper" and self.whisper_model:
                text = self._recognize_with_whisper(audio)
            elif self.speech_backend == "vosk":
                text = self._recognize_with_vosk(audio)
            else:
                # Fall back to Google
                text = self.recognizer.recognize_google(audio)
            
            if text:
                console.print(f" Heard: '{text}'")
                return text.strip()
            else:
                console.print(" No speech detected")
                return None
                
        except Exception as e:
            import speech_recognition as sr
            if isinstance(e, sr.UnknownValueError):
                console.print(" Could not understand audio")
            elif isinstance(e, sr.RequestError):
                console.print(f" Speech service error: {e}")
            elif isinstance(e, sr.WaitTimeoutError):
                console.print(" Timeout - no speech detected")
            else:
                console.print(f" Speech recognition error: {e}")
            return None
    
    def _recognize_with_whisper(self, audio) -> Optional[str]:
        """Recognize speech using Whisper."""
        try:
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(audio.sample_width)
                    wav_file.setframerate(audio.sample_rate)
                    wav_file.writeframes(audio.frame_data)
                
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(tmp_file.name)
                return result["text"]
        except Exception as e:
            console.print(f"[yellow]⚠️ Whisper recognition failed: {e}[/]")
            return None
    
    def _recognize_with_vosk(self, audio) -> Optional[str]:
        """Recognize speech using Vosk."""
        try:
            return self.recognizer.recognize_vosk(audio)
        except Exception as e:
            console.print(f"[yellow]⚠️ Vosk recognition failed: {e}[/]")
            return None
    
    def speak(self, text: str) -> bool:
        """Speak the given text using TTS."""
        if not self.tts_engine:
            console.print("[red]❌ TTS engine not initialized[/]")
            return False
        
        if not text or not text.strip():
            console.print("[dim](No speakable content)[/]")
            return False
        
        try:
            self.tts_engine.say(text.strip())
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            console.print(f"[red]❌ TTS failed: {e}[/]")
            return False
    
    def cleanup(self):
        """Clean up speech resources."""
        try:
            if self.tts_engine:
                self.tts_engine.stop()
                self.tts_engine = None
        except Exception:
            pass  # Ignore cleanup errors
        
        self.recognizer = None
        self.microphone = None
        self.whisper_model = None
        self.is_initialized = False
        console.print("[dim]🧹 Speech handler cleaned up[/]")
    
    def is_ready(self) -> bool:
        """Check if speech handler is ready for use."""
        return (self.tts_engine is not None and 
                self.recognizer is not None and 
                self.microphone is not None)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current speech handler status."""
        return {
            "speech_backend": self.speech_backend,
            "voice_name": self.voice_name,
            "voice_id": self.voice_id,
            "speech_rate": self.speech_rate,
            "tts_ready": self.tts_engine is not None,
            "stt_ready": self.recognizer is not None,
            "whisper_loaded": self.whisper_model is not None,
            "is_ready": self.is_ready()
        }