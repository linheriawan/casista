#!/usr/bin/env python3
"""
Minimal Piper TTS for streaming text-to-speech processing.
Usage: python piper-speak.py "text to speak"
"""

import sys
import numpy as np
import sounddevice as sd
import subprocess

def speak(text, voice_path="en_US-hfc_female-medium.onnx"):
    """
    Convert text to speech and play immediately on laptop speakers.
    
    Args:
        text (str): Text to speak
        voice_path (str): Path to Piper voice model file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use piper CLI to generate raw audio
        process = subprocess.run(
            ["piper", "-m", voice_path, "--output-raw"],
            input=text.encode('utf-8'),
            capture_output=True,
            check=True
        )
        
        # Convert raw audio bytes to numpy array
        audio_data = np.frombuffer(process.stdout, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Play audio on default speakers (22050 Hz is default for most Piper models)
        sd.play(audio_float, samplerate=22050, blocking=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Piper TTS failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Audio error: {e}")
        return False

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python piper-speak.py 'text to speak'")
        print("       python piper-speak.py 'text' voice_model.onnx")
        sys.exit(1)
    
    text = sys.argv[1]
    voice_model = sys.argv[2] if len(sys.argv) > 2 else "en_US-hfc_female-medium.onnx"
    
    if not speak(text, voice_model):
        sys.exit(1)
if __name__ == "__main__":
    main()