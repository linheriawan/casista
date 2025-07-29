#!/usr/bin/env python3
"""
Coder shortcut commands for common operations
"""

import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: coder [command] [args...]")
        print("Commands:")
        print("  set voice [assistant] [voice_id]     - Set voice for assistant")
        print("  set speech [assistant] [backend]     - Set speech recognition backend")
        print("  set image-models [assistant] [models] - Set image model priority")
        print("  show models                          - Show Ollama models and cache")
        print("  list voices                          - List available TTS voices")
        print("  list speech-backends                 - List speech recognition backends")
        print("  list image-models                    - List cached image models")
        print("  remove image-model [model_name]      - Remove image model from cache")
        print("  preload image-model [model_name]     - Download image model")
        print("  enable rag [assistant]               - Enable RAG for assistant")
        print("  disable rag [assistant]              - Disable RAG for assistant")
        print("  rag status [assistant]               - Show RAG status for assistant")
        print("  index [assistant] [project_path]     - Index project for RAG")
        return
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    main_py = script_dir / "main.py"
    
    command = sys.argv[1].lower()
    
    if command == "set" and len(sys.argv) >= 4:
        subcommand = sys.argv[2].lower()
        
        if subcommand == "voice":
            assistant_name = sys.argv[3]
            voice_id = sys.argv[4] if len(sys.argv) > 4 else "0"
            subprocess.run([sys.executable, str(main_py), "--set-voice", assistant_name, voice_id])
            
        elif subcommand == "speech":
            assistant_name = sys.argv[3]
            backend = sys.argv[4] if len(sys.argv) > 4 else "google"
            subprocess.run([sys.executable, str(main_py), "--set-speech-backend", assistant_name, backend])
            
        elif subcommand == "image-models":
            assistant_name = sys.argv[3]
            models = sys.argv[4] if len(sys.argv) > 4 else "prompthero/openjourney-v4"
            subprocess.run([sys.executable, str(main_py), "--set-image-models", assistant_name, models])
    
    elif command == "show" and len(sys.argv) >= 3 and sys.argv[2].lower() == "models":
        subprocess.run([sys.executable, str(main_py), "--show-models"])
        
    elif command == "list" and len(sys.argv) >= 3:
        subcommand = sys.argv[2].lower()
        
        if subcommand == "voices":
            subprocess.run([sys.executable, str(main_py), "--list-voices"])
        elif subcommand == "speech-backends":
            subprocess.run([sys.executable, str(main_py), "--list-speech-backends"])
        elif subcommand == "image-models":
            subprocess.run([sys.executable, str(main_py), "--list-image-models"])
    
    elif command == "remove" and len(sys.argv) >= 4 and sys.argv[2].lower() == "image-model":
        model_name = sys.argv[3]
        subprocess.run([sys.executable, str(main_py), "--remove-image-model", model_name])
    
    elif command == "preload" and len(sys.argv) >= 4 and sys.argv[2].lower() == "image-model":
        model_name = sys.argv[3]
        subprocess.run([sys.executable, str(main_py), "--preload-image-model", model_name])
    
    elif command == "enable" and len(sys.argv) >= 4 and sys.argv[2].lower() == "rag":
        assistant_name = sys.argv[3]
        subprocess.run([sys.executable, str(main_py), "--enable-rag", assistant_name])
    
    elif command == "disable" and len(sys.argv) >= 4 and sys.argv[2].lower() == "rag":
        assistant_name = sys.argv[3]
        subprocess.run([sys.executable, str(main_py), "--disable-rag", assistant_name])
    
    elif command == "rag" and len(sys.argv) >= 4 and sys.argv[2].lower() == "status":
        assistant_name = sys.argv[3]
        subprocess.run([sys.executable, str(main_py), "--rag-status", assistant_name])
    
    elif command == "index" and len(sys.argv) >= 4:
        assistant_name = sys.argv[2]
        project_path = sys.argv[3]
        subprocess.run([sys.executable, str(main_py), "--index-project", assistant_name, project_path])
    
    else:
        # Pass through to main coder command
        subprocess.run([sys.executable, str(main_py)] + sys.argv[1:])

if __name__ == "__main__":
    main()