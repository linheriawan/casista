#!/usr/bin/env python3
"""
Voice selection tool for speech assistants
Allows you to choose from your downloaded system voices
"""

import json
import pyttsx3
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def list_available_voices():
    """List all available system voices"""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        if not voices:
            console.print("[red]❌ No voices found[/]")
            return []
        
        console.print(f"[green]Found {len(voices)} system voices:[/]\n")
        
        # Create table
        table = Table(title="Available System Voices")
        table.add_column("ID", style="dim", width=3)
        table.add_column("Name", style="cyan", min_width=20)
        table.add_column("Language", style="green", width=15)
        table.add_column("Gender", style="magenta", width=10)
        table.add_column("Voice ID", style="dim", width=40)
        
        for i, voice in enumerate(voices):
            name = voice.name
            voice_id = voice.id
            
            # Try to detect language and gender from name/id
            language = "Unknown"
            gender = "Unknown"
            
            # Common language patterns
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
            
            # Common gender patterns
            if any(word in name.lower() for word in ['female', 'woman', 'girl', 'samantha', 'alex', 'victoria', 'allison', 'ava', 'susan', 'karen']):
                gender = "Female"
            elif any(word in name.lower() for word in ['male', 'man', 'boy', 'daniel', 'fred', 'jorge', 'albert', 'bruce', 'ralph']):
                gender = "Male"
            
            table.add_row(str(i), name, language, gender, voice_id[:40] + "..." if len(voice_id) > 40 else voice_id)
        
        console.print(table)
        return voices
        
    except Exception as e:
        console.print(f"[red]❌ Error listing voices: {e}[/]")
        return []

def test_voice(voice_id: str, voice_name: str):
    """Test a specific voice"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('voice', voice_id)
        engine.setProperty('rate', 200)
        
        test_text = f"Hello! I'm {voice_name}. This is how I sound."
        console.print(f"[cyan]🔊 Testing voice: {voice_name}[/]")
        
        engine.say(test_text)
        engine.runAndWait()
        
    except Exception as e:
        console.print(f"[red]❌ Error testing voice: {e}[/]")

def set_assistant_voice(assistant_name: str, voice_id: str, voice_name: str):
    """Set the voice for a specific assistant"""
    context_dir = Path(".ai_context") / assistant_name
    config_file = context_dir / "config.json"
    
    if not config_file.exists():
        console.print(f"[red]❌ Assistant '{assistant_name}' not found[/]")
        console.print(f"Create it first: coder qwen2.5-coder:3b {assistant_name} speech")
        return False
    
    # Load and update config
    config = json.loads(config_file.read_text())
    config['tts_voice'] = voice_id
    config['tts_voice_name'] = voice_name
    
    # Save updated config
    config_file.write_text(json.dumps(config, indent=2))
    
    console.print(f"[green]✅ Set {assistant_name}'s voice to: {voice_name}[/]")
    return True

def main():
    import sys
    
    console.print(Panel.fit(
        "[bold cyan]🎤 Voice Selector for Speech Assistants[/]\n"
        "Choose from your downloaded system voices",
        title="Voice Selection Tool"
    ))
    
    if len(sys.argv) < 2:
        console.print("\n[yellow]Usage:[/]")
        console.print("  python3 voice_selector.py list                    # List all voices")
        console.print("  python3 voice_selector.py test <voice_id>         # Test a voice")
        console.print("  python3 voice_selector.py set <assistant> <id>   # Set assistant voice")
        console.print("\n[dim]Examples:[/]")
        console.print("  python3 voice_selector.py list")
        console.print("  python3 voice_selector.py test 15")
        console.print("  python3 voice_selector.py set anna 15")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        voices = list_available_voices()
        if voices:
            console.print(f"\n[cyan]💡 Tips:[/]")
            console.print("• Test voices: python3 voice_selector.py test <ID>")
            console.print("• Set for assistant: python3 voice_selector.py set <assistant_name> <ID>")
    
    elif command == "test":
        if len(sys.argv) < 3:
            console.print("[red]❌ Please specify voice ID to test[/]")
            return
        
        try:
            voice_index = int(sys.argv[2])
            voices = list_available_voices()
            
            if 0 <= voice_index < len(voices):
                voice = voices[voice_index]
                test_voice(voice.id, voice.name)
            else:
                console.print(f"[red]❌ Invalid voice ID. Use 0-{len(voices)-1}[/]")
                
        except ValueError:
            console.print("[red]❌ Voice ID must be a number[/]")
    
    elif command == "set":
        if len(sys.argv) < 4:
            console.print("[red]❌ Usage: python3 voice_selector.py set <assistant_name> <voice_id>[/]")
            return
        
        assistant_name = sys.argv[2]
        
        try:
            voice_index = int(sys.argv[3])
            voices = list_available_voices()
            
            if 0 <= voice_index < len(voices):
                voice = voices[voice_index]
                if set_assistant_voice(assistant_name, voice.id, voice.name):
                    console.print(f"\n[green]🎉 Voice updated successfully![/]")
                    console.print(f"Test it: coder qwen2.5-coder:3b {assistant_name} speech")
            else:
                console.print(f"[red]❌ Invalid voice ID. Use 0-{len(voices)-1}[/]")
        
        except ValueError:
            console.print("[red]❌ Voice ID must be a number[/]")
    
    else:
        console.print(f"[red]❌ Unknown command: {command}[/]")
        console.print("Available commands: list, test, set")

if __name__ == "__main__":
    main()