#!/usr/bin/env python3
"""
Tool to customize voice assistant personality and speaking style
"""

import json
from pathlib import Path

def customize_voice_assistant(assistant_name: str, personality: str = None):
    """Customize the voice assistant's personality and speaking style"""
    
    # Find the assistant's config
    context_dir = Path(".ai_context") / assistant_name
    config_file = context_dir / "config.json"
    
    if not config_file.exists():
        print(f"❌ Assistant '{assistant_name}' not found. Create it first by running:")
        print(f"   coder qwen2.5-coder:3b {assistant_name} speech")
        return
    
    # Load current config
    config = json.loads(config_file.read_text())
    
    # Personality presets
    personalities = {
        "friendly": "You are {name}, a friendly and cheerful voice assistant talking to {user}. Speak in a warm, conversational tone. Keep responses short and engaging since this is voice conversation.",
        
        "professional": "You are {name}, a professional AI assistant talking to {user}. Speak clearly and concisely. Provide helpful, accurate information in a business-like manner.",
        
        "casual": "You are {name}, a casual and relaxed voice assistant talking to {user}. Speak naturally like a friend. Use simple language and be conversational.",
        
        "enthusiastic": "You are {name}, an enthusiastic and energetic voice assistant talking to {user}. Speak with excitement and positivity! Keep responses lively but brief for voice conversation.",
        
        "wise": "You are {name}, a wise and thoughtful voice assistant talking to {user}. Speak slowly and deliberately. Provide thoughtful insights and gentle guidance.",
        
        "custom": personality  # Use provided custom personality
    }
    
    print(f"Current system prompt for {assistant_name}:")
    print(f"'{config.get('system_prompt', 'Not set')}'")
    print()
    
    if personality and personality in personalities:
        # Apply personality
        new_prompt = personalities[personality].format(
            name=assistant_name,
            user=config.get('user_name', 'user')
        )
        config['system_prompt'] = new_prompt
        
        # Save updated config
        config_file.write_text(json.dumps(config, indent=2))
        print(f"✅ Updated {assistant_name} with {personality} personality:")
        print(f"'{new_prompt}'")
        
    else:
        # Show available personalities
        print("Available personality presets:")
        for preset, prompt in personalities.items():
            if preset != "custom":
                print(f"  {preset}: {prompt.format(name=assistant_name, user=config.get('user_name', 'user'))}")
        print()
        print("Usage:")
        print(f"  python3 set_voice_personality.py {assistant_name} friendly")
        print(f'  python3 set_voice_personality.py {assistant_name} "custom prompt here"')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 set_voice_personality.py <assistant_name> [personality]")
        print("Example: python3 set_voice_personality.py anna friendly")
        sys.exit(1)
    
    assistant_name = sys.argv[1]
    personality = sys.argv[2] if len(sys.argv) > 2 else None
    
    customize_voice_assistant(assistant_name, personality)