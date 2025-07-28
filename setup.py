#!/usr/bin/env python3
"""
Setup script for the AI Assistant system
Handles virtual environment creation and dependency installation
"""

import os
import sys
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()

def run_command(cmd, description):
    """Run a command and show progress"""
    console.print(f"[cyan]📦 {description}...[/]")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        console.print(f"[green]✅ {description} complete[/]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]❌ {description} failed: {e.stderr}[/]")
        return False

def main():
    current_dir = Path(__file__).parent.absolute()
    venv_dir = current_dir / "venv"
    
    console.print("[bold cyan]🚀 Setting up AI Assistant System[/]")
    
    # Check Python version
    if sys.version_info < (3, 8):
        console.print("[red]❌ Python 3.8+ required[/]")
        return
    
    # Create virtual environment
    if not venv_dir.exists():
        if not run_command(f"python3 -m venv {venv_dir}", "Creating virtual environment"):
            return
    else:
        console.print("[green]✅ Virtual environment already exists[/]")
    
    # Install dependencies
    pip_cmd = f"{venv_dir}/bin/pip"
    base_dependencies = "ollama rich typer prompt_toolkit"
    speech_dependencies = "speechrecognition pyttsx3 pyaudio"
    
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return
    
    if not run_command(f"{pip_cmd} install {base_dependencies}", "Installing base dependencies"):
        return
    
    # Try to install speech dependencies (optional)
    console.print("[cyan]📢 Installing speech dependencies (optional)...[/]")
    result = subprocess.run(f"{pip_cmd} install {speech_dependencies}", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        console.print("[green]✅ Speech dependencies installed[/]")
    else:
        console.print("[yellow]⚠️ Speech dependencies failed to install (speech mode won't work)[/]")
        console.print("[dim]You may need to install system audio libraries first[/]")
        if "darwin" in sys.platform:  # macOS
            console.print("[dim]Try: brew install portaudio[/]")
    
    # Make scripts executable
    for script in ["main.py", "install.py", "setup.py"]:
        script_path = current_dir / script
        if script_path.exists():
            script_path.chmod(0o755)
    
    console.print("\n[bold green]🎉 Setup complete![/]")
    console.print("\n[cyan]Next steps:[/]")
    console.print("1. [dim]python3 install.py[/] - Install the 'coder' command globally")
    console.print("2. [dim]coder --list-models[/] - Check available Ollama models")
    console.print("3. [dim]coder qwen2.5-coder:3b mycoder chat[/] - Start chatting!")
    console.print("\n[yellow]💡 Make sure Ollama is running with some models installed[/]")

if __name__ == "__main__":
    main()