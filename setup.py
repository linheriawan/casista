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

def install_requirements(pip_cmd: str, requirements_file: Path, group_name: str, required: bool = True):
    """Install requirements from a specific group"""
    console.print(f"[cyan]📦 Installing {group_name}...[/]")
    
    try:
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        # Parse requirements by group
        current_section = "core"
        requirements = {"core": [], "speech": [], "rag": [], "development": [], "optional": []}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                if "Speech Dependencies" in line:
                    current_section = "speech"
                elif "RAG Dependencies" in line:
                    current_section = "rag"
                elif "Development" in line:
                    current_section = "development"
                elif "Optional" in line:
                    current_section = "optional"
                continue
            
            if not line.startswith('#') and '>' in line:
                requirements[current_section].append(line)
        
        # Install the requested group
        if group_name.lower() in requirements and requirements[group_name.lower()]:
            deps = ' '.join(requirements[group_name.lower()])
            result = subprocess.run(f"{pip_cmd} install {deps}", shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[green]✅ {group_name} dependencies installed[/]")
                return True
            else:
                if required:
                    console.print(f"[red]❌ {group_name} dependencies failed: {result.stderr}[/]")
                    return False
                else:
                    console.print(f"[yellow]⚠️ {group_name} dependencies failed (optional)[/]")
                    return True
        else:
            console.print(f"[yellow]⚠️ No {group_name} dependencies found[/]")
            return True
            
    except Exception as e:
        console.print(f"[red]❌ Error installing {group_name}: {e}[/]")
        return not required

def main():
    current_dir = Path(__file__).parent.absolute()
    venv_dir = current_dir / "venv"
    requirements_file = current_dir / "requirements.txt"
    
    console.print("[bold cyan]🚀 Setting up AI Assistant System[/]")
    
    # Check Python version
    if sys.version_info < (3, 8):
        console.print("[red]❌ Python 3.8+ required[/]")
        return
    
    # Check if requirements.txt exists
    if not requirements_file.exists():
        console.print("[red]❌ requirements.txt not found[/]")
        return
    
    # Create virtual environment
    if not venv_dir.exists():
        if not run_command(f"python3 -m venv {venv_dir}", "Creating virtual environment"):
            return
    else:
        console.print("[green]✅ Virtual environment already exists[/]")
    
    # Install dependencies
    pip_cmd = f"{venv_dir}/bin/pip"
    
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return
    
    # Install core dependencies
    if not install_requirements(pip_cmd, requirements_file, "core", required=True):
        return
    
    # Install optional dependencies
    install_requirements(pip_cmd, requirements_file, "speech", required=False)
    install_requirements(pip_cmd, requirements_file, "rag", required=False)
    
    # Special handling for audio dependencies on macOS
    if "darwin" in sys.platform.lower():
        console.print("[yellow]💡 macOS detected - if speech installation failed, try:[/]")
        console.print("[dim]brew install portaudio[/]")
        console.print("[dim]Then run: pip install pyaudio[/]")
    
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