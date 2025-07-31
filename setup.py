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

def parse_requirements(requirements_file: Path):
    """Parse requirements.txt into groups"""
    requirements = {"core": [], "speech": [], "rag": [], "documents": [], "development": [], "optional": []}
    
    try:
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        current_section = "core"
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments that don't define sections
            if not line:
                continue
                
            # Check for section headers in comments
            if line.startswith('#'):
                if "Core Dependencies" in line:
                    current_section = "core"
                elif "Speech Dependencies" in line:
                    current_section = "speech"
                elif "RAG Dependencies" in line:
                    current_section = "rag"
                elif "Document Processing" in line:
                    current_section = "documents"
                elif "Development" in line:
                    current_section = "development"
                elif "Optional" in line:
                    current_section = "optional"
                continue
            
            # Skip commented out requirements
            if line.startswith('#'):
                continue
                
            # Add requirement to current section if it has a version specifier
            if '>=' in line or '==' in line or '>' in line or '<' in line or '~' in line:
                requirements[current_section].append(line)
        
        return requirements
        
    except Exception as e:
        console.print(f"[red]❌ Error parsing requirements: {e}[/]")
        return requirements

def install_requirements(pip_cmd: str, requirements_file: Path, group_name: str, required: bool = True):
    """Install requirements from a specific group"""
    console.print(f"[cyan]📦 Installing {group_name} dependencies...[/]")
    
    requirements = parse_requirements(requirements_file)
    
    # Install the requested group
    if group_name.lower() in requirements and requirements[group_name.lower()]:
        deps = requirements[group_name.lower()]
        
        # Install dependencies one by one for better error handling
        failed_deps = []
        for dep in deps:
            result = subprocess.run(f"{pip_cmd} install '{dep}'", shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                failed_deps.append(dep)
                if required:
                    console.print(f"[red]❌ Failed to install {dep}: {result.stderr.strip()}[/]")
                else:
                    console.print(f"[yellow]⚠️ Failed to install {dep} (optional)[/]")
        
        if not failed_deps:
            console.print(f"[green]✅ {group_name} dependencies installed successfully[/]")
            return True
        elif not required:
            console.print(f"[yellow]⚠️ Some {group_name} dependencies failed (optional)[/]")
            return True
        else:
            console.print(f"[red]❌ {len(failed_deps)} {group_name} dependencies failed[/]")
            return False
    else:
        console.print(f"[yellow]⚠️ No {group_name} dependencies found[/]")
        return True

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
    console.print("\n[cyan]📋 Installing optional features...[/]")
    speech_success = install_requirements(pip_cmd, requirements_file, "speech", required=False)
    rag_success = install_requirements(pip_cmd, requirements_file, "rag", required=False)
    docs_success = install_requirements(pip_cmd, requirements_file, "documents", required=False)
    
    # Special handling for audio dependencies on macOS
    if not speech_success and "darwin" in sys.platform.lower():
        console.print("\n[yellow]💡 macOS speech setup help:[/]")
        console.print("[dim]brew install portaudio[/]")
        console.print("[dim]Then run: python3 install.py --install-speech[/]")
    
    # Make scripts executable
    for script in ["main.py", "install.py", "setup.py"]:
        script_path = current_dir / script
        if script_path.exists():
            script_path.chmod(0o755)
    
    console.print("\n[bold green]🎉 Setup complete![/]")
    
    # Show feature status
    console.print("\n[cyan]📋 Feature Status:[/]")
    console.print(f"  Core functionality: [green]✅ Ready[/]")
    console.print(f"  Speech mode: {'[green]✅ Ready[/]' if speech_success else '[yellow]⚠️ Not available[/]'}")
    console.print(f"  RAG support: {'[green]✅ Ready[/]' if rag_success else '[yellow]⚠️ Not available[/]'}")
    console.print(f"  Document processing: {'[green]✅ Ready[/]' if docs_success else '[yellow]⚠️ Not available[/]'}")
    
    console.print("\n[cyan]Next steps:[/]")
    console.print("1. [dim]python3 install.py[/] - Install the 'coder' command globally")
    console.print("2. [dim]coder --list-models[/] - Check available Ollama models")
    console.print("3. [dim]coder qwen2.5-coder:3b mycoder chat[/] - Start chatting!")
    
    if not speech_success or not rag_success or not docs_success:
        console.print("\n[cyan]Optional features:[/]")
        if not speech_success:
            console.print("  [dim]python3 install.py --install-speech[/] - Retry speech installation")
        if not rag_success:
            console.print("  [dim]python3 install.py --install-rag[/] - Retry RAG installation")
        if not docs_success:
            console.print("  [dim]python3 install.py --install-docs[/] - Retry document processing")
    
    console.print("\n[yellow]💡 Make sure Ollama is running with some models installed[/]")

if __name__ == "__main__":
    main()