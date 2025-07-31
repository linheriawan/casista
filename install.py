#!/usr/bin/env python3
"""
Installation script for the AI Assistant system
Creates global command 'coder' accessible from anywhere
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess
from rich.console import Console

console = Console()

def create_executable_script():
    """Create the main executable script"""
    current_dir = Path(__file__).parent.absolute()
    
    # Create the executable script content
    script_content = f'''#!/usr/bin/env python3
import sys
import os

# Add the casista directory to Python path
sys.path.insert(0, "{current_dir}")
os.chdir("{current_dir}")

# Activate virtual environment
import subprocess
venv_python = "{current_dir}/venv/bin/python"

# Run the main script with the virtual environment Python
import sys
subprocess.run([venv_python, "{current_dir}/main.py"] + sys.argv[1:])
'''
    
    return script_content

def install_to_local_bin():
    """Install to ~/.local/bin (user-local installation)"""
    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    
    script_path = local_bin / "coder"
    script_content = create_executable_script()
    
    try:
        script_path.write_text(script_content)
        script_path.chmod(0o755)  # Make executable
        
        console.print(f"[green]✅ Installed to {script_path}[/]")
        
        # Check if ~/.local/bin is in PATH
        path_dirs = os.environ.get('PATH', '').split(':')
        if str(local_bin) not in path_dirs:
            console.print(f"[yellow]⚠️ Add {local_bin} to your PATH:[/]")
            console.print(f"[dim]echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc[/]")
            console.print(f"[dim]echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.zshrc[/]")
            console.print(f"[dim]source ~/.bashrc  # or ~/.zshrc[/]")
        
        return True
    except Exception as e:
        console.print(f"[red]❌ Failed to install to ~/.local/bin: {e}[/]")
        return False

def install_to_usr_local_bin():
    """Install to /usr/local/bin (system-wide installation)"""
    usr_local_bin = Path("/usr/local/bin")
    script_path = usr_local_bin / "coder"
    script_content = create_executable_script()
    
    try:
        # This requires sudo
        console.print("[yellow]🔒 Installing to /usr/local/bin requires sudo privileges[/]")
        
        # Write to temp file first
        temp_script = Path("/tmp/coder_install")
        temp_script.write_text(script_content)
        temp_script.chmod(0o755)
        
        # Move with sudo
        result = subprocess.run(['sudo', 'mv', str(temp_script), str(script_path)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print(f"[green]✅ Installed to {script_path}[/]")
            return True
        else:
            console.print(f"[red]❌ Failed: {result.stderr}[/]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Failed to install to /usr/local/bin: {e}[/]")
        return False

def create_symlink():
    """Create symlink to the main script"""
    current_dir = Path(__file__).parent.absolute()
    main_script = current_dir / "main.py"
    
    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    
    symlink_path = local_bin / "coder"
    
    try:
        if symlink_path.exists():
            symlink_path.unlink()
        
        # Create wrapper script instead of symlink for better compatibility
        wrapper_content = f'''#!/bin/bash
cd "{current_dir}"
"{current_dir}/venv/bin/python" "{main_script}" "$@"
'''
        
        symlink_path.write_text(wrapper_content)
        symlink_path.chmod(0o755)
        
        console.print(f"[green]✅ Created wrapper script at {symlink_path}[/]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create wrapper: {e}[/]")
        return False

def uninstall():
    """Remove installed coder command"""
    locations = [
        Path.home() / ".local" / "bin" / "coder",
        Path("/usr/local/bin/coder")
    ]
    
    removed = False
    for location in locations:
        if location.exists():
            try:
                if location.parent == Path("/usr/local/bin"):
                    subprocess.run(['sudo', 'rm', str(location)], check=True)
                else:
                    location.unlink()
                console.print(f"[green]✅ Removed {location}[/]")
                removed = True
            except Exception as e:
                console.print(f"[red]❌ Failed to remove {location}: {e}[/]")
    
    if not removed:
        console.print("[yellow]ℹ️ No installation found to remove[/]")

def check_dependencies():
    """Check if all dependencies are installed"""
    current_dir = Path(__file__).parent.absolute()
    venv_python = current_dir / "venv" / "bin" / "python"
    requirements_file = current_dir / "requirements.txt"
    
    if not venv_python.exists():
        console.print("[red]❌ Virtual environment not found. Run setup first:[/]")
        console.print("[dim]python3 setup.py[/]")
        return False
    
    if not requirements_file.exists():
        console.print("[red]❌ requirements.txt not found[/]")
        return False
    
    # Test core imports
    try:
        result = subprocess.run([str(venv_python), "-c", 
                               "import ollama, rich, typer, prompt_toolkit; print('Core dependencies OK')"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            console.print("[red]❌ Missing core dependencies in virtual environment[/]")
            console.print("[dim]Run: python3 setup.py[/]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Error checking core dependencies: {e}[/]")
        return False
    
    # Check optional dependencies
    optional_deps = {
        "speech": ["speechrecognition", "pyttsx3", "pyaudio"],
        "rag": ["sentence_transformers", "numpy", "faiss"],
        "documents": ["docx", "PyPDF2"]
    }
    
    for category, imports in optional_deps.items():
        try:
            import_str = ", ".join(imports)
            result = subprocess.run([str(venv_python), "-c", 
                                   f"import {import_str}; print('{category.title()} dependencies OK')"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                console.print(f"[green]✅ {category.title()} dependencies available[/]")
            else:
                console.print(f"[yellow]⚠️ {category.title()} dependencies not available[/]")
        except Exception:
            console.print(f"[yellow]⚠️ {category.title()} dependencies not available[/]")
    
    return True

def install_optional_dependencies(category: str):
    """Install optional dependencies for a specific category"""
    current_dir = Path(__file__).parent.absolute()
    venv_python = current_dir / "venv" / "bin" / "python"
    pip_cmd = f"{current_dir}/venv/bin/pip"
    requirements_file = current_dir / "requirements.txt"
    
    if not venv_python.exists():
        console.print("[red]❌ Virtual environment not found. Run setup first[/]")
        return False
    
    if not requirements_file.exists():
        console.print("[red]❌ requirements.txt not found[/]")
        return False
    
    # Parse requirements.txt for the specific category
    try:
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        current_section = "core"
        requirements = {"speech": [], "rag": [], "documents": [], "development": []}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                if "Speech Dependencies" in line:
                    current_section = "speech"
                elif "RAG Dependencies" in line:
                    current_section = "rag"
                elif "Document Processing" in line:
                    current_section = "documents"
                elif "Development" in line:
                    current_section = "development"
                continue
            
            if not line.startswith('#') and '>' in line:
                if current_section in requirements:
                    requirements[current_section].append(line)
        
        # Install the requested category
        if category.lower() in requirements and requirements[category.lower()]:
            deps = ' '.join(requirements[category.lower()])
            console.print(f"[cyan]📦 Installing {category} dependencies...[/]")
            
            result = subprocess.run(f"{pip_cmd} install {deps}", shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[green]✅ {category} dependencies installed successfully[/]")
                return True
            else:
                console.print(f"[red]❌ Failed to install {category} dependencies: {result.stderr}[/]")
                if category == "speech" and "darwin" in sys.platform.lower():
                    console.print("[yellow]💡 On macOS, try: brew install portaudio[/]")
                return False
        else:
            console.print(f"[yellow]⚠️ No {category} dependencies found in requirements.txt[/]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Error installing {category} dependencies: {e}[/]")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Install AI Assistant system")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall the coder command")
    parser.add_argument("--system", action="store_true", help="Install system-wide (requires sudo)")
    parser.add_argument("--check", action="store_true", help="Check installation and dependencies")
    parser.add_argument("--install-speech", action="store_true", help="Install speech dependencies")
    parser.add_argument("--install-rag", action="store_true", help="Install RAG dependencies")
    parser.add_argument("--install-docs", action="store_true", help="Install document processing dependencies")
    parser.add_argument("--install-all", action="store_true", help="Install all optional dependencies")
    
    args = parser.parse_args()
    
    if args.uninstall:
        uninstall()
        return
    
    if args.check:
        console.print("[cyan]🔍 Checking installation...[/]")
        if check_dependencies():
            console.print("[green]✅ Dependencies checked[/]")
        else:
            console.print("[red]❌ Some dependencies missing[/]")
        
        # Check if coder command exists
        if shutil.which("coder"):
            console.print("[green]✅ 'coder' command is available[/]")
        else:
            console.print("[yellow]⚠️ 'coder' command not found in PATH[/]")
        return
    
    # Handle optional dependency installation
    if args.install_speech:
        install_optional_dependencies("speech")
        return
    
    if args.install_rag:
        install_optional_dependencies("rag")
        return
    
    if args.install_docs:
        install_optional_dependencies("documents")
        return
    
    if args.install_all:
        console.print("[cyan]📦 Installing all optional dependencies...[/]")
        install_optional_dependencies("speech")
        install_optional_dependencies("rag")
        install_optional_dependencies("documents")
        return
    
    console.print("[bold cyan]🚀 Installing AI Assistant System[/]")
    
    # Check dependencies first
    if not check_dependencies():
        return
    
    # Install
    if args.system:
        success = install_to_usr_local_bin()
    else:
        success = create_symlink()
    
    if success:
        console.print("\n[bold green]🎉 Installation complete![/]")
        console.print("\n[cyan]Usage examples:[/]")
        console.print("  [dim]coder qwen2.5-coder:3b mycoder chat[/]")
        console.print("  [dim]coder llama3.2:3b advisor chat --query 'Help me with Python'[/]")
        console.print("  [dim]coder --list-models[/]")
        console.print("\n[cyan]Optional features:[/]")
        console.print("  [dim]python3 install.py --install-speech[/] - Enable voice mode")
        console.print("  [dim]python3 install.py --install-rag[/] - Enable document RAG")
        console.print("  [dim]python3 install.py --install-all[/] - Install all optional features")
        console.print("  [dim]python3 install.py --check[/] - Check what's installed")
        console.print("\n[dim]Try: coder --help[/]")

if __name__ == "__main__":
    main()