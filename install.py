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
    
    if not venv_python.exists():
        console.print("[red]❌ Virtual environment not found. Run setup first:[/]")
        console.print("[dim]python3 -m venv venv[/]")
        console.print("[dim]source venv/bin/activate[/]")
        console.print("[dim]pip install ollama rich typer prompt_toolkit[/]")
        return False
    
    # Test imports
    try:
        result = subprocess.run([str(venv_python), "-c", 
                               "import ollama, rich, typer; print('Dependencies OK')"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            console.print("[red]❌ Missing dependencies in virtual environment[/]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Error checking dependencies: {e}[/]")
        return False
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Install AI Assistant system")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall the coder command")
    parser.add_argument("--system", action="store_true", help="Install system-wide (requires sudo)")
    parser.add_argument("--check", action="store_true", help="Check installation and dependencies")
    
    args = parser.parse_args()
    
    if args.uninstall:
        uninstall()
        return
    
    if args.check:
        console.print("[cyan]🔍 Checking installation...[/]")
        if check_dependencies():
            console.print("[green]✅ Dependencies OK[/]")
        else:
            console.print("[red]❌ Dependencies missing[/]")
        
        # Check if coder command exists
        if shutil.which("coder"):
            console.print("[green]✅ 'coder' command is available[/]")
        else:
            console.print("[yellow]⚠️ 'coder' command not found in PATH[/]")
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
        console.print("\n[dim]Try: coder --help[/]")

if __name__ == "__main__":
    main()