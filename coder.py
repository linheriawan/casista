#!/usr/bin/env python3
"""
🤖 Qwen2.5-Coder Assistant
- Uses Ollama + qwen2.5-coder:3b
- File operations via code blocks: create, update, mkdir, delete, read
- Saves context in ./asst/
- Uses current directory as root
- Supports interactive mode and one-shot queries
"""

import os
import json
import typer
from pathlib import Path
from typing import List, Optional
import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

# Try to use prompt_toolkit for better input
try:
    from prompt_toolkit import prompt as ptk_prompt
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False

app = typer.Typer()
console = Console()

# ========================================
# Paths
# ========================================
WORKING_DIR = Path(".").resolve()
CONTEXT_DIR = Path("./asst")
CONTEXT_FILE = CONTEXT_DIR / "context.json"
HISTORY_FILE = CONTEXT_DIR / "history.txt"

# Create required paths
CONTEXT_DIR.mkdir(exist_ok=True)
CONTEXT_FILE.touch(exist_ok=True)
HISTORY_FILE.touch(exist_ok=True)

# Prompt style
prompt_style = Style.from_dict({'prompt': 'ansigreen bold'}) if HAS_PROMPT_TOOLKIT else None

# ========================================
# Load/Save Context
# ========================================
def ensure_context() -> List[dict]:
    if CONTEXT_FILE.exists():
        try:
            return json.loads(CONTEXT_FILE.read_text(encoding='utf-8')).get("messages", [])
        except:
            return []
    return []

def save_context(messages: List[dict]):
    CONTEXT_DIR.mkdir(exist_ok=True)
    CONTEXT_FILE.write_text(json.dumps({"messages": messages}, indent=2, ensure_ascii=False), encoding='utf-8')

# ========================================
# Parse Code Blocks
# ========================================
def parse_code_blocks(text: str) -> List[tuple[str, str, str]]:
    """
    Extracts: (operation, path, code)
    Supports: create, update, mkdir, delete, read
    For mkdir/delete/read, code is ignored.
    """
    import re
    
    # Handle both literal \n and actual newlines
    text = text.replace('\\n', '\n')
    
    # More flexible pattern to handle variations
    pattern = r"```(\w+):([^\n`]+)(?:\n(.*?))?```"
    matches = re.findall(pattern, text, re.DOTALL)
    blocks = []
    
    for op, path, code in matches:
        op = op.strip().lower()
        path = path.strip()
        code = code.strip() if code else ""
        
        # Filter out example text
        if code.startswith("code here"):
            code = code.replace("code here", "").strip()
        
        if op in ['create', 'update', 'write']:
            blocks.append((op, path, code))
        elif op in ['mkdir', 'delete', 'read']:
            blocks.append((op, path, ""))
        # Ignore unknown
    return blocks

# ========================================
# Handle Read Blocks (inject real file content)
# ========================================
def handle_read_blocks(text: str) -> str:
    """Replace ```read:file``` blocks with actual file content."""
    import re
    lines = text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^```read:([^\n]+)", line.strip())
        if match:
            file_path = WORKING_DIR / match.group(1).strip()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1  # skip closing ```
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                result.append(f"```text:read:{file_path.relative_to(WORKING_DIR)}\n")
                result.append(content + "\n")
                result.append("```\n\n")
            else:
                result.append(f"[File not found: {file_path.relative_to(WORKING_DIR)}]\n")
        else:
            result.append(line)
            i += 1
    return "".join(result)

# ========================================
# Apply File Operations
# ========================================
def apply_code_changes(code_blocks: List[tuple], auto_confirm: bool = False):
    """Apply file system operations from parsed blocks."""
    for op, file_path, code in code_blocks:
        full_path = WORKING_DIR / file_path

        if op == "mkdir":
            full_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[blue]📁 Created directory: {full_path.relative_to(WORKING_DIR)}[/]")
            continue

        if op == "delete":
            if full_path.exists():
                if auto_confirm or typer.confirm(f"🗑️ Delete {full_path.relative_to(WORKING_DIR)}?", default=False):
                    if full_path.is_file():
                        full_path.unlink()
                        console.print(f"[red]✅ Deleted file[/]")
                    elif full_path.is_dir():
                        import shutil
                        shutil.rmtree(full_path)
                        console.print(f"[red]✅ Deleted directory[/]")
                else:
                    console.print("[yellow]Skipped deletion.[/]")
            else:
                console.print(f"[dim]Already deleted or not found: {full_path}[/]")
            continue

        if op in ['create', 'update']:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            action = "CREATE" if op == "create" else "UPDATE"

            # Avoid rewriting same content
            if full_path.exists():
                old_content = full_path.read_text(encoding='utf-8')
                if old_content.strip() == code.strip():
                    console.print(f"[green]✅ No changes: {full_path.relative_to(WORKING_DIR)}[/]")
                    continue

            panel_title = f"[yellow]{action}[/] {full_path.relative_to(WORKING_DIR)}"
            console.print(Panel(f"📝 {action} requested", subtitle=panel_title, border_style="yellow"))

            if full_path.exists():
                console.print("[bold]Old content:[/]")
                console.print(Syntax(full_path.read_text(), "python", theme="dracula"))
                console.print("[bold]New content:[/]")
            else:
                console.print("[bold]New file:[/]")

            console.print(Syntax(code, "python", theme="dracula"))

            confirm = auto_confirm or typer.confirm("Apply this change?", default=True)
            if confirm:
                full_path.write_text(code, encoding='utf-8')
                console.print(f"[green]✅ Saved to {full_path.relative_to(WORKING_DIR)}[/]\n")
            else:
                console.print("[red]❌ Skipped.[/]\n")

# ========================================
# Stream Response from Ollama
# ========================================
def stream_response(prompt: str, messages: List[dict]) -> str:
    """Stream response from Ollama with clean UI."""
    full_response = ""
    system_prompt = (
        f"You are a coding assistant. Working directory: {WORKING_DIR}\n\n"
        f"File operations format:\n"
        f"- Create file: ```create:filename.ext\nactual content here\n```\n"
        f"- Update file: ```update:filename.ext\nfull new content\n```\n"
        f"- Read file: ```read:filename.ext\n```\n"
        f"- Delete file: ```delete:filename.ext\n```\n"
        f"- Make directory: ```mkdir:dirname\n```\n\n"
        f"Always provide real, complete file content. Be concise."
    )
    
    # Handle read blocks in user prompt first
    processed_prompt = handle_read_blocks(prompt)
    
    # Prepare messages for Ollama
    messages_with_system = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": processed_prompt}]
    
    try:
        # Show header with rich formatting but switch to plain print for streaming
        console.print("\n[bold cyan]🤖 Qwen2.5-Coder:[/]")
        
        # Stream response from Ollama with options for better control
        stream = ollama.chat(
            model='qwen2.5-coder:3b',
            messages=messages_with_system,
            stream=True,
            options={
                'temperature': 0.1,      # Lower temperature for more focused responses
                'top_p': 0.9,           # Nucleus sampling
                'num_predict': 2048,    # Limit output length
                'stop': ['</assistant>'] # Stop tokens to prevent runaway generation
            }
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                # Use plain print to avoid rich console interference with streaming
                print(content, end='', flush=True)
                full_response += content
                
            # Safety check to prevent extremely long responses
            if len(full_response) > 10000:
                print("\n⚠️ Response truncated (too long)")
                break
        
        print("\n")  # Add final newline
            
        return full_response
        
    except Exception as e:
        console.print(f"[red]❌ Error connecting to Ollama: {e}[/]")
        console.print("[yellow]💡 Make sure Ollama is running and qwen2.5-coder:3b model is installed[/]")
        return ""

# ========================================
# Main Functions
# ========================================
@app.command()
def main(
    query: Optional[str] = typer.Argument(None, help="Direct query to the AI"),
    auto_confirm: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm all file operations"),
    reset: bool = typer.Option(False, "--reset", help="Reset conversation context"),
    apply_last: bool = typer.Option(False, "--apply-last", help="Apply code blocks from last response")
):
    """🤖 Qwen2.5-Coder Assistant - Local AI coding help via Ollama"""
    
    if reset:
        CONTEXT_FILE.write_text('{"messages": []}')
        console.print("[green]✅ Context reset![/]")
        return
    
    messages = ensure_context()
    
    if apply_last:
        # Apply code blocks from the last assistant response
        if messages and messages[-1].get("role") == "assistant":
            last_response = messages[-1]["content"]
            code_blocks = parse_code_blocks(last_response)
            if code_blocks:
                console.print(f"[yellow]📋 Found {len(code_blocks)} file operation(s) in last response[/]\n")
                apply_code_changes(code_blocks, auto_confirm)
            else:
                console.print("[yellow]⚠️ No code blocks found in last response[/]")
        else:
            console.print("[yellow]⚠️ No previous assistant response found[/]")
        return
    
    if query:
        # One-shot mode
        response = stream_response(query, messages)
        if response:
            # Parse and apply code blocks
            code_blocks = parse_code_blocks(response)
            if code_blocks:
                console.print(f"\n[yellow]📋 Found {len(code_blocks)} file operation(s)[/]\n")
                apply_code_changes(code_blocks, auto_confirm)
            
            # Save to context
            messages.append({"role": "user", "content": query})
            messages.append({"role": "assistant", "content": response})
            save_context(messages)
    else:
        # Interactive mode
        console.print("[bold cyan]🤖 Qwen2.5-Coder Assistant[/]")
        console.print(f"Working directory: [dim]{WORKING_DIR}[/]")
        console.print("Type 'exit' to quit, 'reset' to clear context\n")
        
        while True:
            try:
                if HAS_PROMPT_TOOLKIT:
                    user_input = ptk_prompt("💬 ", 
                                          history=FileHistory(str(HISTORY_FILE)),
                                          style=prompt_style)
                else:
                    user_input = input("💬 ")
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("[dim]Goodbye! 👋[/]")
                    break
                
                if user_input.lower() == 'reset':
                    messages = []
                    save_context(messages)
                    console.print("[green]✅ Context reset![/]\n")
                    continue
                
                if not user_input.strip():
                    continue
                
                # Get AI response
                response = stream_response(user_input, messages)
                if response:
                    # Parse and apply code blocks
                    code_blocks = parse_code_blocks(response)
                    if code_blocks:
                        console.print(f"\n[yellow]📋 Found {len(code_blocks)} file operation(s)[/]\n")
                        apply_code_changes(code_blocks, auto_confirm)
                    
                    # Save to context
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": response})
                    save_context(messages)
                
                console.print()  # Add spacing
                
            except KeyboardInterrupt:
                console.print("\n[dim]Goodbye! 👋[/]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/]")

if __name__ == "__main__":
    app()
