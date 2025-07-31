"""
File operations module extracted from coder.py.

Handles file system operations like create, update, delete, mkdir, and read.
"""

import re
import json
from pathlib import Path
from typing import List, Tuple, Optional
from rich.console import Console

console = Console()


class FileOperations:
    """Handles file operations for code generation and analysis."""
    
    def __init__(self, working_dir: Path = None):
        """Initialize with working directory."""
        self.working_dir = working_dir or Path.cwd()
    
    def parse_code_blocks(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extracts: (operation, path, code)
        Supports: create, update, mkdir, delete, read
        For mkdir/delete/read, code is ignored.
        """
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
            # Ignore unknown operations
        
        return blocks
    
    def handle_read_blocks(self, text: str) -> str:
        """Replace ```read:file``` blocks with actual file content."""
        lines = text.splitlines(keepends=True)
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            match = re.match(r"^```read:([^\n]+)", line.strip())
            
            if match:
                file_path = self.working_dir / match.group(1).strip()
                i += 1
                # Skip to closing ```
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    i += 1
                i += 1  # skip closing ```
                
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        result.append(f"```text:read:{file_path.relative_to(self.working_dir)}\n")
                        result.append(content + "\n")
                        result.append("```\n\n")
                    except Exception as e:
                        result.append(f"[Error reading file: {e}]\n")
                else:
                    result.append(f"[File not found: {file_path.relative_to(self.working_dir)}]\n")
            else:
                result.append(line)
                i += 1
        
        return "".join(result)
    
    def apply_code_changes(self, code_blocks: List[Tuple[str, str, str]], 
                          auto_confirm: bool = False) -> bool:
        """Apply file system operations from parsed blocks."""
        from rich.prompt import Confirm
        
        for op, file_path, code in code_blocks:
            full_path = self.working_dir / file_path
            
            try:
                if op == "mkdir":
                    full_path.mkdir(parents=True, exist_ok=True)
                    console.print(f"[blue]📁 Created directory: {full_path.relative_to(self.working_dir)}[/]")
                    continue
                
                if op == "delete":
                    if full_path.exists():
                        if auto_confirm or Confirm.ask(f"🗑️ Delete {full_path.relative_to(self.working_dir)}?", default=False):
                            if full_path.is_file():
                                full_path.unlink()
                                console.print(f"[red]✅ Deleted file: {full_path.relative_to(self.working_dir)}[/]")
                            elif full_path.is_dir():
                                import shutil
                                shutil.rmtree(full_path)
                                console.print(f"[red]✅ Deleted directory: {full_path.relative_to(self.working_dir)}[/]")
                        else:
                            console.print(f"[yellow]⏭️ Skipped deletion[/]")
                    else:
                        console.print(f"[yellow]⚠️ Path not found: {full_path.relative_to(self.working_dir)}[/]")
                    continue
                
                if op in ["create", "update", "write"]:
                    if not code.strip():
                        console.print(f"[yellow]⚠️ Empty content for {op}: {file_path}[/]")
                        continue
                    
                    # Ensure parent directory exists
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Check if file exists for create vs update
                    file_exists = full_path.exists()
                    
                    if op == "create" and file_exists:
                        if not auto_confirm and not Confirm.ask(f"📝 File exists. Overwrite {full_path.relative_to(self.working_dir)}?", default=False):
                            console.print(f"[yellow]⏭️ Skipped overwrite[/]")
                            continue
                    
                    # Write the file
                    full_path.write_text(code, encoding='utf-8')
                    
                    # Show appropriate message
                    if file_exists:
                        console.print(f"[green]✅ Updated: {full_path.relative_to(self.working_dir)}[/]")
                    else:
                        console.print(f"[green]✅ Created: {full_path.relative_to(self.working_dir)}[/]")
            
            except Exception as e:
                console.print(f"[red]❌ Error with {op} {file_path}: {e}[/]")
                return False
        
        return True
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read file content safely."""
        full_path = self.working_dir / file_path
        
        if not full_path.exists():
            return None
        
        try:
            return full_path.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"[red]❌ Error reading {file_path}: {e}[/]")
            return None
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists."""
        full_path = self.working_dir / file_path
        return full_path.exists()
    
    def list_files(self, pattern: str = "*") -> List[str]:
        """List files matching pattern."""
        try:
            files = []
            for path in self.working_dir.rglob(pattern):
                if path.is_file():
                    files.append(str(path.relative_to(self.working_dir)))
            return sorted(files)
        except Exception as e:
            console.print(f"[red]❌ Error listing files: {e}[/]")
            return []