"""
Code generation and analysis functionality.

Handles code-related tasks and analysis.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from rich.console import Console
from rich.syntax import Syntax

console = Console()


class CodeGenerator:
    """Handles code generation and analysis tasks."""
    
    def __init__(self, working_dir: Path = None):
        """Initialize with working directory."""
        self.working_dir = working_dir or Path.cwd()
    
    def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze code structure and extract key information."""
        full_path = self.working_dir / file_path
        
        if not full_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        try:
            content = full_path.read_text(encoding='utf-8')
            suffix = full_path.suffix.lower()
            
            analysis = {
                "file_path": file_path,
                "file_type": suffix,
                "lines": len(content.splitlines()),
                "size": len(content),
                "functions": [],
                "classes": [],
                "imports": [],
                "comments": []
            }
            
            # Language-specific analysis
            if suffix == '.py':
                analysis.update(self._analyze_python(content))
            elif suffix in ['.js', '.ts']:
                analysis.update(self._analyze_javascript(content))
            elif suffix in ['.java']:
                analysis.update(self._analyze_java(content))
            elif suffix in ['.cpp', '.c', '.h']:
                analysis.update(self._analyze_cpp(content))
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error analyzing {file_path}: {e}"}
    
    def _analyze_python(self, content: str) -> Dict[str, List[str]]:
        """Analyze Python code."""
        functions = re.findall(r'def\s+(\w+)\s*\(', content)
        classes = re.findall(r'class\s+(\w+)\s*[\(:]', content)
        imports = re.findall(r'(?:from\s+\S+\s+)?import\s+([^\n]+)', content)
        comments = re.findall(r'#\s*(.+)', content)
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": [imp.strip() for imp in imports],
            "comments": [comment.strip() for comment in comments[:10]]  # First 10 comments
        }
    
    def _analyze_javascript(self, content: str) -> Dict[str, List[str]]:
        """Analyze JavaScript/TypeScript code."""
        functions = re.findall(r'function\s+(\w+)\s*\(|(\w+)\s*:\s*function|const\s+(\w+)\s*=\s*\(', content)
        classes = re.findall(r'class\s+(\w+)\s*[{]', content)
        imports = re.findall(r'import\s+[^;]+from\s+["\']([^"\']+)["\']', content)
        comments = re.findall(r'//\s*(.+)', content)
        
        # Flatten function matches
        flat_functions = [f for match in functions for f in match if f]
        
        return {
            "functions": flat_functions,
            "classes": classes,
            "imports": imports,
            "comments": [comment.strip() for comment in comments[:10]]
        }
    
    def _analyze_java(self, content: str) -> Dict[str, List[str]]:
        """Analyze Java code."""
        functions = re.findall(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\(', content)
        classes = re.findall(r'(?:public|private)?\s*class\s+(\w+)', content)
        imports = re.findall(r'import\s+([^;]+);', content)
        comments = re.findall(r'//\s*(.+)', content)
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": [imp.strip() for imp in imports],
            "comments": [comment.strip() for comment in comments[:10]]
        }
    
    def _analyze_cpp(self, content: str) -> Dict[str, List[str]]:
        """Analyze C++ code."""
        functions = re.findall(r'\w+\s+(\w+)\s*\([^)]*\)\s*[{]', content)
        classes = re.findall(r'class\s+(\w+)\s*[{:]', content)
        includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', content)
        comments = re.findall(r'//\s*(.+)', content)
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": includes,  # Using 'imports' for consistency
            "comments": [comment.strip() for comment in comments[:10]]
        }
    
    def extract_code_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract code blocks from text with language detection."""
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        code_blocks = []
        for lang, code in matches:
            code_blocks.append({
                "language": lang or "text",
                "code": code.strip()
            })
        
        return code_blocks
    
    def format_code(self, code: str, language: str = "python") -> str:
        """Format code with syntax highlighting for display."""
        try:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            return syntax
        except Exception:
            return code
    
    def validate_syntax(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Basic syntax validation for code."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            if language == "python":
                import ast
                ast.parse(code)
            elif language in ["javascript", "js"]:
                # Basic JavaScript validation (checking for common syntax errors)
                if code.count('{') != code.count('}'):
                    result["errors"].append("Mismatched curly braces")
                if code.count('(') != code.count(')'):
                    result["errors"].append("Mismatched parentheses")
            
            if result["errors"]:
                result["valid"] = False
                
        except SyntaxError as e:
            result["valid"] = False
            result["errors"].append(f"Syntax error: {e}")
        except Exception as e:
            result["warnings"].append(f"Validation warning: {e}")
        
        return result
    
    def suggest_improvements(self, code: str, language: str = "python") -> List[str]:
        """Suggest basic code improvements."""
        suggestions = []
        
        if language == "python":
            # Check for common Python issues
            if "import *" in code:
                suggestions.append("Consider avoiding 'import *' statements")
            
            if re.search(r'except\s*:', code):
                suggestions.append("Consider catching specific exceptions instead of bare 'except:'")
            
            if len(code.splitlines()) > 50 and not re.search(r'def\s+\w+|class\s+\w+', code):
                suggestions.append("Consider breaking large code blocks into functions")
            
            # Check for missing docstrings in functions
            functions = re.findall(r'def\s+(\w+)\s*\([^)]*\):\s*\n(?!\s*""")', code)
            if functions:
                suggestions.append(f"Consider adding docstrings to functions: {', '.join(functions[:3])}")
        
        return suggestions