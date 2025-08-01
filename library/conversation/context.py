"""
Context management for conversation history.

Handles loading, saving, and managing conversation context with enhanced parsing.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from rich.console import Console

from .message_parser import MessageParser

console = Console()


class ContextManager:
    """Manages conversation context and history with enhanced message parsing."""
    
    def __init__(self, context_dir: Path, assistant_name: str, parsing_config: Dict[str, Any] = None):
        """Initialize context manager with optional parsing configuration."""
        self.context_dir = context_dir
        self.assistant_name = assistant_name
        self.context_file = context_dir / "context.json"
        
        # Initialize message parser
        self.parser = MessageParser(parsing_config or {}) if parsing_config else None
        
        # System prompt state
        self._system_prompt_set = False
        self._current_system_prompt = ""
        
        # Ensure directory exists
        self.context_dir.mkdir(parents=True, exist_ok=True)
    
    def load_context(self) -> List[Dict]:
        """Load conversation context from file.
        
        This provides conversation memory - model will see previous messages
        and can continue conversations across sessions.
        """
        if self.context_file.exists():
            try:
                data = json.loads(self.context_file.read_text(encoding='utf-8'))
                messages = data.get("messages", [])
                
                # Check if system prompt was set in previous session
                if messages and messages[0].get("role") == "system":
                    self._system_prompt_set = True
                    self._current_system_prompt = messages[0].get("content", "")
                
                return messages
            except Exception as e:
                console.print(f"[yellow]⚠️ Error loading context: {e}[/]")
                return []
        return []
    
    def save_context(self, messages: List[Dict]):
        """Save conversation context to file."""
        try:
            context_data = {
                "messages": messages,
                "assistant_name": self.assistant_name,
                "last_updated": self._get_timestamp()
            }
            
            self.context_file.write_text(
                json.dumps(context_data, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
        except Exception as e:
            console.print(f"[red]❌ Error saving context: {e}[/]")
    
    def add_message(self, role: str, content: str, messages: List[Dict] = None) -> List[Dict]:
        """Add a message to context with enhanced parsing."""
        if messages is None:
            messages = self.load_context()
        
        # Use parser if available
        if self.parser:
            message = self.parser.parse_message(content, role)
        else:
            # Fallback to simple message structure
            message = {
                "role": role,
                "content": content,
                "timestamp": self._get_timestamp()
            }
        
        messages.append(message)
        self.save_context(messages)
        return messages
    
    def clear_context(self):
        """Clear all conversation history."""
        self.save_context([])
        console.print(f"[green]✅ Context cleared for {self.assistant_name}[/]")
    
    def get_context_summary(self) -> Dict:
        """Get summary of current context."""
        messages = self.load_context()
        
        summary = {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if m["role"] == "user"]),
            "assistant_messages": len([m for m in messages if m["role"] == "assistant"]),
            "system_messages": len([m for m in messages if m["role"] == "system"]),
            "first_message": messages[0]["timestamp"] if messages else None,
            "last_message": messages[-1]["timestamp"] if messages else None
        }
        
        return summary
    
    def trim_context(self, max_messages: int = 100):
        """Trim context to keep only recent messages."""
        messages = self.load_context()
        
        if len(messages) > max_messages:
            # Keep system messages and recent messages
            system_messages = [m for m in messages if m["role"] == "system"]
            other_messages = [m for m in messages if m["role"] != "system"]
            
            # Keep the most recent messages
            recent_messages = other_messages[-max_messages:]
            
            trimmed = system_messages + recent_messages
            self.save_context(trimmed)
            
            console.print(f"[yellow]📝 Context trimmed to {len(trimmed)} messages[/]")
            return trimmed
        
        return messages
    
    def search_context(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for messages containing specific text."""
        messages = self.load_context()
        
        matching = []
        query_lower = query.lower()
        
        for msg in messages:
            if query_lower in msg["content"].lower():
                matching.append(msg)
                if len(matching) >= limit:
                    break
        
        return matching
    
    def export_context(self, format: str = "json") -> str:
        """Export context in various formats."""
        messages = self.load_context()
        
        if format == "json":
            return json.dumps(messages, indent=2, ensure_ascii=False)
        
        elif format == "text":
            lines = []
            for msg in messages:
                timestamp = msg.get("timestamp", "Unknown")
                role = msg["role"].title()
                content = msg["content"]
                lines.append(f"[{timestamp}] {role}: {content}\n")
            return "\n".join(lines)
        
        elif format == "markdown":
            lines = ["# Conversation History\n"]
            for msg in messages:
                timestamp = msg.get("timestamp", "Unknown")
                role = msg["role"].title()
                content = msg["content"]
                lines.append(f"## {role} ({timestamp})\n\n{content}\n")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_display_content(self, message: Dict[str, Any], include_reasoning: bool = True) -> str:
        """Get message content for display."""
        if self.parser:
            return self.parser.get_display_content(message, include_reasoning)
        return message.get("content", "")
    
    def get_reasoning_content(self, message: Dict[str, Any]) -> Optional[str]:
        """Get reasoning content from message."""
        if self.parser:
            return self.parser.get_reasoning_content(message)
        return None
    
    def has_reasoning(self, message: Dict[str, Any]) -> bool:
        """Check if message has reasoning content."""
        return bool(message.get("parsed_sections", {}).get("reasoning"))
    
    def ensure_system_prompt(self, system_prompt: str) -> List[Dict]:
        """Ensure system prompt is set at the beginning of conversation.
        
        This establishes the model's role/context without displaying to user.
        System prompt is sent once and persists throughout the conversation.
        """
        messages = self.load_context()
        
        # If system prompt not set or changed, add/update it
        if not self._system_prompt_set or self._current_system_prompt != system_prompt:
            # Remove old system message if exists
            if messages and messages[0].get("role") == "system":
                messages.pop(0)
            
            # Add new system message at the beginning
            system_message = {
                "role": "system",
                "content": system_prompt,
                "timestamp": self._get_timestamp()
            }
            messages.insert(0, system_message)
            
            # Update state
            self._system_prompt_set = True
            self._current_system_prompt = system_prompt
            
            # Save updated context
            self.save_context(messages)
            
            console.print(f"[dim]🎯 System context established for {self.assistant_name}[/]")
        
        return messages
    
    def get_system_prompt_status(self) -> Dict[str, Any]:
        """Get current system prompt status."""
        return {
            "is_set": self._system_prompt_set,
            "prompt": self._current_system_prompt,
            "length": len(self._current_system_prompt)
        }
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def backup_context(self, backup_dir: Path = None):
        """Create a backup of current context."""
        if backup_dir is None:
            backup_dir = self.context_dir / "backups"
        
        backup_dir.mkdir(exist_ok=True)
        timestamp = self._get_timestamp().replace(":", "-")
        backup_file = backup_dir / f"context_backup_{timestamp}.json"
        
        messages = self.load_context()
        backup_file.write_text(
            json.dumps(messages, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        
        console.print(f"[green]✅ Context backed up to: {backup_file}[/]")
        return backup_file
    
    def restore_context(self, backup_file: Path):
        """Restore context from backup."""
        if not backup_file.exists():
            console.print(f"[red]❌ Backup file not found: {backup_file}[/]")
            return False
        
        try:
            messages = json.loads(backup_file.read_text(encoding='utf-8'))
            self.save_context(messages)
            console.print(f"[green]✅ Context restored from: {backup_file}[/]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Error restoring context: {e}[/]")
            return False