"""
Chat management functionality.

Combines context management and Ollama client for complete chat handling.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from rich.console import Console

from .context import ContextManager
from .ollama_client import OllamaClient

console = Console()


class ChatManager:
    """Manages complete chat functionality including context and AI responses."""
    
    def __init__(self, session_config: dict):
        """Initialize chat manager with complete session configuration."""
        self.config = session_config
        
        # Extract key configuration values
        self.assistant_name = session_config["assistant"]["name"]
        self.model = session_config["assistant"]["model"]
        self.temperature = session_config.get("personality", {}).get("temperature", 0.1)
        self.max_tokens = session_config.get("advanced", {}).get("context_window", 2048)
        
        # Get parsing and streaming configuration
        parsing_config = session_config.get("parsing", {})
        streaming_config = session_config.get("streaming", {})
        
        # Get context directory
        context_dir = Path(session_config["paths"]["context_dir"])
        
        # Initialize components with parsing and streaming configuration
        self.context_manager = ContextManager(context_dir, self.assistant_name, parsing_config)
        self.ollama_client = OllamaClient(self.model, self.temperature, self.max_tokens, streaming_config)
        
        # Store system prompt for efficient reuse
        self._system_prompt = session_config.get("personality", {}).get("system_prompt", "")
        self._system_prompt_set = False
        
        # Check model availability
        if not self.ollama_client.check_model_availability():
            console.print(f"[yellow]⚠️ Model {self.model} not found. Attempting to pull...[/]")
            if not self.ollama_client.pull_model():
                raise RuntimeError(f"Failed to pull model: {self.model}")
    
    def send_message(self, user_input: str, system_prompt: str = None, 
                    stream: bool = True, callback=None) -> Dict[str, Any]:
        """Send a message and get AI response with parsed content.
        
        System prompt is automatically managed:
        - Set once at conversation start (not displayed to user)
        - Provides persistent role/context to model
        - Conversation history loaded from context.json for continuity
        """
        # Use provided system prompt or stored one
        if system_prompt is None:
            system_prompt = self._system_prompt
        
        # Ensure system prompt is set (establishes model context)
        messages = self.context_manager.ensure_system_prompt(system_prompt)
        
        # Add user message to context
        messages = self.context_manager.add_message("user", user_input, messages)
        
        # Validate messages (system + conversation history)
        if not self.ollama_client.validate_message_format(messages):
            console.print("[red]❌ Invalid message format[/]")
            return {"content": "Error: Invalid message format", "error": True}
        
        # Generate response (model sees system prompt + conversation history)
        try:
            response = self.ollama_client.generate_response(messages, stream=stream, callback=callback)
            
            # Add assistant response to context (with parsing)
            updated_messages = self.context_manager.add_message("assistant", response, messages)
            
            # Get the parsed message (last message in context)
            parsed_message = updated_messages[-1]
            
            # Return structured response
            return {
                "content": response,
                "parsed_message": parsed_message,
                "clean_answer": self.context_manager.get_display_content(parsed_message, include_reasoning=False),
                "reasoning": self.context_manager.get_reasoning_content(parsed_message),
                "has_reasoning": self.context_manager.has_reasoning(parsed_message),
                "error": False
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            console.print(f"[red]❌ {error_msg}[/]")
            return {"content": error_msg, "error": True}
    
    def get_conversation_history(self, limit: int = None) -> List[Dict]:
        """Get conversation history."""
        messages = self.context_manager.load_context()
        
        if limit:
            return messages[-limit:]
        return messages
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.context_manager.clear_context()
    
    def search_conversation(self, query: str, limit: int = 10) -> List[Dict]:
        """Search conversation history."""
        return self.context_manager.search_context(query, limit)
    
    def get_conversation_summary(self) -> Dict:
        """Get conversation summary statistics."""
        return self.context_manager.get_context_summary()
    
    def export_conversation(self, format: str = "json") -> str:
        """Export conversation in various formats."""
        return self.context_manager.export_context(format)
    
    def backup_conversation(self, backup_dir: Path = None) -> Path:
        """Backup conversation history."""
        return self.context_manager.backup_context(backup_dir)
    
    def restore_conversation(self, backup_file: Path) -> bool:
        """Restore conversation from backup."""
        return self.context_manager.restore_context(backup_file)
    
    def trim_conversation(self, max_messages: int = 100) -> List[Dict]:
        """Trim conversation to recent messages."""
        return self.context_manager.trim_context(max_messages)
    
    def estimate_context_tokens(self) -> int:
        """Estimate total tokens in current context."""
        messages = self.context_manager.load_context()
        total_text = " ".join([msg["content"] for msg in messages])
        return self.ollama_client.estimate_tokens(total_text)
    
    def set_model_parameters(self, temperature: float = None, 
                           max_tokens: int = None):
        """Update model parameters."""
        if temperature is not None:
            self.temperature = temperature
            self.ollama_client.temperature = temperature
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.ollama_client.max_tokens = max_tokens
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current model."""
        return self.ollama_client.get_model_info()
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        return self.ollama_client.test_connection()
    
    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model."""
        try:
            # Create new client with new model
            new_client = OllamaClient(new_model, self.temperature, self.max_tokens)
            
            # Test the new model
            if not new_client.check_model_availability():
                console.print(f"[yellow]⚠️ Model {new_model} not found. Attempting to pull...[/]")
                if not new_client.pull_model():
                    console.print(f"[red]❌ Failed to pull model: {new_model}[/]")
                    return False
            
            # Switch to new model
            self.model = new_model
            self.ollama_client = new_client
            
            console.print(f"[green]✅ Switched to model: {new_model}[/]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error switching model: {e}[/]")
            return False
    
    def add_system_message(self, content: str):
        """Add a system message to context."""
        messages = self.context_manager.load_context()
        self.context_manager.add_message("system", content, messages)
    
    def get_last_response(self) -> Optional[str]:
        """Get the last assistant response."""
        messages = self.context_manager.load_context()
        
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                return msg["content"]
        
        return None
    
    def get_system_prompt(self) -> str:
        """Get the system prompt from configuration."""
        return self.config.get("personality", {}).get("system_prompt", "")
    
    def get_user_name(self) -> str:
        """Get the user name from configuration."""
        return self.config.get("assistant", {}).get("user_name", "user")
    
    def retry_last_response(self, system_prompt: str = None) -> str:
        """Retry generating the last response."""
        messages = self.context_manager.load_context()
        
        # Remove last assistant message if it exists
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()
            self.context_manager.save_context(messages)
        
        # Find last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if not last_user_msg:
            return "No user message found to retry"
        
        # Generate new response
        model_messages = self.ollama_client.prepare_messages(system_prompt, messages)
        response = self.ollama_client.generate_response(model_messages, stream=True)
        
        # Add new response to context
        self.context_manager.add_message("assistant", response, messages)
        
        return response