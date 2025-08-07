"""
Ollama client for AI model communication.

Handles communication with Ollama models and streaming responses.
"""

import ollama
from typing import List, Dict, Optional, Iterator, Any
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class OllamaClient:
    """Client for communicating with Ollama models."""
    
    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 2048, 
                 streaming_config: dict = None):
        """Initialize Ollama client."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.streaming_config = streaming_config or {}
        self.client = ollama
    
    def check_model_availability(self) -> bool:
        """Check if the model is available."""
        try:
            models = self.client.list()
            available_models = [m.model for m in models.get('models', [])]
            return self.model in available_models
        except Exception as e:
            console.print(f"[red]❌ Error checking model availability: {e}[/]")
            return False
    
    def pull_model(self) -> bool:
        """Pull/download the model if not available."""
        try:
            console.print(f"[cyan]📥 Pulling model: {self.model}[/]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}")
            ) as progress:
                task = progress.add_task("Downloading model...", total=None)
                
                # Pull the model
                self.client.pull(self.model)
                progress.update(task, description="Model downloaded successfully!")
            
            console.print(f"[green]✅ Model {self.model} is ready[/]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error pulling model: {e}[/]")
            return False
    
    def generate_response(self, messages: List[Dict], stream: bool = True, callback=None) -> str:
        """Generate response from the model.
        
        Messages should already include system prompt and conversation history.
        Model will see: [system_prompt, user1, assistant1, user2, assistant2, ...]
        """
        try:
            if stream:
                return self._generate_streaming(messages, callback)
            else:
                return self._generate_simple(messages)
        
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            console.print(f"[red]❌ {error_msg}[/]")
            return error_msg
    
    def _generate_simple(self, messages: List[Dict]) -> str:
        """Generate simple non-streaming response."""
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        )
        
        return response.get('message', {}).get('content', 'No response generated')
    
    def _generate_streaming(self, messages: List[Dict], callback=None) -> str:
        """Generate streaming response with callback support for fullscreen display."""
        response_text = ""
        
        try:
            # Signal start of streaming
            if callback:
                callback("start", "")
            
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    response_text += content
                    
                    # Send chunk to callback for fullscreen display
                    if callback:
                        callback("chunk", response_text)
            
            # Signal completion
            if callback:
                callback("complete", response_text)
        
        except KeyboardInterrupt:
            if callback:
                callback("interrupted", response_text if response_text else "Response interrupted")
            return response_text if response_text else "Response interrupted"
        
        except Exception as e:
            error_msg = f"Streaming error: {e}"
            if callback:
                callback("error", error_msg)
            return response_text if response_text else error_msg
        
        return response_text
    
    def generate_embeddings(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for text (if model supports it)."""
        try:
            response = self.client.embeddings(
                model=self.model,
                prompt=text
            )
            return response.get('embedding')
        except Exception as e:
            console.print(f"[yellow]⚠️ Embeddings not supported by {self.model}: {e}[/]")
            return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        try:
            models = self.client.list()
            return models.get('models', [])
        except Exception as e:
            console.print(f"[red]❌ Error listing models: {e}[/]")
            return []
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current model."""
        try:
            models = self.list_available_models()
            for model in models:
                if model.get('name') == self.model or model.get('model') == self.model:
                    return model
            return None
        except Exception as e:
            console.print(f"[red]❌ Error getting model info: {e}[/]")
            return None
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            # Try to list models as a connection test
            self.client.list()
            return True
        except Exception as e:
            console.print(f"[red]❌ Ollama connection failed: {e}[/]")
            console.print("[yellow]💡 Make sure Ollama is running: `ollama serve`[/]")
            return False
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def prepare_messages(self, system_prompt: str, messages: List[Dict]) -> List[Dict]:
        """Prepare messages with system prompt for the model."""
        formatted_messages = []
        
        # Add system message if provided
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation messages
        formatted_messages.extend(messages)
        
        return formatted_messages
    
    def validate_message_format(self, messages: List[Dict]) -> bool:
        """Validate message format for Ollama."""
        required_fields = ['role', 'content']
        valid_roles = ['system', 'user', 'assistant']
        
        for msg in messages:
            # Check required fields
            if not all(field in msg for field in required_fields):
                return False
            
            # Check valid roles
            if msg['role'] not in valid_roles:
                return False
            
            # Check content is string
            if not isinstance(msg['content'], str):
                return False
        
        return True