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
    
    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 2048):
        """Initialize Ollama client."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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
    
    def generate_response(self, messages: List[Dict], stream: bool = True) -> str:
        """Generate response from the model."""
        try:
            if stream:
                return self._generate_streaming(messages)
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
    
    def _generate_streaming(self, messages: List[Dict]) -> str:
        """Generate streaming response with real-time display."""
        response_text = ""
        
        try:
            with Live(console=console, refresh_per_second=10) as live:
                live.update(Panel("🤖 Thinking...", title=f"{self.model}"))
                
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
                        
                        # Update live display with current response
                        live.update(Panel(
                            response_text + "▊",  # Add cursor
                            title=f"🤖 {self.model}",
                            title_align="left"
                        ))
                
                # Final update without cursor
                live.update(Panel(
                    response_text,
                    title=f"🤖 {self.model}",
                    title_align="left"
                ))
        
        except KeyboardInterrupt:
            console.print("[yellow]⚠️ Response generation interrupted[/]")
            return response_text if response_text else "Response interrupted"
        
        except Exception as e:
            console.print(f"[red]❌ Streaming error: {e}[/]")
            return response_text if response_text else f"Error: {e}"
        
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