"""
Session management for coordinating all components with proper configuration.

This module provides a centralized way to initialize and manage all components
of an assistant session, ensuring all configuration is properly applied.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

from .conversation.chat_manager import ChatManager
from .coding.file_ops import FileOperations
from .image_generation.generation import ImageGenerator

console = Console()


class SessionManager:
    """Manages an assistant session with all properly configured components."""
    
    def __init__(self, session_config: Dict[str, Any]):
        """Initialize session manager with complete configuration."""
        self.config = session_config
        self.assistant_name = session_config["assistant"]["name"]
        self.model = session_config["assistant"]["model"]
        self.working_dir = Path(session_config["session"]["working_dir"])
        
        # Initialize core components
        self.chat_manager: Optional[ChatManager] = None
        self.file_ops: Optional[FileOperations] = None
        self.image_generator: Optional[ImageGenerator] = None
        
        console.print(f"[green]🚀 Session initialized for assistant '{self.assistant_name}'[/]")
    
    def get_chat_manager(self) -> ChatManager:
        """Get or create properly configured chat manager."""
        if self.chat_manager is None:
            self.chat_manager = ChatManager(self.config)
            console.print("[dim]💬 Chat manager initialized[/]")
        return self.chat_manager
    
    def get_file_operations(self, enable: bool = True) -> Optional[FileOperations]:
        """Get or create file operations if enabled."""
        if not enable:
            return None
            
        file_ops_enabled = self.config.get("capabilities", {}).get("file_operations", True)
        if not file_ops_enabled:
            return None
            
        if self.file_ops is None:
            # Use coding configuration if available
            coding_config = self.config.get("coding", {})
            self.file_ops = FileOperations(self.working_dir)
            
            # Apply coding preferences
            if coding_config.get("auto_format", True):
                console.print("[dim]📝 Auto-formatting enabled[/]")
            if coding_config.get("include_tests", True):
                console.print("[dim]🧪 Test generation enabled[/]")
                
            console.print("[dim]📁 File operations initialized[/]")
        
        return self.file_ops
    
    def get_image_generator(self) -> Optional[ImageGenerator]:
        """Get or create image generator if enabled."""
        image_enabled = self.config.get("capabilities", {}).get("image_generation", False)
        if not image_enabled:
            return None
            
        if self.image_generator is None:
            image_config = self.config.get("image", {})
            models = image_config.get("models", ["hakurei/waifu-diffusion"])
            images_dir = self.working_dir / image_config.get("output_subdir", "./")
            
            self.image_generator = ImageGenerator(models, images_dir)
            console.print("[dim]🎨 Image generator initialized[/]")
            
        return self.image_generator
    
    def get_voice_config(self) -> Dict[str, Any]:
        """Get voice configuration for speech functionality."""
        voice_config = self.config.get("voice", {})
        return {
            "voice_id": voice_config.get("voice_id", ""),
            "voice_name": voice_config.get("voice_name", "Default"),
            "speech_backend": voice_config.get("speech_backend", "google"),
            "speech_rate": voice_config.get("speech_rate", 200),
            "noise_level": voice_config.get("noise_level", "normal")
        }
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return self.config.get("rag", {})
    
    def get_parsing_config(self) -> Dict[str, Any]:
        """Get message parsing configuration."""
        return self.config.get("parsing", {})
    
    def get_streaming_config(self) -> Dict[str, Any]:
        """Get streaming behavior configuration."""
        return self.config.get("streaming", {})
    
    def is_capability_enabled(self, capability: str) -> bool:
        """Check if a capability is enabled."""
        return self.config.get("capabilities", {}).get(capability, False)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.config.get("personality", {}).get("system_prompt", "")
    
    def get_user_name(self) -> str:
        """Get the user name."""
        return self.config.get("assistant", {}).get("user_name", "user")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration information."""
        return {
            "model": self.model,
            "temperature": self.config.get("personality", {}).get("temperature", 0.1),
            "max_tokens": self.config.get("advanced", {}).get("context_window", 2048),
            "personality": self.config.get("personality", {}).get("personality_id", "unknown")
        }
    
    def update_working_directory(self, new_dir: Path):
        """Update working directory for current session."""
        self.working_dir = new_dir.resolve()
        
        # Update file operations if initialized
        if self.file_ops is not None:
            self.file_ops.working_dir = self.working_dir
            
        console.print(f"[cyan]📁 Working directory updated: {self.working_dir}[/]")
    
    def cleanup(self):
        """Clean up session resources."""
        if self.chat_manager:
            # ChatManager doesn't need explicit cleanup currently
            pass
            
        # Reset components
        self.chat_manager = None
        self.file_ops = None
        self.image_generator = None
        
        console.print("[dim]🧹 Session cleaned up[/]")
    
    def get_mode_specific_system_prompt(self, mode: str) -> str:
        """Get system prompt customized for specific mode."""
        base_prompt = self.get_system_prompt()
        
        if mode == "chat":
            return base_prompt
        elif mode == "speech":
            return f"{base_prompt}\n\nYou are currently in voice conversation mode. Keep responses concise and conversational since they will be spoken aloud."
        elif mode == "image":
            return f"""You are an expert image generation assistant based on this personality: {base_prompt}

Your role in image mode:
1. Understand user's image requests in context of our conversation
2. Enhance and refine image prompts for better visual results
3. Provide helpful suggestions about style, composition, and technical aspects
4. Remember previous image requests to build upon them
5. Consider the user's preferences and conversation history

Always respond with enhanced prompts and brief explanations to help create better images."""
        else:
            return base_prompt
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get comprehensive session information."""
        model_info = self.get_model_info()
        
        return {
            "assistant_name": self.assistant_name,
            "model": model_info["model"],
            "temperature": model_info["temperature"],
            "max_tokens": model_info["max_tokens"],
            "personality": model_info["personality"],
            "working_dir": str(self.working_dir),
            "capabilities": {
                "file_operations": self.is_capability_enabled("file_operations"),
                "rag_enabled": self.is_capability_enabled("rag_enabled"),
                "image_generation": self.is_capability_enabled("image_generation"),
                "voice_enabled": self.is_capability_enabled("voice_enabled")
            },
            "components_initialized": {
                "chat_manager": self.chat_manager is not None,
                "file_ops": self.file_ops is not None,
                "image_generator": self.image_generator is not None
            },
            "mode_support": {
                "chat": True,
                "speech": self.is_capability_enabled("voice_enabled"),
                "image": self.is_capability_enabled("image_generation")
            }
        }