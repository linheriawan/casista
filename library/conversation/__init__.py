"""
Conversation management module.

This module handles chat management, context handling, and AI model communication.
"""

from .context import ContextManager
from .ollama_client import OllamaClient
from .chat_manager import ChatManager

__all__ = ['ContextManager', 'OllamaClient', 'ChatManager']