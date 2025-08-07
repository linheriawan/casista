#!/usr/bin/env python3
"""
Response Renderer - Advanced rendering for AI responses with multiple display modes.

Provides unified rendering for console output, speech synthesis, and advanced layouts
including tables, streaming, progress bars, and window-style positioning.
"""

import sys
from typing import Dict, Any, Optional, Generator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.live import Live
from rich.layout import Layout


class ResponseRenderer:
    """Advanced response renderer with multiple display modes and layouts."""
    
    def __init__(self, console: Console = None, speech_handler=None):
        """Initialize response renderer.
        
        Args:
            console: Rich console instance
            speech_handler: Optional speech handler for TTS
        """
        self.console = console or Console()
        self.speech_handler = speech_handler
        self.current_line = 0
        self.layout_zones = {}  # For window-style rendering
        self.live_display = None  # For streaming updates
    
    def render_response(self, mode: str, assistant_name: str, response_data: Dict[str, Any], 
                       style: str = "chat") -> bool:
        """Render AI response with specified mode and style.
        
        Args:
            mode: "text", "speech", "text_and_speech"
            assistant_name: Name of the assistant 
            response_data: Response data from chat manager
            style: "chat", "table", "panel", "stream"
            
        Returns:
            bool: Success status
        """
        if response_data.get("error"):
            self.console.print(f"[red]❌ {response_data['content']}[/]")
            return False
        
        # Determine content for display and TTS
        if response_data.get("has_reasoning"):
            display_response = response_data['clean_answer']
            tts_response = response_data['clean_answer']
            reasoning = response_data.get('reasoning')
        else:
            display_response = response_data['content']
            tts_response = response_data['content']
            reasoning = None
        
        # Render based on style
        if style == "chat":
            self._render_chat_style(assistant_name, display_response, reasoning)
        elif style == "table":
            self._render_table_style(assistant_name, display_response, reasoning)
        elif style == "panel":
            self._render_panel_style(assistant_name, display_response, reasoning)
        else:
            # Default to chat style
            self._render_chat_style(assistant_name, display_response, reasoning)
        
        # Handle speech output
        if mode in ["speech", "text_and_speech"] and self.speech_handler and tts_response:
            return self.speech_handler.speak(tts_response)
        
        return True
    
    def _render_chat_style(self, assistant_name: str, response: str, reasoning: str = None):
        """Render in chat style with emoji and assistant name."""
        self.console.print(f"🤖 [{assistant_name}]: {response}")
        if reasoning:
            self.console.print(f"[dim]💭 Reasoning: {reasoning}[/]")
    
    def _render_table_style(self, assistant_name: str, response: str, reasoning: str = None):
        """Render response in a rich table format."""
        table = Table(title=f"🤖 {assistant_name} Response")
        table.add_column("Content", style="white", no_wrap=False)
        table.add_row(response)
        
        if reasoning:
            table.add_row(f"[dim]💭 Reasoning: {reasoning}[/]")
        
        self.console.print(table)
    
    def _render_panel_style(self, assistant_name: str, response: str, reasoning: str = None):
        """Render response in a rich panel format."""
        content = response
        if reasoning:
            content += f"\n\n[dim]💭 Reasoning: {reasoning}[/]"
        
        panel = Panel.fit(
            content,
            title=f"🤖 {assistant_name}",
            border_style="blue"
        )
        self.console.print(panel)
    
    def render_stream(self, assistant_name: str, stream_generator: Generator, 
                     show_reasoning: bool = True) -> Dict[str, Any]:
        """Render real-time streaming response with live updates.
        
        Args:
            assistant_name: Name of the assistant
            stream_generator: Generator yielding response chunks
            show_reasoning: Whether to display reasoning separately
            
        Returns:
            Dict with final response data
        """
        response_text = ""
        reasoning_text = ""
        
        with Live(console=self.console, refresh_per_second=4) as live:
            for chunk in stream_generator:
                if chunk.get('type') == 'reasoning':
                    reasoning_text += chunk.get('content', '')
                else:
                    response_text += chunk.get('content', '')
                
                # Update live display
                display_content = f"🤖 [{assistant_name}]: {response_text}"
                if reasoning_text and show_reasoning:
                    display_content += f"\n\n[dim]💭 Reasoning: {reasoning_text}[/]"
                
                live.update(display_content)
        
        return {
            'content': response_text,
            'reasoning': reasoning_text,
            'has_reasoning': bool(reasoning_text)
        }
    
    def render_progress(self, task_name: str, progress_value: float, total: float = 100.0,
                       override_line: bool = True):
        """Render progress bar with optional line override.
        
        Args:
            task_name: Name of the task
            progress_value: Current progress value
            total: Total progress value
            override_line: Whether to override current line (like htop)
        """
        if override_line:
            # Move cursor up and clear line for htop-style updates
            sys.stdout.write('\033[F\033[K')
        
        percentage = (progress_value / total) * 100
        bar_length = 40
        filled_length = int(bar_length * progress_value // total)
        
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        self.console.print(f"{task_name}: [{bar}] {percentage:.1f}%")
    
    def render_table(self, data: Dict[str, Any], title: str = None, 
                    columns: Dict[str, str] = None) -> Table:
        """Render structured data in a rich table.
        
        Args:
            data: Dictionary or list of data to display
            title: Optional table title
            columns: Column definitions {"key": "Header Name", ...}
            
        Returns:
            Rich Table object
        """
        table = Table(title=title)
        
        if isinstance(data, dict):
            # Simple key-value table
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")
            
            for key, value in data.items():
                display_key = columns.get(key, key.title()) if columns else key.title()
                table.add_row(display_key, str(value))
        
        elif isinstance(data, list) and data:
            # List of dictionaries - multi-column table
            if isinstance(data[0], dict):
                keys = data[0].keys()
                
                for key in keys:
                    header = columns.get(key, key.title()) if columns else key.title()
                    table.add_column(header, style="white")
                
                for item in data:
                    row = [str(item.get(key, "")) for key in keys]
                    table.add_row(*row)
        
        self.console.print(table)
        return table
    
    def create_layout_zone(self, zone_name: str, start_line: int, height: int) -> Dict[str, int]:
        """Create a layout zone for window-style positioning.
        
        Args:
            zone_name: Unique name for the zone
            start_line: Starting line number
            height: Height in lines
            
        Returns:
            Zone configuration dictionary
        """
        zone_config = {
            'start_line': start_line,
            'height': height,
            'current_line': start_line,
            'end_line': start_line + height - 1
        }
        
        self.layout_zones[zone_name] = zone_config
        return zone_config
    
    def render_in_zone(self, zone_name: str, content: str, clear_zone: bool = False):
        """Render content in a specific layout zone.
        
        Args:
            zone_name: Name of the zone to render in
            content: Content to display
            clear_zone: Whether to clear the zone first
        """
        if zone_name not in self.layout_zones:
            self.console.print(f"[yellow]⚠️ Zone '{zone_name}' not found[/]")
            return
        
        zone = self.layout_zones[zone_name]
        
        if clear_zone:
            # Clear the zone
            for line in range(zone['height']):
                sys.stdout.write(f'\033[{zone["start_line"] + line};0H\033[K')
            zone['current_line'] = zone['start_line']
        
        # Position cursor and render content
        sys.stdout.write(f'\033[{zone["current_line"]};0H')
        self.console.print(content)
        zone['current_line'] += content.count('\n') + 1
    
    def clear_zones(self):
        """Clear all defined layout zones."""
        for zone_name in self.layout_zones:
            self.render_in_zone(zone_name, "", clear_zone=True)
    
    def set_speech_handler(self, speech_handler):
        """Update the speech handler for TTS functionality."""
        self.speech_handler = speech_handler
    
    def render_system_info(self, info: Dict[str, Any], style: str = "table"):
        """Render system information in specified style.
        
        Args:
            info: System information dictionary
            style: Display style ("table", "panel", "list")
        """
        if style == "table":
            self.render_table(info, title="🔧 System Information")
        elif style == "panel":
            content = "\n".join([f"**{key}**: {value}" for key, value in info.items()])
            panel = Panel.fit(content, title="🔧 System Information", border_style="green")
            self.console.print(panel)
        else:  # list style
            self.console.print("[bold cyan]🔧 System Information:[/]")
            for key, value in info.items():
                self.console.print(f"  • {key}: [green]{value}[/]")
    
    def render_error(self, error_message: str, details: str = None):
        """Render error message in consistent format.
        
        Args:
            error_message: Main error message
            details: Optional detailed error information
        """
        self.console.print(f"[red]❌ {error_message}[/]")
        if details:
            self.console.print(f"[dim]Details: {details}[/]")
    
    def render_success(self, message: str, details: str = None):
        """Render success message in consistent format.
        
        Args:
            message: Success message
            details: Optional additional details
        """
        self.console.print(f"[green]✅ {message}[/]")
        if details:
            self.console.print(f"[dim]{details}[/]")
    
    def render_warning(self, message: str, details: str = None):
        """Render warning message in consistent format.
        
        Args:
            message: Warning message
            details: Optional additional details
        """
        self.console.print(f"[yellow]⚠️ {message}[/]")
        if details:
            self.console.print(f"[dim]{details}[/]")