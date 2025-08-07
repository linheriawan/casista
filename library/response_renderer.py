#!/usr/bin/env python3
"""
Response Renderer - Advanced rendering for AI responses with fullscreen display.

Provides unified rendering for console output, speech synthesis, and fullscreen layouts
including tables, live chat updates, and real-time streaming responses.
"""

from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
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
        self.live_display = None  # For fullscreen Live display
    
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
    
    
    
    def initialize_fullscreen_mode(self, header_data: list, initial_footer: str = "Ready...", title: str = "📊 Session Info"):
        """Initialize htop-style fullscreen mode with live layout management.
        
        Args:
            header_data: System info data for header table
            initial_footer: Initial footer message
            title: Custom title for header panel
        """
        # Store layout state
        self.fullscreen_state = {
            'header_data': header_data,
            'chat_messages': [],
            'footer_message': initial_footer,
            'header_title': title,
            'is_active': True
        }
        
        # Create initial layout
        initial_layout = self._create_layout()
        
        # Start Live display for in-place updates (screen=False to allow input prompt)
        self.live_display = Live(initial_layout, refresh_per_second=10, screen=False)
        self.live_display.start()
    
    def add_chat_message(self, role: str, message: str, assistant_name: str = "Assistant"):
        """Add a message to the chat area and refresh layout.
        
        Args:
            role: "user" or "assistant"
            message: Message content
            assistant_name: Name of assistant for display
        """
        if not hasattr(self, 'fullscreen_state') or not self.fullscreen_state.get('is_active'):
            # Fallback to regular rendering if fullscreen not active
            if role == "assistant":
                self.console.print(f"[bold cyan]🤖 {assistant_name}:[/] {message}")
            else:
                self.console.print(f"[bold green]👤 You:[/] {message}")
            return
        
        # Format message for chat display
        if role == "assistant":
            formatted_msg = f"[bold cyan]🤖 {assistant_name}:[/] {message}"
        else:
            formatted_msg = f"[bold green]👤 You:[/] {message}"
        
        # Add to chat history
        self.fullscreen_state['chat_messages'].append(formatted_msg)
        
        # Keep only last 20 messages for better scrolling in limited space
        if len(self.fullscreen_state['chat_messages']) > 20:
            self.fullscreen_state['chat_messages'] = self.fullscreen_state['chat_messages'][-20:]
        
        # Update Live display
        self._update_live_display()
    
    def update_footer_message(self, message: str):
        """Update footer status message and refresh layout.
        
        Args:
            message: New footer message
        """
        if hasattr(self, 'fullscreen_state') and self.fullscreen_state.get('is_active'):
            self.fullscreen_state['footer_message'] = message
            self._update_live_display()
        else:
            # Fallback to regular status message
            self.console.print(f"[dim]{message}[/]")
    
    def _update_live_display(self):
        """Update the live display with current layout state."""
        if not hasattr(self, 'fullscreen_state') or not self.fullscreen_state.get('is_active'):
            return
        
        if not hasattr(self, 'live_display') or not self.live_display:
            return
        
        # Create updated layout
        layout = self._create_layout()
        
        # Update the live display in-place (no screen clearing!)
        self.live_display.update(layout)
    
    def _create_layout(self) -> Layout:
        """Create the layout structure with current state."""
        state = self.fullscreen_state
        
        # Create layout with input box
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6),       # Session info (needs space for rows + borders)
            Layout(name="status", size=2),       # Status bar (needs space for borders)
            Layout(name="input_box", size=2),     # Input box (compact)
            Layout(name="content", ratio=1)     # Chat conversation (flexible)
        )
        
        # Header: Compact system info table
        header_table = Table(show_header=False, box=None, padding=(0, 1))
        header_table.add_column("Prop1", style="cyan", no_wrap=True, min_width=10)
        header_table.add_column("Val1", style="white", no_wrap=True, min_width=12)
        header_table.add_column("Prop2", style="cyan", no_wrap=True, min_width=10)
        header_table.add_column("Val2", style="white", no_wrap=True, min_width=12)
        header_table.add_column("Prop3", style="cyan", no_wrap=True, min_width=10)
        header_table.add_column("Val3", style="white", no_wrap=True, min_width=12)
        
        for item in state['header_data']:
            prop1 = item.get("Property", "")
            val1 = item.get("Value", "")
            prop2 = item.get("Property2", "")
            val2 = item.get("Value2", "")
            prop3 = item.get("Property3", "")
            val3 = item.get("Value3", "")
            header_table.add_row(prop1, val1, prop2, val2, prop3, val3)
        
        # Use stored dynamic title
        header_title = state.get('header_title', '📊 Session Info')
        layout["header"].update(Panel(header_table, title=header_title, 
                                    border_style="green", padding=(0, 1)))
        
        # Status: Between header and content
        layout["status"].update(Panel(state['footer_message'], title="📝 Status", 
                                    border_style="yellow", padding=(0, 1)))
        
        # Content: Chat messages (automatically scrollable with newest at bottom)
        if state['chat_messages']:
            # Show most recent messages that fit in the space
            chat_content = "\n".join(state['chat_messages'])
        else:
            chat_content = "[dim]Conversation will appear here...[/]"
        
        content_panel = Panel(chat_content, title="💬 Chat", border_style="blue", padding=(0, 1))
        layout["content"].update(content_panel)
        
        # Input box: Compact input area with border
        layout["input_box"].update(Panel("", title="💬 Input", border_style="cyan", padding=(0, 1)))
        
        return layout
    
    def pause_live_display(self):
        """Pause live display to allow user input without interference."""
        if hasattr(self, 'live_display') and self.live_display:
            try:
                self.live_display.stop()
                self._display_was_paused = True
            except:
                pass  # Ignore errors during pause
    
    def resume_live_display(self):
        """Resume live display after user input is complete."""
        if hasattr(self, 'fullscreen_state') and self.fullscreen_state.get('is_active'):
            if hasattr(self, '_display_was_paused') and self._display_was_paused:
                try:
                    # Recreate and start live display
                    layout = self._create_layout()
                    self.live_display = Live(layout, refresh_per_second=10, screen=False)
                    self.live_display.start()
                    self._display_was_paused = False
                except:
                    pass  # Ignore errors during resume
    
    def handle_streaming_response(self, event: str, content: str, assistant_name: str):
        """Handle streaming response events from AI model.
        
        Args:
            event: "start", "chunk", "complete", "interrupted", or "error"
            content: Current content (partial or complete)
            assistant_name: Name of assistant
        """
        if not hasattr(self, 'fullscreen_state') or not self.fullscreen_state.get('is_active'):
            # Fallback to regular rendering if fullscreen not active
            if event == "chunk":
                # Show streaming content with cursor
                print(f"\r🤖 {assistant_name}: {content}▊", end="", flush=True)
            elif event == "complete":
                # Final output
                print(f"\r🤖 {assistant_name}: {content}")
            return
        
        if event == "start":
            # Update status to thinking
            self.update_footer_message("🤖 Thinking...")
            # Add placeholder message that will be updated
            self._streaming_message_index = len(self.fullscreen_state['chat_messages'])
            self.fullscreen_state['chat_messages'].append(f"[bold cyan]🤖 {assistant_name}:[/] [dim]Thinking...[/]")
            self._update_live_display()
            
        elif event == "chunk":
            # Update the streaming message in place
            if hasattr(self, '_streaming_message_index'):
                formatted_msg = f"[bold cyan]🤖 {assistant_name}:[/] {content}[dim]▊[/]"
                self.fullscreen_state['chat_messages'][self._streaming_message_index] = formatted_msg
                self._update_live_display()
                
        elif event == "complete":
            # Finalize the message
            if hasattr(self, '_streaming_message_index'):
                formatted_msg = f"[bold cyan]🤖 {assistant_name}:[/] {content}"
                self.fullscreen_state['chat_messages'][self._streaming_message_index] = formatted_msg
                self._update_live_display()
                delattr(self, '_streaming_message_index')
            self.update_footer_message("Ready for input...")
            
        elif event in ["interrupted", "error"]:
            # Handle errors
            if hasattr(self, '_streaming_message_index'):
                formatted_msg = f"[bold cyan]🤖 {assistant_name}:[/] [red]{content}[/]"
                self.fullscreen_state['chat_messages'][self._streaming_message_index] = formatted_msg
                self._update_live_display()
                delattr(self, '_streaming_message_index')
            self.update_footer_message(f"❌ {event.title()}")
    
    def exit_fullscreen_mode(self):
        """Exit fullscreen mode and return to normal rendering."""
        if hasattr(self, 'fullscreen_state'):
            self.fullscreen_state['is_active'] = False
        
        # Stop live display properly
        if hasattr(self, 'live_display') and self.live_display:
            try:
                self.live_display.stop()
            except:
                pass  # Ignore errors during cleanup
            self.live_display = None
    
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