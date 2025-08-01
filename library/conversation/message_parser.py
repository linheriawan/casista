"""
Message parsing for extracting reasoning, code blocks, and other structured content.

Handles model-specific response patterns and enhances message storage.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from rich.console import Console

console = Console()


class MessageParser:
    """Parses messages according to configured rules and extracts structured content."""
    
    def __init__(self, parsing_config: Dict[str, Any]):
        """Initialize message parser with configuration."""
        self.config = parsing_config
        self.extract_reasoning = parsing_config.get("extract_reasoning", True)
        self.reasoning_patterns = parsing_config.get("reasoning_patterns", [])
        self.extract_code_blocks = parsing_config.get("extract_code_blocks", True)
        self.preserve_original = parsing_config.get("preserve_original", True)
        self.response_sections = parsing_config.get("response_sections", ["reasoning", "answer", "code"])
    
    def parse_message(self, content: str, role: str = "assistant") -> Dict[str, Any]:
        """Parse message content and extract structured information."""
        parsed = {
            "content": content,
            "role": role,
            "timestamp": self._get_timestamp(),
            "parsed_sections": {}
        }
        
        if role == "assistant" and content:
            # Extract reasoning sections
            if self.extract_reasoning:
                reasoning_data = self._extract_reasoning(content)
                if reasoning_data:
                    parsed["parsed_sections"].update(reasoning_data)
            
            # Extract code blocks
            if self.extract_code_blocks:
                code_blocks = self._extract_code_blocks(content)
                if code_blocks:
                    parsed["parsed_sections"]["code_blocks"] = code_blocks
            
            # Generate clean answer (without reasoning tags)
            clean_answer = self._generate_clean_answer(content)
            if clean_answer != content:
                parsed["parsed_sections"]["clean_answer"] = clean_answer
                
            # Calculate content metrics
            parsed["metrics"] = self._calculate_metrics(content, parsed["parsed_sections"])
        
        return parsed
    
    def _extract_reasoning(self, content: str) -> Dict[str, Any]:
        """Extract reasoning content based on configured patterns."""
        reasoning_data = {}
        
        for pattern_config in self.reasoning_patterns:
            tag = pattern_config["tag"]
            reasoning_type = pattern_config["type"]
            strip_tags = pattern_config.get("strip_tags", True)
            
            # Create regex pattern for the tag
            if strip_tags:
                # Extract content without tags
                pattern = rf"<{tag}>(.*?)</{tag}>"
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            else:
                # Extract content with tags
                pattern = rf"<{tag}>.*?</{tag}>"
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if matches:
                if len(matches) == 1:
                    reasoning_data[reasoning_type] = matches[0].strip()
                else:
                    reasoning_data[reasoning_type] = [match.strip() for match in matches]
        
        return reasoning_data
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from content."""
        code_blocks = []
        
        # Pattern for code blocks with language specification
        pattern = r"```(\w+)?\n?(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        
        for lang, code in matches:
            code_blocks.append({
                "language": lang.strip() if lang else "text",
                "code": code.strip(),
                "type": "code_block"
            })
        
        # Pattern for inline code
        inline_pattern = r"`([^`]+)`"
        inline_matches = re.findall(inline_pattern, content)
        
        for inline_code in inline_matches:
            if len(inline_code.strip()) > 3:  # Ignore very short inline code
                code_blocks.append({
                    "language": "text",
                    "code": inline_code.strip(),
                    "type": "inline_code"
                })
        
        return code_blocks
    
    def _generate_clean_answer(self, content: str) -> str:
        """Generate clean answer by removing reasoning tags."""
        clean_content = content
        
        # Remove reasoning patterns
        for pattern_config in self.reasoning_patterns:
            tag = pattern_config["tag"]
            pattern = rf"<{tag}>.*?</{tag}>"
            clean_content = re.sub(pattern, "", clean_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        clean_content = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_content)
        clean_content = clean_content.strip()
        
        return clean_content
    
    def _calculate_metrics(self, content: str, parsed_sections: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics about the message content."""
        return {
            "total_length": len(content),
            "word_count": len(content.split()),
            "line_count": len(content.splitlines()),
            "has_reasoning": bool(any(key in parsed_sections for key in ["reasoning", "analysis", "reflection"])),
            "has_code": bool(parsed_sections.get("code_blocks")),
            "sections_count": len(parsed_sections)
        }
    
    def get_display_content(self, parsed_message: Dict[str, Any], 
                           include_reasoning: bool = True) -> str:
        """Get content for display based on preferences."""
        parsed_sections = parsed_message.get("parsed_sections", {})
        
        if not include_reasoning and "clean_answer" in parsed_sections:
            return parsed_sections["clean_answer"]
        
        return parsed_message["content"]
    
    def get_reasoning_content(self, parsed_message: Dict[str, Any]) -> Optional[str]:
        """Extract reasoning content for separate display."""
        parsed_sections = parsed_message.get("parsed_sections", {})
        
        reasoning_parts = []
        for reasoning_type in ["reasoning", "analysis", "reflection"]:
            if reasoning_type in parsed_sections:
                content = parsed_sections[reasoning_type]
                if isinstance(content, list):
                    reasoning_parts.extend(content)
                else:
                    reasoning_parts.append(content)
        
        return "\n\n".join(reasoning_parts) if reasoning_parts else None
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def update_parsing_config(self, new_config: Dict[str, Any]):
        """Update parsing configuration."""
        self.config.update(new_config)
        self.extract_reasoning = self.config.get("extract_reasoning", True)
        self.reasoning_patterns = self.config.get("reasoning_patterns", [])
        self.extract_code_blocks = self.config.get("extract_code_blocks", True)
        self.preserve_original = self.config.get("preserve_original", True)
        self.response_sections = self.config.get("response_sections", ["reasoning", "answer", "code"])
        
        console.print("[dim]🔄 Message parsing configuration updated[/]")