"""
Prompt template configuration management.

Handles system prompts, prompt templates, and dynamic prompt generation.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console

from .config_loader import ConfigLoader

console = Console()


class PromptConfig(ConfigLoader):
    """Manages system prompts and prompt templates."""
    
    def __init__(self, config_dir: Path = None):
        """Initialize prompt configuration manager."""
        super().__init__(config_dir)
        
        # Check if default prompt templates exist
        if not self.config_exists("default.prompt.toml"):
            console.print(f"[yellow]⚠️ Default prompt config not found, please ensure default.prompt.toml exists[/]")
    
    def get_template(self, template_id: str, config_file: str = "default.prompt.toml") -> Optional[Dict[str, Any]]:
        """Get a prompt template by ID."""
        config = self.load_toml(config_file)
        return config.get(template_id)
    
    def list_templates(self, config_file: str = "default.prompt.toml") -> List[Dict[str, str]]:
        """List all available prompt templates."""
        config = self.load_toml(config_file)
        templates = []
        
        for template_id, template_data in config.items():
            if isinstance(template_data, dict):
                templates.append({
                    "id": template_id,
                    "name": template_data.get("name", template_id),
                    "description": template_data.get("description", "No description"),
                    "variables": template_data.get("variables", [])
                })
        
        return sorted(templates, key=lambda x: x["name"])
    
    def create_template(self, template_id: str, name: str, description: str,
                       template: str, variables: List[str] = None,
                       config_file: str = "custom.prompt.toml") -> bool:
        """Create a new prompt template."""
        # Extract variables from template if not provided
        if variables is None:
            variables = self.extract_variables(template)
        
        # Load existing config or create new
        config = self.load_toml(config_file) if self.config_exists(config_file) else {}
        
        template_data = {
            "name": name,
            "description": description,
            "template": template,
            "variables": variables,
            "created_at": self._get_timestamp()
        }
        
        config[template_id] = template_data
        
        success = self.save_toml(config_file, config)
        
        if success:
            console.print(f"[green]✅ Template '{template_id}' created in {config_file}[/]")
        
        return success
    
    def extract_variables(self, template: str) -> List[str]:
        """Extract variable placeholders from a template."""
        # Find all {variable_name} patterns
        variables = re.findall(r'\\{([^}]+)\\}', template)
        return sorted(list(set(variables)))
    
    def render_prompt(self, template_id: str, variables: Dict[str, Any],
                     config_file: str = "default.prompt.toml") -> Optional[str]:
        """Render a prompt template with provided variables."""
        template_data = self.get_template(template_id, config_file)
        
        if not template_data:
            console.print(f"[red]❌ Template '{template_id}' not found in {config_file}[/]")
            return None
        
        template = template_data["template"]
        required_vars = template_data.get("variables", [])
        
        # Check for missing variables
        missing_vars = [var for var in required_vars if var not in variables]
        if missing_vars:
            console.print(f"[yellow]⚠️ Missing variables: {', '.join(missing_vars)}[/]")
            # Provide default values for missing variables
            for var in missing_vars:
                variables[var] = f"[{var}]"
        
        try:
            rendered = template.format(**variables)
            return rendered.strip()
        except KeyError as e:
            console.print(f"[red]❌ Error rendering template: Missing variable {e}[/]")
            return None
        except Exception as e:
            console.print(f"[red]❌ Error rendering template: {e}[/]")
            return None
    
    def generate_system_prompt(self, assistant_config: Dict[str, Any],
                              template_id: str = "base_assistant",
                              config_file: str = "default.prompt.toml") -> str:
        """Generate a system prompt from assistant configuration."""
        # Extract variables from config
        variables = self._extract_variables_from_config(assistant_config)
        
        # Render the prompt
        rendered = self.render_prompt(template_id, variables, config_file)
        
        if rendered is None:
            # Fallback to basic prompt
            assistant_name = assistant_config.get("assistant", {}).get("name", "Assistant")
            user_name = assistant_config.get("assistant", {}).get("user_name", "user")
            return f"You are {assistant_name}, a helpful AI assistant talking to {user_name}."
        
        return rendered
    
    def _extract_variables_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract variables from assistant configuration."""
        variables = {}
        
        # Basic assistant info
        assistant = config.get("assistant", {})
        variables["assistant_name"] = assistant.get("name", "Assistant")
        variables["user_name"] = assistant.get("user_name", "user")
        variables["mode"] = assistant.get("mode", "chat")
        variables["working_dir"] = str(assistant.get("working_dir", "."))
        
        # Personality
        personality = config.get("personality", {})
        variables["personality_description"] = personality.get("description", "")
        
        # Capabilities
        capabilities = config.get("capabilities", {})
        enabled_caps = [cap for cap, enabled in capabilities.items() if enabled]
        variables["capabilities_list"] = "\\n".join(f"- {cap.replace('_', ' ').title()}" for cap in enabled_caps)
        
        # Custom instructions
        advanced = config.get("advanced", {})
        variables["custom_instructions"] = advanced.get("custom_instructions", "")
        
        # Voice settings
        voice = config.get("voice", {})
        variables["voice_name"] = voice.get("voice_name", "Default")
        variables["speech_backend"] = voice.get("speech_backend", "google")
        variables["noise_level"] = voice.get("noise_level", "normal")
        
        # RAG settings
        rag = config.get("rag", {})
        variables["rag_document_count"] = str(rag.get("document_count", 0))
        variables["rag_domains"] = ", ".join(rag.get("domains", []))
        variables["rag_last_updated"] = rag.get("last_updated", "Never")
        variables["rag_knowledge_path"] = rag.get("knowledge_base_path", "")
        
        # Programming context
        variables["primary_language"] = config.get("coding", {}).get("primary_language", "Python")
        
        # Image generation
        image = config.get("image", {})
        variables["image_models"] = ", ".join(image.get("models", []))
        variables["output_dir"] = image.get("output_dir", "./")
        
        # Research context
        research = config.get("research", {})
        variables["research_domains"] = ", ".join(research.get("domains", []))
        variables["sources_available"] = ", ".join(research.get("sources", []))
        
        return variables
    
    def validate_template(self, template: str, variables: List[str] = None) -> Dict[str, List[str]]:
        """Validate a prompt template."""
        errors = []
        warnings = []
        
        # Extract variables from template
        template_vars = self.extract_variables(template)
        
        # Check for empty template
        if not template.strip():
            errors.append("Template cannot be empty")
        
        # Check template length
        if len(template) < 50:
            warnings.append("Template seems very short")
        elif len(template) > 5000:
            warnings.append("Template is very long, may be truncated")
        
        # Check for unclosed braces
        open_braces = template.count('{')
        close_braces = template.count('}')
        if open_braces != close_braces:
            errors.append("Mismatched braces in template")
        
        # Check for invalid variable names
        for var in template_vars:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                errors.append(f"Invalid variable name: {var}")
        
        # Check against provided variable list
        if variables is not None:
            missing_in_template = set(variables) - set(template_vars)
            extra_in_template = set(template_vars) - set(variables)
            
            if missing_in_template:
                warnings.append(f"Variables not used in template: {', '.join(missing_in_template)}")
            
            if extra_in_template:
                warnings.append(f"Variables in template not in list: {', '.join(extra_in_template)}")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def preview_prompt(self, template_id: str, sample_variables: Dict[str, Any] = None,
                      config_file: str = "default.prompt.toml") -> str:
        """Preview a rendered prompt with sample data."""
        if sample_variables is None:
            sample_variables = self._get_sample_variables()
        
        rendered = self.render_prompt(template_id, sample_variables, config_file)
        
        if rendered is None:
            return "Error: Could not render template"
        
        return rendered
    
    def _get_sample_variables(self) -> Dict[str, Any]:
        """Get sample variables for preview purposes."""
        return {
            "assistant_name": "CodeBot",
            "user_name": "Developer",
            "personality_description": "Professional coding assistant focused on quality and best practices",
            "capabilities_list": "- File Operations\\n- Code Analysis\\n- Documentation Generation",
            "working_dir": "/home/user/project",
            "mode": "chat",
            "custom_instructions": "Always follow PEP 8 for Python code and include docstrings.",
            "voice_name": "Alex",
            "speech_backend": "google",
            "noise_level": "normal",
            "rag_document_count": "150",
            "rag_domains": "Python, Web Development, APIs",
            "rag_last_updated": "2024-01-15",
            "rag_knowledge_path": ".ai_context/knowledge",
            "primary_language": "Python",
            "image_models": "Stable Diffusion v1.5, DALL-E 2",
            "output_dir": "./",
            "research_domains": "Technology, Science, Software Development",
            "sources_available": "Academic papers, Documentation, Web sources"
        }
    
    def list_prompt_configs(self) -> List[str]:
        """List all prompt configuration files."""
        return self.list_configs("*.prompt.toml")
    
    def create_custom_prompt_config(self, filename: str, templates_data: Dict[str, Any]) -> bool:
        """Create a custom prompt configuration file."""
        if not filename.endswith('.prompt.toml'):
            filename += '.prompt.toml'
        
        success = self.save_toml(filename, templates_data)
        
        if success:
            console.print(f"[green]✅ Created custom prompt config: {filename}[/]")
        
        return success
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()