#!/usr/bin/env python3
"""
Operation Handler - Manages assistant sessions and heavy business logic.

Handles session implementations while main.py handles routing.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel

from library.assistant_cfg import AssistantConfig
from library.config_loader import ConfigLoader
from library.response_renderer import ResponseRenderer

console = Console()

class OperationHandler:
    """Handles assistant session implementations and setup."""
    
    def __init__(self):
        """Initialize operation handler."""
        self.config_loader = ConfigLoader()
        self.syscfg = self.config_loader.load_toml("sys.definition.toml")
        self.assistant_config = AssistantConfig()
    
    def run_setup(self) -> bool:
        """Handle initial setup."""
        renderer = ResponseRenderer(console)
        
        setup_info = {
            "Task": "AI Assistant Setup",
            "Description": "This will guide you through the initial setup process"
        }
        renderer.render_system_info(setup_info, style="panel")
        
        # Create default directories
        Path("knowledge").mkdir(exist_ok=True)
        Path("configuration").mkdir(exist_ok=True)
        
        renderer.render_success("Created default directories")
        
        # Ask if user wants to create first assistant
        from rich.prompt import Confirm
        if Confirm.ask("Would you like to create your first assistant?"):
            from helper.manage_agent import AgentManager
            agent_manager = AgentManager()
            agent_manager.create_agent_interactive()
        
        completion_info = {
            "Status": "Setup Complete!",
            "Next Steps": "You can now use the AI assistant system",
            "Try": "coder --list-agents"
        }
        renderer.render_system_info(completion_info, style="panel")
        
        return True
    
    def run_session(self, args) -> bool:
        """Run a session (conversation or image generation)."""
        renderer = ResponseRenderer(console)
        
        if not args.assistant_name:
            renderer.render_error("Assistant name required", 
                                "Use --create-agent to create a new assistant or --list-agents to see available assistants")
            return False
        
        # Load assistant configuration
        session_working_dir = self._get_session_working_dir(args)
        session_config = self._load_or_create_assistant(args, session_working_dir)
        
        if not session_config:
            renderer.render_error(f"Failed to load assistant '{args.assistant_name}'")
            return False
        
        # Override model if specified
        if args.model:
            session_config["assistant"]["model"] = args.model
            renderer.render_success(f"Using model for this session: {args.model}")
        
        # Prepare all session components
        try:
            components = self.prepare(args)
        except RuntimeError as e:
            renderer.render_error(str(e))
            return False
        
        # Display session info
        self._display_session_info(components, args)
        
        # Main interactive loop
        return self._run_interactive_loop(components, args)
    
    def _run_interactive_loop(self, components: Dict[str, Any], args) -> bool:
        """Run the main interactive conversation loop."""
        # Extract components
        session = components['session']
        chat_manager = components['chat_manager']
        file_ops = components['file_ops']
        user_name = components['user_name']
        renderer = components['renderer']
        speech_handler = components['speech_handler']
        generator = components['generator']
        
        # Loop state
        current_mode = components['current_mode']
        last_generated_image = components['last_generated_image']
        current_image_model = components.get('current_image_model')
        
        try:
            while True:
                try:
                    # Get user input based on current mode
                    if current_mode == "speech" and speech_handler:
                        # Voice input
                        user_input = speech_handler.listen()
                        if not user_input:
                            continue  # No speech detected, try again
                    elif current_mode == "image":
                        # Text input for image mode
                        user_input = input(f"\n🎨 [{user_name}] ({session.working_dir.name}): ").strip()
                    else:
                        # Regular text input for chat mode
                        user_input = input(f"\n💬 [{user_name}] ({session.working_dir.name}): ").strip()
                    
                    if user_input.lower() in ['exit', 'quit', 'q', 'goodbye', 'bye']:
                        renderer.render_success("Goodbye! 👋")
                        break
                    
                    if user_input.lower() == 'reset':
                        chat_manager.clear_conversation()
                        last_generated_image = None
                        renderer.render_success("Conversation and context reset")
                        continue
                    
                    # Handle mode switching
                    if user_input.startswith('/mode='):
                        new_mode = user_input[6:].strip().lower()
                        if new_mode in ['chat', 'speech', 'image']:
                            current_mode = new_mode
                            renderer.render_success(f"Switched to {new_mode} mode")
                        else:
                            renderer.render_error("Invalid mode. Use: /mode=chat, /mode=speech, or /mode=image")
                        continue
                    
                    # Handle directory change
                    if user_input.startswith('/dir='):
                        new_dir_str = user_input[5:].strip()
                        try:
                            new_dir = Path(new_dir_str).expanduser().resolve()
                            if new_dir.exists() and new_dir.is_dir():
                                session.update_working_directory(new_dir)
                                renderer.render_success(f"Changed directory to: {new_dir}")
                            else:
                                renderer.render_error(f"Directory not found: {new_dir}")
                        except Exception as e:
                            renderer.render_error(f"Invalid directory path: {e}")
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Process input based on current mode
                    if current_mode == "image" and generator:
                        # Handle image generation
                        result_data = self.generate_conversational_image(
                            session, chat_manager, generator, user_input, current_image_model or "stable-diffusion", args, last_generated_image
                        )
                        
                        if result_data and result_data.get('success'):
                            last_generated_image = result_data
                            renderer.render_success(f"Image saved: {result_data['path']}")
                        elif result_data and not result_data.get('success'):
                            renderer.render_error("Image generation failed")
                    else:
                        # Handle regular chat/conversation
                        response_data = chat_manager.send_message(user_input)
                        
                        if response_data.get("error"):
                            renderer.render_error(response_data['content'])
                            continue
                        
                        # Use ResponseRenderer for unified output and speech
                        mode = "text_and_speech" if current_mode == "speech" else "text"
                        renderer.render_response(mode, session.assistant_name, response_data, style="chat")
                        
                        # Handle file operations
                        if file_ops and response_data['content']:
                            code_blocks = file_ops.parse_code_blocks(response_data['content'])
                            if code_blocks:
                                file_ops.apply_code_changes(code_blocks, args.auto_confirm)
                
                except KeyboardInterrupt:
                    renderer.render_success("Goodbye! 👋")
                    break
                except Exception as e:
                    renderer.render_error(f"Error: {e}")
        
        finally:
            # Clean up speech resources
            if speech_handler:
                speech_handler.cleanup()
        
        return True
    
    def _display_session_info(self, components: Dict[str, Any], args) -> None:
        """Display session information panel."""
        session = components['session']
        model_info = components['model_info']
        speech_handler = components['speech_handler']
        renderer = components['renderer']
        
        # Use prepared components for mode info
        mode_name = components['mode_name']
        mode_icon = components['mode_icon']
        
        # Build session info data structure
        session_info = {
            "Assistant": session.assistant_name,
            "Mode": mode_name,
            "Model": model_info['model'],
            "Temperature": str(model_info['temperature']),
            "Max Tokens": str(model_info['max_tokens']),
            "Personality": model_info['personality'],
            "Working Directory": str(session.working_dir)
        }
        
        # Add mode-specific info
        current_mode = components['current_mode']
        if current_mode == "image":
            image_models = session.config.get('image', {}).get('models', ['stable-diffusion'])
            session_info["Image Models"] = ', '.join(image_models)
            commands = "'exit', 'reset', '/mode=chat', '/mode=speech', 'list models', 'switch model <name>'"
        elif speech_handler:
            status = speech_handler.get_status()
            session_info["Speech Backend"] = status['speech_backend']
            session_info["Voice"] = f"{status['voice_name']} (ID: {status['voice_id']})"
            session_info["Speech Rate"] = f"{status['speech_rate']} WPM"
            commands = "'exit', 'reset', '/mode=chat', '/mode=image', or speak naturally"
        else:
            commands = "'exit', 'reset', '/dir=<path>', '/mode=speech', '/mode=image'"
        
        session_info["Commands"] = commands
        
        # Use ResponseRenderer to display session info
        renderer.render_system_info(session_info, style="table")
        
        # Display mode-specific ready messages using ResponseRenderer
        if current_mode == "image":
            components['current_image_model'] = session.config.get("image", {}).get("models", ["stable-diffusion"])[0]
            renderer.render_success("Image generation mode ready! Describe what you want to create...", 
                                  "Try: 'list models', 'switch model <name>', '/mode=chat', '/mode=speech'")
        elif speech_handler:
            renderer.render_success("Speech mode ready! Start speaking...", 
                                  "Say 'exit', 'quit', or 'reset' for commands")
        else:
            renderer.render_success("Chat mode ready!", 
                                  "Type your messages or use commands like '/mode=speech', '/mode=image'")
    
    def _get_session_working_dir(self, args) -> Path:
        """Get the working directory for the session."""
        import os
        
        # The install script sets ORIGINAL_CWD to the directory where the user ran coder
        original_cwd = os.environ.get('ORIGINAL_CWD')
        if original_cwd and Path(original_cwd).exists():
            user_cwd = Path(original_cwd)
        else:
            # Fallback to current Python working directory
            user_cwd = Path.cwd()
        
        return args.working_dir or user_cwd
    
    def _load_or_create_assistant(self, args, session_working_dir: Path) -> Optional[Dict[str, Any]]:
        """Load existing assistant or create new one with default settings."""
        session_config = self.assistant_config.load_assistant_for_session(
            args.assistant_name, 
            session_working_dir
        )
        
        if not session_config:
            # Auto-create assistant with default settings
            renderer = ResponseRenderer(console)
            renderer.render_warning(f"Assistant '{args.assistant_name}' not found. Creating with default settings...")
            
            # Determine default model and personality based on mode
            default_model = args.model or "qwen2.5-coder:3b"
            default_personality = "coder"
            
            # Create assistant with default configuration
            success = self.assistant_config.create_assistant(
                assistant_name=args.assistant_name,
                model=default_model,
                personality=default_personality,
                working_dir=session_working_dir
            )
            
            if not success:
                renderer.render_error(f"Failed to create assistant '{args.assistant_name}'")
                return None
            
            # Load the newly created assistant
            session_config = self.assistant_config.load_assistant_for_session(
                args.assistant_name, 
                session_working_dir
            )
        
        return session_config
    
    def generate_single_image(self, args):
        """Generate a single image (one-shot mode) without full session setup."""
        renderer = ResponseRenderer(console)
        
        if not args.assistant_name:
            renderer.render_error("Assistant name required for image generation")
            return False
        
        renderer.render_success(f"Processing image request: '{args.prompt}'")
        
        # Load minimal assistant configuration for image generation
        session_working_dir = self._get_session_working_dir(args)
        session_config = self._load_or_create_assistant(args, session_working_dir)
        
        if not session_config:
            renderer.render_error(f"Failed to load assistant '{args.assistant_name}'")
            return False
        
        # Initialize minimal session manager for image generation
        from library.session_manager import SessionManager
        session = SessionManager(session_config)
        
        # Get image generator
        generator = session.get_image_generator()
        if not generator:
            renderer.render_error("Image generation not enabled for this assistant",
                                "Enable with: coder configure <assistant> and set image_generation = true")
            return False
        
        # Get chat manager for prompt enhancement
        chat_manager = session.get_chat_manager()
        
        # Get AI assistance for prompt enhancement
        response_data = chat_manager.send_message(
            f"Please help me create an image: {args.prompt}",
            system_prompt=session.get_mode_specific_system_prompt("image")
        )
        
        if response_data.get("error"):
            renderer.render_warning("AI assistance failed, using original prompt")
            enhanced_prompt = args.prompt
        else:
            # Display AI's response and suggestions using ResponseRenderer
            renderer.render_response("text", session.assistant_name, response_data, style="panel")
            
            enhanced_prompt = args.prompt  # Could extract enhanced prompt from AI response
        
        # Generate image
        image_config = session.config.get("image", {})
        models = image_config.get("models", [])
        
        result = generator.generate_image(
            prompt=enhanced_prompt,
            width=getattr(args, 'width', 512),
            height=getattr(args, 'height', 512),
            model_name=models[0] if models else None
        )
        
        if result:
            renderer.render_success("Image generated successfully!", f"Saved to: {result}")
            return True
        else:
            renderer.render_error("Image generation failed")
            return False
    
    def handle_one_shot_query(self, args) -> bool:
        """Handle one-shot query without interactive mode."""
        renderer = ResponseRenderer(console)
        
        if not args.assistant_name:
            renderer.render_error("Assistant name required for query")
            return False
        
        renderer.render_success(f"Processing query: '{args.query}'")
        
        # Load assistant configuration
        session_working_dir = self._get_session_working_dir(args)
        session_config = self._load_or_create_assistant(args, session_working_dir)
        
        if not session_config:
            renderer.render_error(f"Failed to load assistant '{args.assistant_name}'")
            return False
        
        # Initialize session manager (text-only for one-shot queries)
        from library.session_manager import SessionManager
        session = SessionManager(session_config)
        
        # Get configured components
        chat_manager = session.get_chat_manager()
        file_ops = session.get_file_operations(enable=not args.no_file_ops)
        
        # Send query
        response_data = chat_manager.send_message(args.query)
        
        if response_data.get("error"):
            renderer.render_error(response_data['content'])
            return False
        
        # Use ResponseRenderer for unified output (text-only for one-shot queries)
        renderer.render_response("text", session.assistant_name, response_data, style="chat")
        
        # Handle file operations
        if file_ops and response_data['content']:
            code_blocks = file_ops.parse_code_blocks(response_data['content'])
            if code_blocks:
                file_ops.apply_code_changes(code_blocks, args.auto_confirm)
        
        # No speech resources to clean up in one-shot mode
        
        return True
    
    def prepare(self, args) -> Dict[str, Any]:
        """Prepare all session components and configuration.
        
        Returns:
            Dict containing all prepared components: session, model_info, speech_handler,
            generator, chat_manager, file_ops, user_name, renderer, etc.
        """
        from library.session_manager import SessionManager
        from library.speech_handler import SpeechHandler
        
        # Load assistant configuration if not already loaded
        if not hasattr(self, '_current_session_config'):
            session_working_dir = self._get_session_working_dir(args)
            session_config = self._load_or_create_assistant(args, session_working_dir)
            
            if not session_config:
                raise RuntimeError(f"Failed to load assistant '{args.assistant_name}'")
            
            # Override model if specified
            if args.model:
                session_config["assistant"]["model"] = args.model
                # Note: renderer not available here, will be displayed in run_session
            
            self._current_session_config = session_config
        else:
            session_config = self._current_session_config
        
        # Initialize session manager
        session = SessionManager(session_config)
        model_info = session.get_model_info()
        
        # Initialize speech handler for voice mode
        speech_handler = None
        if args.mode == "speech":
            voice_config = session.get_voice_config()
            speech_handler = SpeechHandler(voice_config)
            
            # Setup speech components
            if not speech_handler.setup_tts() or not speech_handler.setup_stt():
                speech_handler = None
                args.mode = "chat"
                # Error will be displayed by speech_handler setup methods
            else:
                speech_handler.calibrate_microphone()
        
        # Setup image generation if in image mode
        generator = None
        if args.mode == "image":
            generator = session.get_image_generator()
            if not generator:
                raise RuntimeError("Image generation not enabled for this assistant. Enable with: coder configure <assistant> and set image_generation = true")
        
        # Get properly configured components
        chat_manager = session.get_chat_manager()
        file_ops = session.get_file_operations(enable=not args.no_file_ops)
        user_name = session.get_user_name()
        
        # Create renderer with speech handler if available
        renderer = ResponseRenderer(console, speech_handler)
        
        # Determine mode display info
        if args.mode == "image":
            mode_name = "Image Generation Mode"
            mode_icon = "🎨"
        elif speech_handler:
            mode_name = "Speech Mode"
            mode_icon = "🎤"
        else:
            mode_name = "Chat Mode"
            mode_icon = "🤖"
        
        return {
            'session': session,
            'model_info': model_info,
            'speech_handler': speech_handler,
            'generator': generator,
            'chat_manager': chat_manager,
            'file_ops': file_ops,
            'user_name': user_name,
            'renderer': renderer,
            'current_mode': args.mode,
            'last_generated_image': None,
            'current_image_model': None,
            'mode_name': mode_name,
            'mode_icon': mode_icon
        }
    
    def generate_conversational_image(self, session, chat_manager, generator, user_input, current_model, args, last_image=None):
        """Generate image with conversational AI assistance."""
        
        # Build context for the AI assistant
        context_info = ""
        if last_image:
            context_info = f"\nPrevious image: '{last_image['prompt']}' → {last_image['path']}"
        
        # Get AI assistance for understanding the request
        ai_prompt = self.syscfg["main"]["ai_prompt"].format(user_input=user_input, context_info=context_info)    
        response_data = chat_manager.send_message(ai_prompt)
        
        renderer = ResponseRenderer(console)
        
        if response_data.get("error"):
            renderer.render_warning("AI assistance failed, using original request")
            enhanced_prompt = user_input
            explanation = "Using original request as provided."
        else:
            # Display AI's response using ResponseRenderer
            renderer.render_response("text", session.assistant_name, response_data, style="chat")
            
            # Extract enhanced prompt from AI response (simplified for now)
            enhanced_prompt = user_input
            explanation = response_data.get('clean_answer', response_data.get('content', ''))
        
        # Generate the image
        renderer.render_success(f"Generating with model: {current_model}")
        
        result = generator.generate_image(
            prompt=enhanced_prompt,
            width=getattr(args, 'width', 512),
            height=getattr(args, 'height', 512),
            model_name=current_model
        )
        
        if result:
            # Add to conversation context
            image_context = f"Generated image: '{enhanced_prompt}' using {current_model} → {result}"
            chat_manager.context_manager.add_message("assistant", image_context)
            
            return {
                'success': True,
                'path': result,
                'prompt': enhanced_prompt,
                'model': current_model,
                'explanation': explanation
            }
        else:
            # Add failure to context
            failure_context = f"Failed to generate image: '{enhanced_prompt}' using {current_model}"
            chat_manager.context_manager.add_message("assistant", failure_context)
            
            return {
                'success': False,
                'prompt': enhanced_prompt,
                'model': current_model,
                'error': 'Generation failed'
            }
    
    def run_enhanced_image_pipeline(self, session, chat_manager, generator, pipeline_definition, current_model, last_image=None):
        """Run an enhanced image pipeline with multiple step types."""
        renderer = ResponseRenderer(console)
        
        renderer.render_success(f"Running enhanced pipeline: {pipeline_definition}")
        
        # Parse pipeline definition
        # Format: "step1 -> step2 -> step3" or "generate:prompt -> refine:new prompt -> upscale:2x"
        steps = [step.strip() for step in pipeline_definition.split('->')]
        
        pipeline_info = {
            "Pipeline": pipeline_definition,
            "Steps": f"{len(steps)} steps",
            "Steps Detail": ' → '.join(steps)
        }
        renderer.render_system_info(pipeline_info, style="table")
        
        current_image = last_image
        pipeline_steps = []
        
        # Parse each step
        for step_str in steps:
            if ':' in step_str:
                step_type, step_content = step_str.split(':', 1)
                step_type = step_type.strip().lower()
                step_content = step_content.strip()
            else:
                step_type = 'generate'
                step_content = step_str
            
            step_config = {
                'type': step_type,
                'prompt': step_content,
                'model': current_model
            }
            
            # Add step-specific parameters
            if step_type == 'upscale':
                try:
                    scale_factor = int(step_content.replace('x', ''))
                    step_config['scale_factor'] = scale_factor
                    step_config['prompt'] = ''
                except:
                    step_config['scale_factor'] = 2
            elif step_type == 'refine':
                step_config['strength'] = 0.7
            
            pipeline_steps.append(step_config)
        
        # Execute pipeline using generator's run_pipeline method
        base_prompt = pipeline_steps[0]['prompt'] if pipeline_steps else "abstract art"
        
        # If we have a starting image, modify the first step to be a refinement
        if current_image and current_image.get('path'):
            if pipeline_steps and pipeline_steps[0]['type'] == 'generate':
                pipeline_steps[0]['type'] = 'refine'
                pipeline_steps[0]['base_image'] = current_image['path']
        
        result_path = generator.run_pipeline(base_prompt, pipeline_steps)
        
        if result_path:
            result_data = {
                'success': True,
                'path': result_path,
                'prompt': f"pipeline result: {pipeline_definition}",
                'model': current_model,
                'type': 'pipeline'
            }
            
            # Add to conversation context
            context_msg = f"Completed image pipeline: {pipeline_definition} → {result_path}"
            chat_manager.context_manager.add_message("assistant", context_msg)
            
            renderer.render_success(f"Pipeline completed! Result: {result_path}")
            return result_data
        else:
            renderer.render_error("Pipeline failed")
            return None