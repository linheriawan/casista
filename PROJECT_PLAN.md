# Project Plan - Casista AI Assistant System

## Overview
Development roadmap for transforming Casista into a streamlined, modular AI assistant system with clear component separation and enhanced capabilities.

## Target Architecture

### Core Components

#### 1. **Main Application** 
- **Status:** 🟢 Complete
- **Description:** Central configuration and management hub
- **Features:**
  - ✅ Pure routing architecture (294 lines)
  - ✅ Argument parsing and command routing
  - ✅ Management command handling
  - ✅ Direct operation calls (--prompt, --setup)
  - ✅ Session routing to operation handler
- **Dependencies:** operation_handler.py for business logic
- **Priority:** HIGH

#### 2. **Agents System**
- **Status:** 🟢 Complete
- **Description:** AI assistant instances with different capabilities
- **Features:**
  - ✅ Agent creation and management
  - ✅ Multiple assistant personas
  - ✅ Per-assistant configuration
  - ✅ Working directory management
  - ✅ Model and personality assignment
- **Dependencies:** AssistantConfig, helper/manage_agent.py
- **Priority:** HIGH

#### 3. **Speech Mode Agent**
- **Status:** 🟢 Complete
- **Description:** Unified conversation with speech capability
- **Features:**
  - ✅ Speech mode as conversation toggle
  - ✅ Dedicated SpeechHandler class (272 lines)
  - ✅ TTS/STT management with multiple backends
  - ✅ Microphone calibration and noise handling
- **Dependencies:** library/speech_handler.py, conversation system
- **Priority:** MEDIUM

#### 4. **Conversation System**
- **Status:** 🟢 Complete
- **Description:** Unified chat/speech conversation handler
- **Features:**
  - ✅ Unified run_conversation() function
  - ✅ Speech/chat mode switching
  - ✅ Interactive and one-shot query support
  - ✅ Context management and file operations
  - ✅ Working directory commands (/dir=path)
- **Dependencies:** SessionManager, SpeechHandler
- **Priority:** HIGH

#### 5. **SpeechRecognition Module**
- **Status:** 🟢 Complete
- **Description:** Multi-backend voice input processing
- **Features:**
  - ✅ Google Speech API (online)
  - ✅ Whisper integration (offline)
  - ✅ Vosk support (lightweight offline)
  - ✅ Automatic fallback between backends
- **Dependencies:** speechrecognition, whisper, vosk
- **Priority:** MEDIUM

#### 6. **TextToSpeech Module**
- **Status:** 🟢 Complete
- **Description:** System TTS voice output
- **Features:**
  - ✅ pyttsx3 integration
  - ✅ Multiple system voice support
  - ✅ Configurable speech rate
  - ✅ Voice selection and testing
- **Dependencies:** pyttsx3
- **Priority:** MEDIUM

#### 7. **File Operations System**
- **Status:** 🟢 Partially Complete
- **Description:** Comprehensive file management
- **Features:**
  - List Directory/file
  - Find Regex in Directories/files
  - create, read, write, remove Directory/file
  - Modify partial file content
- **Dependencies:** None
- **Priority:** HIGH

#### 8. **Model Management**
- **Status:** 🟡 In Progress
- **Description:** AI model handling
- **Features:**
  - loads/constructor
  - input processing
  - output response generation
  - download/remove models
- **Dependencies:** None
- **Priority:** HIGH

#### 9. **Multi-Model System**
- **Status:** 🔴 To Do
- **Description:** Multiple model coordination
- **Features:**
  - Model → Load Chat capabilities
  - Model → Load Speech Recognition
  - Model → Load TextToSpeech
- **Dependencies:** Model, SpeechRecognition, TextToSpeech
- **Priority:** MEDIUM

#### 10. **Image Generation System**
- **Status:** 🔴 To Do
- **Description:** Extends Model for visual generation
- **Features:**
  - Image generation tasks
  - Visual content creation
- **Dependencies:** Model
- **Priority:** LOW

#### 11. **RAG (Retrieval-Augmented Generation)**
- **Status:** 🟡 In Progress
- **Description:** Knowledge base management
- **Features:**
  - make knowledge bases
  - remove knowledge bases
  - Vector search integration
- **Dependencies:** None
- **Priority:** MEDIUM

## Development Phases

### Phase 1: Foundation (✅ COMPLETED)
**Focus:** Stabilize core architecture
- ✅ Main application structure (router + business logic separation)
- ✅ Basic file operations
- ✅ Model management basics
- ✅ Complete Conversation system (unified chat/speech)
- ✅ Agent framework (creation, management, configuration)

### Phase 2: Speech Integration (✅ COMPLETED)
**Focus:** Voice capabilities
- ✅ SpeechRecognition module (Google/Whisper/Vosk)
- ✅ TextToSpeech module (pyttsx3 system voices)
- ✅ Speech Mode agents (unified with conversation)
- ✅ Voice-enabled conversations

### Phase 3: Advanced Features (🟡 In Progress)
**Focus:** Enhanced capabilities
- ✅ Multi-Model coordination
- ✅ Image Generation integration (interactive studio)
- 🟡 Advanced RAG features (basic implementation complete)
- [ ] API service mode
- 🟡 **NEXT:** Extract one-shot query handling, unify image mode into conversation

### Phase 4: Polish & Optimization
**Focus:** Performance and usability
- [ ] Performance optimization
- [ ] Error handling improvements
- [ ] Documentation completion
- [ ] Testing suite

## Implementation Priorities

### HIGH Priority (Phase 1)
1. **Main Application** - Central hub
2. **Agents System** - Core functionality
3. **Conversation System** - Essential interaction
4. **File Operations** - Complete existing features
5. **Model Management** - Stable model handling

### MEDIUM Priority (Phase 2-3)
1. **SpeechRecognition** - Voice input
2. **TextToSpeech** - Voice output  
3. **Speech Mode Agent** - Specialized voice agent
4. **Multi-Model System** - Coordination layer
5. **RAG System** - Knowledge enhancement

### LOW Priority (Phase 4)
1. **Image Generation** - Visual features
2. **API Service** - External integration
3. **Advanced optimization** - Performance tuning

## Technical Specifications

### Architecture Principles
- **Modular Design:** Each component is independent and replaceable
- **Clear Interfaces:** Well-defined APIs between components
- **Configuration-Driven:** TOML-based configuration system
- **Extensibility:** Easy to add new models and capabilities

### Integration Points
- **Main ↔ Agents:** Management and configuration
- **Agents ↔ Conversation:** Communication layer
- **Conversation ↔ Speech Modules:** Voice processing
- **All Components ↔ File Operations:** Data persistence
- **Agents ↔ Model:** AI processing
- **Model ↔ RAG:** Knowledge enhancement

## Progress Tracking

### Current Sprint (Conversation Unification)
- ✅ Complete Agent framework
- ✅ Enhanced Conversation system (unified chat/speech)
- ✅ Stable Model management
- ✅ TOML-based configuration system
- 🟡 **CURRENT:** Extract one-shot query handler from run_conversation
- 🟡 **NEXT:** Integrate image generation into unified conversation

### Next Sprint (Final Unification)
- ✅ Implement SpeechRecognition (multi-backend)
- ✅ Implement TextToSpeech (system voices)
- ✅ Create Speech Mode agents (SpeechHandler)
- ✅ Integrate voice capabilities
- 🟡 **TODO:** Create dedicated one-shot query function
- 🟡 **TODO:** Integrate image mode into run_conversation with mode switching

## Success Metrics
- All core components functional
- Speech mode working reliably
- File operations complete
- Configuration system stable
- Multi-agent management working
- RAG knowledge system operational

## Notes
- Focus on completing one component fully before moving to next
- Maintain backward compatibility where possible
- Document APIs as components are completed
- Regular testing of integrated functionality