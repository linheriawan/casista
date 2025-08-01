# System Design: Best Practices Implementation

## System Prompt & Context Management

### ✅ **Best Practice: System Prompt Handling**

**System Prompt Flow:**
1. **Set Once**: System prompt established at conversation start
2. **Not Displayed**: Internal instruction, never shown to user
3. **Persistent**: Stays active throughout entire conversation
4. **Role Establishment**: Model knows its role/personality/capabilities

**Implementation:**
```python
# System prompt automatically managed by ChatManager
messages = context_manager.ensure_system_prompt(system_prompt)
# Result: [{"role": "system", "content": "You are Anna, a teacher..."}, ...]
```

**User Experience:**
```
User: "What's 2+2?"
Display: 🤖 [Anna]: 4! Let me explain... 
Hidden: Model received system prompt defining Anna as teacher
```

### ✅ **Best Practice: Context.json - Conversation Memory**

**Context.json Structure:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Anna, a patient teacher...",
      "timestamp": "2025-08-01T..."
    },
    {
      "role": "user", 
      "content": "What's machine learning?",
      "timestamp": "2025-08-01T..."
    },
    {
      "role": "assistant",
      "content": "Machine learning is...",
      "parsed_sections": {
        "reasoning": "I should explain this step by step...",
        "clean_answer": "Machine learning is..."
      },
      "timestamp": "2025-08-01T..."
    }
  ],
  "assistant_name": "anna",
  "last_updated": "2025-08-01T..."
}
```

**Context Loading Benefits:**
- **Session Continuity**: Resume conversations after app restart
- **Model Memory**: AI remembers previous exchanges
- **Context Awareness**: Model can refer to earlier topics
- **Learning**: Model adapts to user's communication style

**Example:**
```
Session 1:
User: "I'm learning Python"
Anna: "Great! Python is excellent for beginners..."

[App closes/restarts]

Session 2:
User: "What about functions?"  
Anna: "Since you're learning Python, functions are..."
      ↑ Anna remembers Python context from previous session
```

## Message Flow Architecture

### **Complete Message Flow:**

1. **User Input**: `"What's machine learning?"`

2. **System Context Setup** (First time only):
   ```python
   # ContextManager.ensure_system_prompt()
   messages = [
     {"role": "system", "content": "You are Anna, a teacher..."}
   ]
   ```

3. **Load Conversation History**:
   ```python
   # ContextManager.load_context()
   messages = [
     {"role": "system", "content": "..."},
     {"role": "user", "content": "Previous question"},
     {"role": "assistant", "content": "Previous answer"},
     # ... more history
   ]
   ```

4. **Add User Message**:
   ```python
   messages.append({
     "role": "user", 
     "content": "What's machine learning?",
     "timestamp": "..."
   })
   ```

5. **Send to Model**:
   ```python
   # Model receives complete context:
   # [system_prompt, conversation_history..., new_user_message]
   response = ollama_client.generate_response(messages)
   ```

6. **Model Generates** (with streaming):
   ```
   Raw: "<think>User wants to learn ML basics</think>Machine learning is..."
   ```

7. **Parse Response**:
   ```python
   parsed = {
     "content": "Full response with <think> tags",
     "clean_answer": "Machine learning is...",  
     "reasoning": "User wants to learn ML basics",
     "has_reasoning": True
   }
   ```

8. **Display to User**:
   ```
   🤖 [Anna]: Machine learning is...
   💭 Reasoning: User wants to learn ML basics
   ```

9. **Save to Context**:
   ```python
   # Updated context.json includes new exchange
   # Model will remember this in next interaction
   ```

## Configuration-Driven Behavior

### **Streaming Configuration:**
```toml
[streaming]
show_placeholder_for_reasoning = true  # Show "Thinking..." for reasoning models
placeholder_text = "🤖 Thinking..."    # Custom placeholder
show_raw_stream = false               # Don't show <think> tags during stream  
```

### **Parsing Configuration:**
```toml
[parsing]
extract_reasoning = true
reasoning_patterns = [
  {tag = "think", type = "reasoning", strip_tags = true}
]
```

### **Model Behavior Examples:**

#### **Qwen3 (Reasoning Model):**
- **During Stream**: Shows `"🤖 Thinking..."` 
- **After Complete**: Shows clean answer + reasoning
- **TTS**: Speaks only clean answer
- **Context**: Stores full response + parsed sections

#### **Llama3 (Normal Model):**
- **During Stream**: Shows real-time response
- **After Complete**: No additional display
- **TTS**: Speaks full response
- **Context**: Stores response normally

## Benefits Achieved

### ✅ **System Prompt Best Practices:**
- **Set once** at conversation start
- **Never displayed** to user (internal instruction)
- **Persistent context** throughout conversation
- **Proper role establishment** for AI

### ✅ **Context Memory:**
- **Conversation continuity** across sessions
- **Model awareness** of conversation history  
- **Context.json persistence** enables memory
- **Automatic loading** on session start

### ✅ **Enhanced User Experience:**
- **Clean display** (no duplicate responses)
- **Reasoning visibility** when available
- **Speech-friendly** output (no reasoning in TTS)
- **Configurable behavior** per assistant

### ✅ **Technical Excellence:**
- **Structured responses** with parsing
- **Configuration-driven** behavior
- **Model-agnostic** architecture
- **Backward compatible** implementation

## Summary

The system now properly implements AI conversation best practices:

1. **System prompts** establish model context without user display
2. **Context.json** provides persistent conversation memory
3. **Structured parsing** handles reasoning models elegantly  
4. **Configuration** allows per-assistant customization
5. **Clean UX** with no duplicate displays or TTS reasoning

This creates a professional, memory-enabled AI assistant that maintains context, handles reasoning models properly, and provides an excellent user experience.