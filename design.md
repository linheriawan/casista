Project Structure:

main.py    -->     called by by .local/bin/coder
├── library
│   ├── coding
│   ├── conversation
│   ├── image generation
├── configuration
│   ├── model traits (system_prompt,personalities)
│   ├── system config (model directory)
├── helper
│   ├── manage agent
│   │   ├── create agent
│   │   ├── set agent to ai model
│   │   ├── set agent voice
│   │   ├── set agent voice
│   │   ├── set agent RAG
│   ├── manage voice
│   │   ├── list local tts / voice
│   │   ├── list available Speech Regognition model
│   │   ├── download Speech Regognition model
│   │   ├── remove Speech Regognition model
│   ├── manage model
│   │   ├── set huggingface cache dir
│   │   ├── list ai model
│   │   ├── download model
│   │   ├── remove model
│   ├── RAG
│   │   ├── make RAG knowledge
│   │   ├── list RAG knowledge
│   │   ├── update RAG knowledge
│   │   ├── remove RAG knowledge
requirements.txt     --> pip install
setup.py
install.py

todo:
You're absolutely right! There are several issues with the current speech implementation:
1. Not using configured speech backend: It's hardcoded to use Google instead of your configured Whisper
2. Not using configured voice: It's not properly applying voice ID 31
3. Missing config display: The info box should show current speech settings
4. Speaking reasoning: The <think> tags should be filtered out from TTS