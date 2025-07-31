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
