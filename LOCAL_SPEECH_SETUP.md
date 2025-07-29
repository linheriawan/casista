# Local Speech Recognition Setup

This guide helps you set up local speech recognition alternatives to avoid depending on Google's online service.

## Option 1: OpenAI Whisper (Recommended)

**Best accuracy, completely offline**

### Installation
```bash
pip install openai-whisper
```

### Model Sizes
- `tiny` - ~39 MB, fastest, lowest accuracy
- `base` - ~74 MB, good balance (default)
- `small` - ~244 MB, better accuracy
- `medium` - ~769 MB, great accuracy  
- `large` - ~1550 MB, best accuracy

### Setup
```bash
# Set Whisper as backend for your assistant
python3 main.py --set-speech-backend jeany whisper

# Optional: Set specific model size (in assistant config)
# Edit .ai_context/jeany/config.json:
# "whisper_model": "small"
```

### First Run
On first use, Whisper will download the selected model (~74MB for base).

---

## Option 2: Vosk 

**Fast, real-time recognition**

### Installation
```bash
pip install vosk
```

### Model Download
```bash
# Download a model (choose your language)
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
mv vosk-model-en-us-0.22 vosk-model
```

### Setup
```bash
# Set Vosk as backend
python3 main.py --set-speech-backend jeany vosk

# Make sure model path is correct in config:
# "vosk_model_path": "./vosk-model"
```

---

## Comparison

| Backend | Type | Accuracy | Speed | Size | Internet |
|---------|------|----------|-------|------|----------|
| Google | Online | ⭐⭐⭐⭐⭐ | Fast | 0MB | Required |
| Whisper | Local | ⭐⭐⭐⭐⭐ | Slow | 74MB-1.5GB | None |
| Vosk | Local | ⭐⭐⭐ | Fast | ~50MB | None |

## Usage Commands

```bash
# List available backends
python3 main.py --list-speech-backends

# Switch to Whisper (local, best accuracy)
python3 main.py --set-speech-backend jeany whisper

# Switch to Vosk (local, fast)
python3 main.py --set-speech-backend jeany vosk

# Switch back to Google (online)
python3 main.py --set-speech-backend jeany google

# Check current config
python3 main.py --config jeany speech
```

## Whisper Model Management

The first time you use each model size, it will be downloaded to `~/.cache/whisper/`. You can pre-download models:

```python
import whisper
whisper.load_model("base")  # Downloads base model
whisper.load_model("small") # Downloads small model
```

## Troubleshooting

### Whisper Issues
- **Slow first run**: Model is downloading, be patient
- **CUDA errors**: Install CPU-only version: `pip install --upgrade --no-deps --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu`
- **Memory issues**: Use smaller model (`tiny` or `base`)

### Vosk Issues  
- **Model not found**: Check `vosk_model_path` in assistant config
- **Poor accuracy**: Try a larger model from https://alphacephei.com/vosk/models
- **Audio errors**: Ensure pyaudio is installed: `pip install pyaudio`

### General Audio Issues
- **No microphone**: Check System Preferences > Privacy & Security > Microphone
- **Permission denied**: Grant microphone access to Terminal/Python
- **Audio quality**: Speak clearly, reduce background noise

## Privacy Benefits

**Local speech recognition means:**
- ✅ No data sent to Google/internet
- ✅ Works offline completely  
- ✅ Faster processing (no network delay)
- ✅ Better privacy/security
- ✅ No API rate limits

**Whisper is recommended** for best accuracy while staying completely local.