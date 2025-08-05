# Indonesian Piper TTS Training Guide

This guide walks you through training Indonesian text-to-speech models using Mozilla Common Voice 17.0 dataset and Piper TTS.

## Overview

This setup creates **two separate TTS models**:
- **Male Indonesian voice** - Consistent masculine voice
- **Female Indonesian voice** - Consistent feminine voice

This prevents voice inconsistency that occurs when mixing genders in a single model.

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.9-3.11 (Python 3.11 recommended)
- **Storage**: ~10GB free space for dataset and models
- **RAM**: 8GB minimum (16GB recommended)
- **CPU**: Multi-core processor (training will be slow on CPU)

### Required Software
- **FFmpeg** - For audio conversion
- **Git** - For cloning repositories
- **Python 3** - System Python installation

## Step 1: Environment Setup

### Install System Dependencies

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg

# Install espeak-ng (for phonemization)
brew install espeak
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y ffmpeg git python3 python3-pip espeak-ng espeak-ng-data libespeak-ng-dev
```

**Windows:**
```bash
# Install FFmpeg from https://ffmpeg.org/download.html
# Install Git from https://git-scm.com/download/win
# Install Python from https://python.org/downloads/
```

### Install Python Dependencies

```bash
# Install core dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Lightning (compatible version)
pip3 install "pytorch-lightning>=1.9.0,<2.0.0"

# Install training dependencies
pip3 install pandas tqdm soundfile librosa onnxruntime tensorboard phonemizer numpy scipy matplotlib tensorboardX Unidecode inflect webrtcvad jiwer numba Cython

# Install Piper phonemizer
pip3 install piper-phonemize
```

## Step 2: Download Dataset

### Option A: Download from Common Voice Website (Recommended)

1. Go to https://commonvoice.mozilla.org/en/datasets
2. Find **Indonesian** in the language list
3. Click **"Download Dataset"** and accept the terms
4. Download the `.tar.gz` file (approximately 2-3GB)
5. Extract it to your project directory:

```bash
tar -xzf cv-corpus-17.0-*-id.tar.gz
```

### Option B: Alternative Dataset Sources

If Common Voice is unavailable, you can use other Indonesian speech datasets, but you'll need to format them as:
- WAV files (16kHz, mono)
- metadata.csv with format: `filename|text`

## Step 3: Clone Piper Training Code

```bash
# Clone the Piper repository
git clone https://github.com/rhasspy/piper.git
cd piper/src/python

# Build the monotonic alignment extension
cd piper_train/vits/monotonic_align
python3 setup.py build_ext --inplace
cd ../../..

# Install Piper training module
pip3 install -e . --no-deps
```

## Step 4: Prepare Dataset

### Using the Automated Script

```bash
# Make sure you're in the main project directory
cd /path/to/your/project

# Run the dataset preparation script
python3 prepare-dataset.py
```

The script will:
1. **Load** the Common Voice Indonesian dataset
2. **Filter** for high-quality samples (positive vote ratios)
3. **Separate** male and female voices into different datasets
4. **Convert** MP3 files to 16kHz WAV format
5. **Generate** metadata.csv files
6. **Run** Piper preprocessing to create dataset.jsonl files
7. **Create** training configurations

### Manual Preparation (Alternative)

If the automated script fails, you can prepare manually:

```bash
# For female dataset
cd /path/to/your/project
python3 -m piper_train.preprocess \
  --language id \
  --input-dir piper_data_female \
  --output-dir piper_data_female \
  --dataset-format ljspeech \
  --sample-rate 16000 \
  --single-speaker

# For male dataset  
python3 -m piper_train.preprocess \
  --language id \
  --input-dir piper_data_male \
  --output-dir piper_data_male \
  --dataset-format ljspeech \
  --sample-rate 16000 \
  --single-speaker
```

## Step 5: Training

### Training Parameters

- **Default Epochs**: 1000 (can take 12-48 hours on CPU)
- **Batch Size**: 8 (for CPU), 16-32 (for GPU)
- **Models**: Separate training for male and female voices

### Start Training

**Train Female Model:**
```bash
python3 -m piper_train \
  --dataset-dir piper_data_female \
  --batch-size 8 \
  --accelerator cpu \
  --devices 1 \
  --max_epochs 1000
```

**Train Male Model:**
```bash
python3 -m piper_train \
  --dataset-dir piper_data_male \
  --batch-size 8 \
  --accelerator cpu \
  --devices 1 \
  --max_epochs 1000
```

### Training Options

**Faster Training (Fewer Epochs):**
```bash
--max_epochs 100  # Quick test training
--max_epochs 500  # Reasonable quality
--max_epochs 1000 # Full quality (default)
```

**GPU Training (If Available):**
```bash
--accelerator gpu \
--devices 1 \
--batch-size 16
```

**Early Stopping:**
```bash
--max_time "12:00:00"  # Stop after 12 hours
```

## Step 6: Monitor Training

### Training Progress

Training progress is displayed in the console:
```
Epoch 1/1000: 100%|██████████| 1250/1250 [45:23<00:00,  2.18s/it, loss=2.45, v_num=0]
```

### TensorBoard (Optional)

```bash
# Install tensorboard if not already installed
pip3 install tensorboard

# View training logs
tensorboard --logdir lightning_logs
```

Open http://localhost:6006 in your browser to see training graphs.

### Training Output

Models are saved in:
```
lightning_logs/
├── version_0/           # Female model training
│   └── checkpoints/
│       └── epoch=999-step=X.ckpt
└── version_1/           # Male model training  
    └── checkpoints/
        └── epoch=999-step=Y.ckpt
```

## Step 7: Export Models

After training, convert checkpoints to ONNX format:

```bash
# Export female model
python3 -m piper_train.export_onnx \
  lightning_logs/version_0/checkpoints/epoch=999-step=X.ckpt \
  piper_model_female.onnx

# Export male model  
python3 -m piper_train.export_onnx \
  lightning_logs/version_1/checkpoints/epoch=999-step=Y.ckpt \
  piper_model_male.onnx
```

## Step 8: Test Your Models

### Install Piper for Inference

```bash
# Download Piper binary or install via pip
pip3 install piper-tts
```

### Test the Models

```bash
# Test female voice
echo "Selamat pagi! Bagaimana kabar Anda hari ini?" | piper \
  --model piper_model_female.onnx \
  --output_file test_female.wav

# Test male voice
echo "Selamat pagi! Bagaimana kabar Anda hari ini?" | piper \
  --model piper_model_male.onnx \
  --output_file test_male.wav
```

## Troubleshooting

### Common Issues

**1. "No module named 'piper_phonemize'"**
```bash
pip3 install piper-phonemize
```

**2. "monotonic_align module not found"**
```bash
cd piper/src/python/piper_train/vits/monotonic_align
python3 setup.py build_ext --inplace
```

**3. "PyTorch Lightning version conflict"**
```bash
pip3 install "pytorch-lightning>=1.9.0,<2.0.0" --force-reinstall
```

**4. Out of Memory during training**
```bash
# Reduce batch size
--batch-size 4
# Or use gradient accumulation
--accumulate_grad_batches 2
```

**5. Training too slow**
```bash
# Reduce epochs for testing
--max_epochs 100
# Or use fewer samples
--limit_train_batches 0.1  # Use only 10% of data
```

### Performance Tips

1. **Use GPU if available** - 10-50x faster than CPU
2. **Increase batch size** on powerful machines
3. **Use SSD storage** for faster data loading
4. **Close other applications** to free up RAM
5. **Monitor training loss** - should decrease over time

### Quality Tips

1. **More epochs = better quality** (up to a point)
2. **Larger datasets = better generalization**
3. **Clean data = better results** (the script already filters for quality)
4. **Single gender = consistent voice** (this setup already does this)

## Expected Results

### Training Time
- **CPU Training**: 12-48 hours per model
- **GPU Training**: 2-8 hours per model  

### Model Quality
- **100 epochs**: Basic intelligibility
- **500 epochs**: Good quality
- **1000 epochs**: High quality (recommended)

### Final Models
- **Female Model**: ~50MB ONNX file
- **Male Model**: ~50MB ONNX file
- **Both models**: Natural Indonesian speech synthesis

## Directory Structure

After completion, your project should look like:

```
project/
├── GUIDES.md
├── prepare-dataset.py
├── cv-corpus-17.0-2024-03-15/     # Downloaded dataset
├── piper/                         # Piper training code
├── piper_data_female/            # Female training data
│   ├── wavs/
│   ├── metadata.csv
│   ├── dataset.jsonl
│   └── config.json
├── piper_data_male/              # Male training data
│   ├── wavs/
│   ├── metadata.csv  
│   ├── dataset.jsonl
│   └── config.json
├── lightning_logs/               # Training checkpoints
└── piper_model_*.onnx           # Final models
```

## Support

For issues:
1. Check the troubleshooting section above
2. Review Piper documentation: https://github.com/rhasspy/piper
3. Check Common Voice dataset: https://commonvoice.mozilla.org/

---

**Happy Training! 🎉**

Your Indonesian TTS models will provide natural, consistent speech synthesis for both male and female voices.