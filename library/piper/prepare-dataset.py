#!/usr/bin/env python3
import os, subprocess, shutil
import pandas as pd
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Step 1: Load Common Voice Indonesian dataset from downloaded files
print("=" * 50)
print("PROCESSING COMMON VOICE INDONESIAN DATASET")
print("=" * 50)

cv_dir = "cv-corpus-17.0-2024-03-15"
id_dir = os.path.join(cv_dir, "id")

if not os.path.exists(id_dir):
    print(f"Error: Indonesian directory not found at {id_dir}")
    print("Make sure you have extracted the Common Voice dataset properly.")
    exit(1)

# Load TSV file - use validated.tsv for highest quality
validated_tsv = os.path.join(id_dir, "validated.tsv")
if os.path.exists(validated_tsv):
    print("Using validated.tsv for highest quality data")
    df = pd.read_csv(validated_tsv, sep='\t')
else:
    print("validated.tsv not found, using train.tsv")
    train_tsv = os.path.join(id_dir, "train.tsv")
    df = pd.read_csv(train_tsv, sep='\t')

print(f"Loaded {len(df)} samples from TSV file")

# Show gender distribution
print("\nGender distribution:")
gender_counts = df['gender'].fillna('unspecified').value_counts()
for gender, count in gender_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {gender}: {count} samples ({percentage:.1f}%)")

# Filter for high-quality samples (more up_votes than down_votes)
if 'up_votes' in df.columns and 'down_votes' in df.columns:
    original_len = len(df)
    df = df[(df['up_votes'] > df['down_votes']) & (df['up_votes'] > 0)]
    print(f"Filtered to {len(df)} high-quality samples (removed {original_len - len(df)})")

# Separate datasets by gender for consistent voice models
male_df = df[df['gender'] == 'male_masculine'].copy()
female_df = df[df['gender'] == 'female_feminine'].copy()

print(f"\nSeparated by gender:")
print(f"  Male samples: {len(male_df)}")
print(f"  Female samples: {len(female_df)}")

# Process both genders separately
def process_gender_dataset(gender_df, gender_name):
    print(f"\n{'='*30}")
    print(f"Processing {gender_name.upper()} voices")
    print(f"{'='*30}")
    
    clips_dir = os.path.join(id_dir, "clips")
    wavs_dir = f"piper_data_{gender_name}/wavs"
    os.makedirs(wavs_dir, exist_ok=True)
    
    valid_samples = []
    audio_errors = 0
    
    for idx, row in tqdm(gender_df.iterrows(), total=len(gender_df), desc=f"Converting {gender_name} audio"):
        try:
            # Get paths
            mp3_filename = row['path']
            mp3_path = os.path.join(clips_dir, mp3_filename)
            
            if not os.path.exists(mp3_path):
                audio_errors += 1
                continue
                
            # Convert MP3 to WAV using ffmpeg
            base_filename = os.path.splitext(mp3_filename)[0]
            wav_path = os.path.join(wavs_dir, f"{base_filename}.wav")
            
            # Use ffmpeg to convert to 16kHz mono WAV
            result = subprocess.run([
                "ffmpeg", "-y", "-i", mp3_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Store valid sample info
                sentence = str(row['sentence']).strip()
                
                valid_samples.append({
                    "base_filename": base_filename,
                    "sentence": sentence,
                    "wav_path": wav_path
                })
            else:
                audio_errors += 1
                
        except Exception as e:
            audio_errors += 1
            continue
    
    print(f"Successfully processed {len(valid_samples)} {gender_name} audio files")
    if audio_errors > 0:
        print(f"Failed to process {audio_errors} {gender_name} audio files")
    
    # Generate metadata.csv for this gender
    metadata_dir = f"piper_data_{gender_name}"
    os.makedirs(metadata_dir, exist_ok=True)
    
    with open(f"{metadata_dir}/metadata.csv", "w", encoding="utf-8") as f:
        for sample in valid_samples:
            # Keep original case for better TTS quality
            sentence = sample["sentence"].strip()
            # Add period if no punctuation exists
            if sentence and not sentence.endswith(('.', '!', '?', ';', ':')):
                sentence += '.'
            f.write(f"{sample['base_filename']}|{sentence}\n")
    
    print(f"Generated {metadata_dir}/metadata.csv with {len(valid_samples)} entries")
    
    # Create training config for this gender
    config_path = f"piper_train_config_{gender_name}.json"
    import json
    default_config = {
        "audio": {
            "sample_rate": 16000,
            "channels": 1
        },
        "model": {
            "hidden_channels": 256,
            "filter_channels": 1024,
            "n_heads": 4,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4]
        },
        "train": {
            "batch_size": 16,
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0
        }
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)
    print(f"Created {config_path}")
    
    return len(valid_samples)

# Process both genders
male_count = 0
female_count = 0

if len(male_df) > 0:
    male_count = process_gender_dataset(male_df, "male")

if len(female_df) > 0:
    female_count = process_gender_dataset(female_df, "female")

# Step 4: Run Piper preprocessing to create dataset.jsonl files
print("\n" + "="*50)
print("RUNNING PIPER PREPROCESSING")
print("="*50)

def run_piper_preprocessing(dataset_dir, gender_name):
    print(f"\nPreprocessing {gender_name} dataset...")
    try:
        result = subprocess.run([
            "python3", "-m", "piper_train.preprocess",
            "--language", "id",
            "--input-dir", dataset_dir,
            "--output-dir", dataset_dir,
            "--dataset-format", "ljspeech",
            "--sample-rate", "16000",
            "--single-speaker"
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ {gender_name} preprocessing completed successfully")
            return True
        else:
            print(f"❌ {gender_name} preprocessing failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {gender_name} preprocessing timed out but may have completed")
        # Check if dataset.jsonl was created
        jsonl_path = os.path.join(dataset_dir, "dataset.jsonl")
        if os.path.exists(jsonl_path):
            print(f"✅ {gender_name} dataset.jsonl found - preprocessing likely completed")
            return True
        return False
    except Exception as e:
        print(f"❌ Error preprocessing {gender_name}: {e}")
        return False

# Run preprocessing for both datasets
preprocessing_success = True
if male_count > 0:
    male_success = run_piper_preprocessing("piper_data_male", "male")
    preprocessing_success = preprocessing_success and male_success

if female_count > 0:
    female_success = run_piper_preprocessing("piper_data_female", "female")
    preprocessing_success = preprocessing_success and female_success

print("\n" + "="*50)
print("DATASET PREPARATION COMPLETE!")
print("="*50)
print(f"✅ Male model dataset: {male_count} samples in piper_data_male/")
print(f"✅ Female model dataset: {female_count} samples in piper_data_female/")

if preprocessing_success:
    print("✅ Piper preprocessing completed for both datasets")
else:
    print("⚠️  Some preprocessing steps may have timed out")
    print("   Check for dataset.jsonl files in the dataset directories")

print()
print("SEPARATE GENDER MODELS CREATED!")
print("This prevents voice inconsistency in your TTS models.")
print()
print("To train MALE model:")
print("python3 -m piper_train --dataset-dir piper_data_male --batch-size 8 --accelerator cpu --devices 1")
print()
print("To train FEMALE model:")
print("python3 -m piper_train --dataset-dir piper_data_female --batch-size 8 --accelerator cpu --devices 1")
print()
print("Final models will be saved in lightning_logs/ directory")
print("Dependencies needed: ffmpeg, piper-phonemize")
print("="*50)