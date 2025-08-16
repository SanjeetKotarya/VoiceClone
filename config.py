"""
Configuration file for Voice Cloner App
"""

import os
from typing import Dict, Any

# Audio processing settings
AUDIO_CONFIG = {
    "target_sample_rate": 22050,
    "hop_length": 512,
    "win_length": 2048,
    "n_fft": 2048,
    "n_mels": 80,
    "n_mfcc": 13,
    "silence_threshold_db": -40.0,
    "max_audio_duration": 300,  # 5 minutes
    "min_audio_duration": 1.0,  # 1 second
}

# RVC model settings
RVC_CONFIG = {
    "default_model": "default",
    "model_path": "models/",
    "device": "cpu",  # or "cuda" for GPU
    "batch_size": 1,
    "inference_steps": 50,
}

# File handling settings
FILE_CONFIG = {
    "upload_dir": "uploads/",
    "output_dir": "outputs/",
    "temp_dir": "temp/",
    "supported_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
    "output_format": "wav",
    "max_file_size_mb": 100,
}

# Web interface settings
WEB_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": False,
    "show_error": True,
    "quiet": False,
    "max_file_size": "100MB",
}

# Available models
AVAILABLE_MODELS = {
    "default": {
        "name": "Default RVC Model",
        "description": "Balanced quality and speed",
        "quality": "medium",
        "speed": "medium"
    },
    "high_quality": {
        "name": "High Quality RVC Model",
        "description": "Best quality, slower processing",
        "quality": "high",
        "speed": "slow"
    },
    "fast": {
        "name": "Fast Processing Model",
        "description": "Fast processing, good quality",
        "quality": "good",
        "speed": "fast"
    }
}

# Processing parameters
PROCESSING_CONFIG = {
    "voice_conversion_strength": 0.1,
    "content_extraction_method": "mel_spectrogram",
    "speaker_embedding_method": "mfcc_statistics",
    "normalization_method": "rms",
    "segmentation_length": 10,  # seconds
}

# Model structure paths
MODEL_PATHS = {
    "rvc_models": "models/rvc/",
    "hubert_models": "models/hubert/",
    "config_files": "models/config/",
    "pretrained": "models/pretrained/",
    "checkpoints": "models/checkpoints/"
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "audio": AUDIO_CONFIG,
        "rvc": RVC_CONFIG,
        "file": FILE_CONFIG,
        "web": WEB_CONFIG,
        "models": AVAILABLE_MODELS,
        "processing": PROCESSING_CONFIG,
        "model_paths": MODEL_PATHS
    }

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        FILE_CONFIG["upload_dir"],
        FILE_CONFIG["output_dir"],
        FILE_CONFIG["temp_dir"],
        RVC_CONFIG["model_path"]
    ]
    
    # Add model subdirectories
    for path in MODEL_PATHS.values():
        directories.append(path)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def validate_config() -> bool:
    """Validate configuration settings."""
    try:
        # Check audio settings
        if AUDIO_CONFIG["target_sample_rate"] <= 0:
            return False
        
        # Check file settings
        if not FILE_CONFIG["supported_formats"]:
            return False
        
        # Check web settings
        if WEB_CONFIG["server_port"] < 1 or WEB_CONFIG["server_port"] > 65535:
            return False
        
        return True
    except Exception:
        return False

def update_config(key_path: str, value: Any):
    """
    Update configuration value using dot notation.
    
    Args:
        key_path: Path to config key (e.g., "audio.target_sample_rate")
        value: New value to set
    """
    keys = key_path.split(".")
    config = globals()
    
    for key in keys[:-1]:
        if key in config and isinstance(config[key], dict):
            config = config[key]
        else:
            raise KeyError(f"Invalid config path: {key_path}")
    
    config[keys[-1]] = value
