#!/usr/bin/env python3
"""
Model Downloader for Voice Cloner App

Automatically downloads required RVC and Hubert models.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import logging
from tqdm import tqdm
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Download and setup models for voice cloning."""
    
    def __init__(self):
        self.base_dir = Path("models")
        self.config = self._load_config()
        
    def _load_config(self):
        """Load model configuration."""
        return {
            "hubert": {
                "base": {
                    "url": "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin",
                    "filename": "hubert_base.pt",
                    "size_mb": 95,
                    "required": True
                }
            },
            "rvc": {
                "default": {
                    "url": "https://huggingface.co/datasets/voice-cloning/rvc-models/resolve/main/default_model.pth",
                    "filename": "default_model.pth",
                    "size_mb": 150,
                    "required": False
                }
            }
        }
    
    def create_directories(self):
        """Create model directory structure."""
        directories = [
            "models/rvc",
            "models/hubert", 
            "models/config",
            "models/pretrained",
            "models/checkpoints"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def download_file(self, url: str, filepath: Path, expected_size_mb: int = None):
        """Download a file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify file size
            actual_size_mb = filepath.stat().st_size / (1024 * 1024)
            if expected_size_mb and actual_size_mb < expected_size_mb * 0.9:
                logger.warning(f"File size seems small: {actual_size_mb:.1f}MB (expected ~{expected_size_mb}MB)")
            
            logger.info(f"Downloaded: {filepath.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def download_hubert_models(self):
        """Download Hubert content encoder models."""
        logger.info("Downloading Hubert models...")
        
        hubert_dir = self.base_dir / "hubert"
        hubert_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_info in self.config["hubert"].items():
            filepath = hubert_dir / model_info["filename"]
            
            if filepath.exists():
                logger.info(f"Hubert model already exists: {filepath.name}")
                continue
            
            logger.info(f"Downloading {model_name} Hubert model...")
            success = self.download_file(
                model_info["url"], 
                filepath, 
                model_info["size_mb"]
            )
            
            if not success and model_info["required"]:
                logger.error(f"Required Hubert model failed to download: {model_name}")
                return False
        
        return True
    
    def download_rvc_models(self):
        """Download RVC voice conversion models."""
        logger.info("Downloading RVC models...")
        
        rvc_dir = self.base_dir / "rvc"
        rvc_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_info in self.config["rvc"].items():
            filepath = rvc_dir / model_info["filename"]
            
            if filepath.exists():
                logger.info(f"RVC model already exists: {filepath.name}")
                continue
            
            logger.info(f"Downloading {model_name} RVC model...")
            success = self.download_file(
                model_info["url"], 
                filepath, 
                model_info["size_mb"]
            )
            
            if not success and model_info["required"]:
                logger.error(f"Required RVC model failed to download: {model_name}")
                return False
        
        return True
    
    def create_sample_config(self):
        """Create sample configuration files."""
        config_dir = self.base_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # RVC config
        rvc_config = {
            "model_type": "rvc",
            "version": "2.0",
            "sample_rate": 22050,
            "hop_length": 512,
            "win_length": 2048,
            "n_fft": 2048,
            "n_mels": 80,
            "f0_min": 50,
            "f0_max": 1100,
            "f0_mel_min": 1127 * np.log(1 + 50 / 700),
            "f0_mel_max": 1127 * np.log(1 + 1100 / 700)
        }
        
        config_file = config_dir / "rvc_config.json"
        if not config_file.exists():
            with open(config_file, 'w') as f:
                json.dump(rvc_config, f, indent=2)
            logger.info("Created sample RVC config")
    
    def download_from_huggingface(self, repo_name: str, target_dir: str):
        """Download models from HuggingFace repositories."""
        try:
            import huggingface_hub
            
            logger.info(f"Downloading from HuggingFace: {repo_name}")
            
            # Download all files from the repository
            huggingface_hub.snapshot_download(
                repo_id=repo_name,
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Downloaded {repo_name} to {target_dir}")
            return True
            
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"Failed to download from HuggingFace: {e}")
            return False
    
    def list_available_models(self):
        """List available models for download."""
        logger.info("Available models for download:")
        
        for category, models in self.config.items():
            logger.info(f"\n{category.upper()} Models:")
            for name, info in models.items():
                status = "✅" if (self.base_dir / category / info["filename"]).exists() else "❌"
                required = " (Required)" if info["required"] else ""
                logger.info(f"  {status} {name}: {info['filename']}{required}")
    
    def download_all(self):
        """Download all required models."""
        logger.info("🚀 Starting model download...")
        
        # Create directories
        self.create_directories()
        
        # Download models
        hubert_success = self.download_hubert_models()
        rvc_success = self.download_rvc_models()
        
        # Create sample configs
        self.create_sample_config()
        
        if hubert_success and rvc_success:
            logger.info("✅ All models downloaded successfully!")
            self.list_available_models()
            return True
        else:
            logger.error("❌ Some models failed to download")
            return False
    
    def download_custom_model(self, url: str, filename: str, category: str = "rvc"):
        """Download a custom model from URL."""
        target_dir = self.base_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = target_dir / filename
        logger.info(f"Downloading custom model: {filename}")
        
        success = self.download_file(url, filepath)
        if success:
            logger.info(f"✅ Custom model downloaded: {filepath}")
        else:
            logger.error(f"❌ Failed to download custom model: {filename}")
        
        return success

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for Voice Cloner App")
    parser.add_argument("--all", action="store_true", help="Download all required models")
    parser.add_argument("--hubert", action="store_true", help="Download Hubert models only")
    parser.add_argument("--rvc", action="store_true", help="Download RVC models only")
    parser.add_argument("--custom", nargs=3, metavar=("URL", "FILENAME", "CATEGORY"), 
                       help="Download custom model: URL FILENAME CATEGORY")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--huggingface", nargs=2, metavar=("REPO", "TARGET_DIR"),
                       help="Download from HuggingFace: REPO_NAME TARGET_DIR")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.all:
        downloader.download_all()
    elif args.hubert:
        downloader.download_hubert_models()
    elif args.rvc:
        downloader.download_rvc_models()
    elif args.custom:
        url, filename, category = args.custom
        downloader.download_custom_model(url, filename, category)
    elif args.list:
        downloader.list_available_models()
    elif args.huggingface:
        repo, target_dir = args.huggingface
        downloader.download_from_huggingface(repo, target_dir)
    else:
        # Default: download all
        downloader.download_all()

if __name__ == "__main__":
    # Import numpy for config creation
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy not available, skipping config creation")
    
    main()

