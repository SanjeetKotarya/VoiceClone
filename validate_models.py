#!/usr/bin/env python3
"""
Model Validation Script for Voice Cloner App

Checks if all required models are properly installed and accessible.
"""

import os
import sys
from pathlib import Path
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    """Validate model installation and accessibility."""
    
    def __init__(self):
        self.base_dir = Path("models")
        self.required_models = {
            "hubert": {
                "hubert_base.pt": {
                    "min_size_mb": 90,
                    "description": "Hubert content encoder (required)"
                }
            },
            "rvc": {
                "default_model.pth": {
                    "min_size_mb": 100,
                    "description": "Default RVC model (recommended)"
                }
            }
        }
    
    def check_directory_structure(self):
        """Check if model directory structure exists."""
        logger.info("🔍 Checking directory structure...")
        
        required_dirs = [
            "models",
            "models/rvc",
            "models/hubert",
            "models/config",
            "models/pretrained",
            "models/checkpoints"
        ]
        
        missing_dirs = []
        for directory in required_dirs:
            if not Path(directory).exists():
                missing_dirs.append(directory)
                logger.error(f"❌ Missing directory: {directory}")
            else:
                logger.info(f"✅ Directory exists: {directory}")
        
        if missing_dirs:
            logger.warning("Some directories are missing. Creating them...")
            for directory in missing_dirs:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created: {directory}")
        
        return len(missing_dirs) == 0
    
    def validate_model_file(self, filepath: Path, min_size_mb: float, description: str):
        """Validate a single model file."""
        if not filepath.exists():
            logger.error(f"❌ Missing: {filepath.name} - {description}")
            return False
        
        try:
            file_size = filepath.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            if size_mb < min_size_mb:
                logger.warning(f"⚠️  Small file: {filepath.name} ({size_mb:.1f}MB < {min_size_mb}MB)")
                return False
            
            logger.info(f"✅ Valid: {filepath.name} ({size_mb:.1f}MB) - {description}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating {filepath.name}: {e}")
            return False
    
    def check_models(self):
        """Check all required models."""
        logger.info("🔍 Checking model files...")
        
        validation_results = {}
        total_models = 0
        valid_models = 0
        
        for category, models in self.required_models.items():
            category_dir = self.base_dir / category
            validation_results[category] = {}
            
            for filename, requirements in models.items():
                total_models += 1
                filepath = category_dir / filename
                
                is_valid = self.validate_model_file(
                    filepath, 
                    requirements["min_size_mb"], 
                    requirements["description"]
                )
                
                validation_results[category][filename] = {
                    "valid": is_valid,
                    "exists": filepath.exists(),
                    "size_mb": filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0,
                    "requirements": requirements
                }
                
                if is_valid:
                    valid_models += 1
        
        return validation_results, total_models, valid_models
    
    def check_config_files(self):
        """Check configuration files."""
        logger.info("🔍 Checking configuration files...")
        
        config_dir = self.base_dir / "config"
        config_files = ["rvc_config.json"]
        
        missing_configs = []
        for config_file in config_files:
            filepath = config_dir / config_file
            if not filepath.exists():
                missing_configs.append(config_file)
                logger.warning(f"⚠️  Missing config: {config_file}")
            else:
                logger.info(f"✅ Config exists: {config_file}")
        
        return len(missing_configs) == 0
    
    def generate_report(self, validation_results, total_models, valid_models):
        """Generate a validation report."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MODEL VALIDATION REPORT")
        logger.info("=" * 60)
        
        # Summary
        logger.info(f"Total Models: {total_models}")
        logger.info(f"Valid Models: {valid_models}")
        logger.info(f"Success Rate: {(valid_models/total_models*100):.1f}%" if total_models > 0 else "N/A")
        
        # Detailed results
        for category, models in validation_results.items():
            logger.info(f"\n{category.upper()} MODELS:")
            for filename, result in models.items():
                status = "✅" if result["valid"] else "❌"
                size_info = f"({result['size_mb']:.1f}MB)" if result["exists"] else "(missing)"
                logger.info(f"  {status} {filename} {size_info}")
        
        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS:")
        
        if valid_models == 0:
            logger.info("❌ No valid models found!")
            logger.info("   Run: python download_models.py --all")
        elif valid_models < total_models:
            logger.info("⚠️  Some models are missing or invalid")
            logger.info("   Run: python download_models.py --list")
        else:
            logger.info("🎉 All models are valid!")
            logger.info("   You can now run: python app.py")
        
        return valid_models == total_models
    
    def validate_all(self):
        """Run complete validation."""
        logger.info("🚀 Starting model validation...")
        
        # Check directory structure
        self.check_directory_structure()
        
        # Check models
        validation_results, total_models, valid_models = self.check_models()
        
        # Check config files
        self.check_config_files()
        
        # Generate report
        all_valid = self.generate_report(validation_results, total_models, valid_models)
        
        return all_valid

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Voice Cloner App models")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix missing directories")
    parser.add_argument("--report", action="store_true", help="Generate detailed report only")
    
    args = parser.parse_args()
    
    validator = ModelValidator()
    
    if args.report:
        # Just check existing models and report
        validation_results, total_models, valid_models = validator.check_models()
        validator.generate_report(validation_results, total_models, valid_models)
    else:
        # Full validation
        success = validator.validate_all()
        
        if success:
            logger.info("\n🎉 Model validation completed successfully!")
            return 0
        else:
            logger.error("\n❌ Model validation found issues!")
            logger.info("Run 'python download_models.py --all' to download missing models")
            return 1

if __name__ == "__main__":
    exit(main())

