# Models Directory

This directory contains all the RVC models and related files for voice cloning.

## 📁 Directory Structure

```
models/
├── rvc/                    # RVC model files (.pth)
│   ├── default_model.pth
│   ├── high_quality.pth
│   └── fast_model.pth
├── hubert/                 # Hubert content encoder models
│   ├── hubert_base.pt
│   └── hubert_large.pt
├── config/                 # Model configuration files
│   ├── rvc_config.json
│   └── model_settings.yaml
├── pretrained/            # Pre-trained model weights
│   └── speaker_encoder.pth
├── checkpoints/           # Training checkpoints
│   └── latest_checkpoint.pth
└── README.md              # This file
```

## 🎯 Model Types

### RVC Models (.pth files)
- **Location**: `models/rvc/`
- **Purpose**: Main voice conversion models
- **Format**: PyTorch model files
- **Naming**: `{model_name}.pth`

### Hubert Models (.pt files)
- **Location**: `models/hubert/`
- **Purpose**: Content encoder for speech recognition
- **Format**: PyTorch model files
- **Required**: `hubert_base.pt` (minimum)

### Configuration Files
- **Location**: `models/config/`
- **Purpose**: Model settings and parameters
- **Format**: JSON, YAML, or Python config files

## 📥 Adding New Models

### 1. Download RVC Models
```bash
# Example: Download from HuggingFace
git lfs install
git clone https://huggingface.co/username/rvc-model-name
cp rvc-model-name/*.pth models/rvc/
```

### 2. Download Hubert Models
```bash
# Download Hubert base model
wget https://huggingface.co/facebook/hubert-base-ls960/resolve/main/pytorch_model.bin
mv pytorch_model.bin models/hubert/hubert_base.pt
```

### 3. Update Configuration
Edit `config.py` to add your new model:
```python
AVAILABLE_MODELS = {
    "my_model": {
        "name": "My Custom RVC Model",
        "description": "Custom trained model",
        "quality": "high",
        "speed": "medium"
    }
}
```

## 🔧 Model Requirements

### Minimum Requirements
- **RVC Model**: At least one `.pth` file in `models/rvc/`
- **Hubert Model**: `hubert_base.pt` in `models/hubert/`

### Recommended
- **Multiple RVC Models**: For different voice styles
- **Configuration Files**: For model-specific settings
- **Speaker Encoder**: For better voice quality

## 📚 Model Sources

### Official RVC Models
- [RVC Official Repository](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [HuggingFace RVC Models](https://huggingface.co/models?search=rvc)

### Hubert Models
- [Facebook Hubert](https://github.com/pytorch/fairseq/tree/main/examples/hubert)
- [HuggingFace Hubert](https://huggingface.co/facebook/hubert-base-ls960)

### Community Models
- [Civitai](https://civitai.com/) - Search for "RVC"
- [HuggingFace](https://huggingface.co/) - Search for "voice conversion"

## ⚠️ Important Notes

1. **Model Size**: RVC models can be 100MB-1GB+
2. **Compatibility**: Ensure models match your RVC version
3. **Licensing**: Check model licenses before commercial use
4. **Performance**: Larger models = better quality but slower processing

## 🚀 Quick Start

1. **Download Models**: Place in appropriate directories
2. **Update Config**: Add model names to `AVAILABLE_MODELS`
3. **Restart App**: Reload the voice cloner application
4. **Select Model**: Choose from dropdown in web interface

## 🔍 Troubleshooting

### Model Not Found
- Check file paths in `config.py`
- Ensure model files are in correct directories
- Verify file permissions

### Model Loading Errors
- Check PyTorch version compatibility
- Verify model file integrity
- Check available GPU memory

### Poor Quality
- Try different RVC models
- Ensure target voice is clear (10-30 seconds)
- Check source audio quality

---

**Happy Voice Cloning! 🎤✨**

