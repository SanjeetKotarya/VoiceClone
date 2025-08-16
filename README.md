# Voice Cloner App

A web-based voice cloning application using RVC (Retrieval-based Voice Conversion) technology.

## Features

- **Source Audio Upload**: Upload any audio file containing speech to convert
- **Target Voice Sample**: Upload a 10-30 second clip of the target voice
- **RVC Model Integration**: Advanced voice conversion using retrieval-based methods
- **Real-time Processing**: Convert voices with high quality output
- **Web Interface**: User-friendly Gradio interface
- **Audio Preview**: Listen to results directly in the browser
- **Download Support**: Save converted audio files
- **Multiple Models**: Support for different RVC models and quality levels

## App Flow

1. **Upload**: User uploads source audio and target voice sample
2. **Processing**: Backend extracts content and speaker embeddings
3. **Conversion**: RVC model generates new audio with target voice
4. **Output**: Downloadable converted audio with browser preview

## Installation

### 1. Clone and Setup
```bash
git clone <your-repo>
cd CLone
pip install -r requirements.txt
```

### 2. Download Models (Required)
```bash
# Windows: Double-click download_models.bat
# Or manually:
python download_models.py --all
```

### 3. Install FFmpeg
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

## Usage

### Quick Start
1. **Windows Users**: Double-click `start_app.bat`
2. **Others**: Run `python app.py`
3. **Open browser** to `http://localhost:7860`
4. **Upload files** and click "Convert Voice"

### Model Management
```bash
# List available models
python download_models.py --list

# Download specific models
python download_models.py --hubert    # Hubert models only
python download_models.py --rvc       # RVC models only

# Download custom model
python download_models.py --custom "URL" "filename.pth" "rvc"

# Download from HuggingFace
python download_models.py --huggingface "username/repo" "target_dir"
```

## File Structure

```
CLone/
├── app.py                 # Main Gradio web application
├── voice_converter.py     # RVC voice conversion logic
├── audio_processor.py     # Audio processing utilities
├── config.py             # Configuration settings
├── batch_processor.py     # Batch processing script
├── download_models.py     # Model downloader script
├── test_setup.py         # Setup verification script
├── requirements.txt       # Python dependencies
├── start_app.bat         # Windows startup script
├── download_models.bat   # Windows model downloader
├── README.md             # Project documentation
├── SETUP_GUIDE.md        # Detailed setup guide
├── uploads/              # Temporary upload storage
├── outputs/              # Generated audio files
├── models/               # RVC models and configurations
│   ├── rvc/             # RVC model files (.pth)
│   ├── hubert/          # Hubert content encoder (.pt)
│   ├── config/          # Model configuration files
│   ├── pretrained/      # Pre-trained weights
│   └── checkpoints/     # Training checkpoints
└── temp/                 # Temporary processing files
```

## Model Requirements

### Essential Models
- **Hubert Base**: Content encoder for speech recognition
- **RVC Model**: Voice conversion model (at least one)

### Model Sources
- **Official RVC**: [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- **HuggingFace**: [RVC Models](https://huggingface.co/models?search=rvc)
- **Community**: [Civitai](https://civitai.com/) (search "RVC")

## Technical Details

- **RVC Model**: Retrieval-based Voice Conversion for high-quality voice cloning
- **Audio Processing**: Librosa and PyDub for audio manipulation
- **Web Interface**: Gradio for easy deployment and sharing
- **Model Management**: Support for multiple voice models and configurations
- **Content Encoder**: Hubert models for linguistic content extraction
- **Speaker Embedding**: MFCC-based voice characteristics extraction

## Notes

- Target voice samples should be 10-30 seconds for best results
- Source audio quality affects output quality
- Processing time depends on audio length and model complexity
- Ensure sufficient disk space for models and temporary files
- Models can be 100MB-1GB+ in size

## Troubleshooting

### Common Issues
```bash
# Test setup
python test_setup.py

# Check models
python download_models.py --list

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Model Issues
- Ensure models are in correct directories
- Check model file integrity
- Verify PyTorch compatibility
- Check available disk space

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please respect the licenses of included models and dependencies.

---

**Happy Voice Cloning! 🎤✨**
