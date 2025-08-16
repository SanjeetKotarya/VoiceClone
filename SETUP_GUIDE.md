# Voice Cloner App - Setup Guide

This guide will help you set up and run the Voice Cloner App on your system.

## 🚀 Quick Start (Windows)

1. **Double-click** `start_app.bat` to automatically:
   - Check Python installation
   - Install dependencies
   - Create necessary directories
   - Launch the app

2. **Open your browser** to `http://localhost:7860`

## 📋 Prerequisites

### Required Software
- **Python 3.8+** - Download from [python.org](https://python.org)
- **FFmpeg** - Required for audio processing

### FFmpeg Installation

#### Windows
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH
4. Restart command prompt

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

## 🔧 Manual Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Directories
```bash
mkdir uploads outputs models temp
```

### 3. Run the App
```bash
python app.py
```

### 4. Open Browser
Navigate to `http://localhost:7860`

## 🧪 Testing Your Setup

Run the test script to verify everything is working:
```bash
python test_setup.py
```

This will check:
- ✅ Module imports
- ✅ Local modules
- ✅ Audio processor
- ✅ Voice converter
- ✅ Configuration
- ✅ Directory structure
- ✅ Sample audio creation

## 📁 Project Structure

```
CLone/
├── app.py                 # Main Gradio web application
├── voice_converter.py     # RVC voice conversion logic
├── audio_processor.py     # Audio processing utilities
├── config.py             # Configuration settings
├── batch_processor.py     # Batch processing script
├── test_setup.py         # Setup verification script
├── requirements.txt       # Python dependencies
├── start_app.bat         # Windows startup script
├── README.md             # Project documentation
├── SETUP_GUIDE.md        # This setup guide
├── uploads/              # Temporary upload storage
├── outputs/              # Generated audio files
├── models/               # RVC model storage
└── temp/                 # Temporary processing files
```

## 🎯 Usage Instructions

### Basic Voice Cloning
1. **Upload Source Audio**: The speech you want to convert
2. **Upload Target Voice**: 10-30 second sample of target voice
3. **Select Model**: Choose RVC model quality/speed
4. **Click Convert**: Wait for processing
5. **Download Result**: Preview and save cloned audio

### Batch Processing
For multiple files, use the batch processor:
```bash
python batch_processor.py input_folder target_voice.wav -o output_folder
```

## ⚠️ Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
pip install -r requirements.txt
```

#### FFmpeg not found
- Ensure FFmpeg is installed and in PATH
- Restart terminal/command prompt after installation

#### Audio file errors
- Check file format (WAV, MP3, M4A, FLAC, OGG supported)
- Ensure file is not corrupted
- Check file size (max 100MB)

#### Port already in use
Change port in `config.py` or kill process using port 7860

### Performance Issues

#### Slow processing
- Use "fast" model for quicker results
- Reduce audio file length
- Close other applications

#### Memory errors
- Process shorter audio files
- Use CPU instead of GPU
- Increase system swap/virtual memory

## 🔒 Security Notes

- App runs locally on your machine
- No audio data is sent to external servers
- Temporary files are stored locally
- Clean up `temp/` folder periodically

## 📚 Advanced Configuration

### Customizing Settings
Edit `config.py` to modify:
- Audio processing parameters
- Model settings
- File handling options
- Web interface settings

### Adding Custom Models
1. Place RVC model files in `models/` folder
2. Update `AVAILABLE_MODELS` in `config.py`
3. Restart the application

## 🆘 Getting Help

### Check Logs
- App logs appear in terminal/command prompt
- Look for error messages and warnings

### Test Components
```bash
python test_setup.py
```

### Verify Dependencies
```bash
pip list | grep -E "(gradio|torch|librosa|numpy)"
```

## 🎉 Success!

Once everything is working:
- ✅ Web interface loads at `http://localhost:7860`
- ✅ File uploads work
- ✅ Voice conversion processes successfully
- ✅ Audio output is generated and downloadable

## 🔄 Updates

To update the app:
1. Download latest version
2. Replace existing files
3. Run `pip install -r requirements.txt`
4. Restart the application

---

**Happy Voice Cloning! 🎤✨**

