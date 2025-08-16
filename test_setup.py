#!/usr/bin/env python3
"""
Test script to verify Voice Cloner App setup
"""

import os
import sys
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    required_modules = [
        "gradio",
        "torch",
        "torchaudio", 
        "librosa",
        "numpy",
        "scipy",
        "soundfile",
        "pydub"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All required modules imported successfully!")
    return True

def test_local_modules():
    """Test if local modules can be imported."""
    print("\n🔍 Testing local modules...")
    
    local_modules = [
        "audio_processor",
        "voice_converter", 
        "config"
    ]
    
    failed_imports = []
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import local modules: {', '.join(failed_imports)}")
        return False
    
    print("✅ All local modules imported successfully!")
    return True

def test_audio_processor():
    """Test audio processor functionality."""
    print("\n🔍 Testing audio processor...")
    
    try:
        from audio_processor import AudioProcessor
        
        # Create processor
        processor = AudioProcessor()
        print("✅ AudioProcessor created")
        
        # Test validation
        test_file = "test_audio.wav"
        is_valid, message = processor.validate_audio_file(test_file)
        print(f"✅ File validation test: {message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processor test failed: {e}")
        return False

def test_voice_converter():
    """Test voice converter functionality."""
    print("\n🔍 Testing voice converter...")
    
    try:
        from voice_converter import RVCVoiceConverter
        
        # Create converter
        converter = RVCVoiceConverter()
        print("✅ RVCVoiceConverter created")
        
        # Test model info
        info = converter.get_model_info()
        print(f"✅ Model info: {info['sample_rate']}Hz, {info['device']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice converter test failed: {e}")
        return False

def test_config():
    """Test configuration functionality."""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import get_config, create_directories, validate_config
        
        # Test config
        config = get_config()
        print("✅ Configuration loaded")
        
        # Test directory creation
        create_directories()
        print("✅ Directories created")
        
        # Test validation
        is_valid = validate_config()
        print(f"✅ Configuration validation: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist."""
    print("\n🔍 Testing directories...")
    
    required_dirs = [
        "uploads",
        "outputs", 
        "models",
        "temp"
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("✅ All required directories exist!")
    return True

def create_sample_audio():
    """Create a sample audio file for testing."""
    print("\n🔍 Creating sample audio...")
    
    try:
        import numpy as np
        import soundfile as sf
        
        # Create a simple sine wave
        sample_rate = 22050
        duration = 2.0  # 2 seconds
        frequency = 440.0  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Save as WAV
        sf.write("test_audio.wav", audio, sample_rate)
        print("✅ Sample audio created: test_audio.wav")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample audio: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Voice Cloner App - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Local Modules", test_local_modules),
        ("Audio Processor", test_audio_processor),
        ("Voice Converter", test_voice_converter),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("Sample Audio", create_sample_audio)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Voice Cloner App is ready to use.")
        print("\n🚀 To start the app, run:")
        print("   python app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check if FFmpeg is installed")
        print("   3. Ensure all Python files are in the same directory")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

