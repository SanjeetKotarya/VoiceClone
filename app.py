import gradio as gr
import os
import time
import logging
from typing import Tuple, Optional
import tempfile
import shutil

from voice_converter import RVCVoiceConverter, VoiceCloningPipeline
from audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
audio_processor = AudioProcessor()
voice_converter = RVCVoiceConverter()
pipeline = VoiceCloningPipeline(voice_converter)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Available models (you can add more)
AVAILABLE_MODELS = {
    "default": "Default RVC Model",
    "high_quality": "High Quality RVC Model",
    "fast": "Fast Processing Model"
}

def validate_upload(file_path: str, file_type: str) -> Tuple[bool, str]:
    """Validate uploaded file."""
    if file_path is None:
        return False, f"No {file_type} file uploaded"
    
    try:
        is_valid, message = audio_processor.validate_audio_file(file_path)
        return is_valid, message
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def clone_voice(source_audio, target_voice, model_name, progress=gr.Progress()):
    """
    Main voice cloning function for Gradio interface.
    
    Args:
        source_audio: Source audio file
        target_voice: Target voice sample
        model_name: Selected model name
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (output_audio, info_text, download_link)
    """
    try:
        # Validate inputs
        progress(0.1, desc="Validating inputs...")
        
        source_valid, source_msg = validate_upload(source_audio, "source audio")
        target_valid, target_msg = validate_upload(target_voice, "target voice")
        
        if not source_valid:
            return None, f"❌ Source audio error: {source_msg}", None
        if not target_valid:
            return None, f"❌ Target voice error: {target_msg}", None
        
        progress(0.2, desc="Processing voice cloning...")
        
        # Process voice cloning
        result = pipeline.process_voice_cloning(source_audio, target_voice, model_name)
        
        if not result["success"]:
            return None, f"❌ Voice cloning failed: {result['error']}", None
        
        progress(0.9, desc="Finalizing...")
        
        # Prepare output
        output_path = result["output_path"]
        processing_time = result["processing_time"]
        source_duration = result["source_duration"]
        target_duration = result["target_duration"]
        
        # Create info text
        info_text = f"""
        ✅ Voice cloning completed successfully!
        
        📊 Results:
        • Processing time: {processing_time:.2f} seconds
        • Source duration: {source_duration:.1f} seconds
        • Target duration: {target_duration:.1f} seconds
        • Model used: {model_name}
        • Sample rate: {result['sample_rate']} Hz
        
        🎵 Your cloned audio is ready!
        """
        
        # Create download link
        download_link = f"📥 [Download Cloned Audio]({output_path})"
        
        progress(1.0, desc="Complete!")
        
        return output_path, info_text, download_link
        
    except Exception as e:
        logger.error(f"Voice cloning error: {e}")
        return None, f"❌ Unexpected error: {str(e)}", None

def synthesize_speech(target_voice, text_input, model_name, progress=gr.Progress()):
    """
    Text-to-speech synthesis using the uploaded voice sample.
    
    Args:
        target_voice: Target voice sample
        text_input: Text to synthesize
        model_name: Selected model name
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (output_audio, info_text, download_link)
    """
    try:
        # Validate inputs
        progress(0.1, desc="Validating inputs...")
        
        if not text_input or text_input.strip() == "":
            return None, "❌ Please enter some text to synthesize", None
            
        if len(text_input) > 1000:
            return None, "❌ Text too long. Please keep it under 1000 characters.", None
        
        target_valid, target_msg = validate_upload(target_voice, "target voice")
        if not target_valid:
            return None, f"❌ Target voice error: {target_msg}", None
        
        progress(0.2, desc="Processing text-to-speech...")
        
        # Process TTS synthesis
        result = pipeline.process_tts_synthesis(target_voice, text_input, model_name)
        
        if not result["success"]:
            return None, f"❌ TTS synthesis failed: {result['error']}", None
        
        progress(0.9, desc="Finalizing...")
        
        # Prepare output
        output_path = result["output_path"]
        processing_time = result["processing_time"]
        text_length = len(text_input)
        
        # Create info text
        info_text = f"""
        ✅ Text-to-speech synthesis completed successfully!
        
        📊 Results:
        • Processing time: {processing_time:.2f} seconds
        • Text length: {text_length} characters
        • Model used: {model_name}
        • Sample rate: {result['sample_rate']} Hz
        
        🎵 Your synthesized audio is ready!
        """
        
        # Create download link
        download_link = f"📥 [Download Synthesized Audio]({output_path})"
        
        progress(1.0, desc="Complete!")
        
        return output_path, info_text, download_link
        
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        return None, f"❌ Unexpected error: {str(e)}", None

def create_interface():
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .upload-area {
        border: 2px dashed #4CAF50 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        text-align: center !important;
    }
    .success-box {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 5px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    .error-box {
        background-color: #f8d7da !important;
        border: 1px solid #f5c6cb !important;
        border-radius: 5px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Voice Cloner App") as interface:
        
        # Header
        gr.Markdown("""
        # 🎤 Voice Cloner App
        
        Transform any voice using RVC (Retrieval-based Voice Conversion) technology.
        Upload source audio and a target voice sample to create amazing voice clones!
        """)
        
        # Create tabs for different features
        with gr.Tabs():
            # Tab 1: Voice Cloning
            with gr.Tab("🎤 Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # File uploads
                        gr.Markdown("### 📁 Upload Files")
                        
                        source_audio = gr.Audio(
                            label="Source Audio (Speech to Convert)",
                            type="filepath"
                        )
                        
                        target_voice = gr.Audio(
                            label="Target Voice Sample (10-30 seconds)",
                            type="filepath"
                        )
                        
                        # Model selection
                        gr.Markdown("### ⚙️ Model Settings")
                        model_name = gr.Dropdown(
                            choices=list(AVAILABLE_MODELS.keys()),
                            value="default",
                            label="Select Model"
                        )
                        
                        # Convert button
                        convert_btn = gr.Button(
                            "🚀 Convert Voice",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        # Results section
                        gr.Markdown("### 🎵 Results")
                        
                        output_audio = gr.Audio(
                            label="Cloned Audio Output",
                            type="filepath",
                            interactive=False
                        )
                        
                        info_text = gr.Markdown(
                            value="Upload files and click 'Convert Voice' to start...",
                            elem_classes=["success-box"]
                        )
                        
                        download_link = gr.Markdown(
                            value="",
                            elem_classes=["success-box"]
                        )
            
            # Tab 2: Text-to-Speech
            with gr.Tab("📝 Text-to-Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Voice upload and text input
                        gr.Markdown("### 🎵 Voice Sample")
                        
                        tts_target_voice = gr.Audio(
                            label="Target Voice Sample (10-30 seconds)",
                            type="filepath"
                        )
                        
                        gr.Markdown("### 📝 Text Input")
                        
                        text_input = gr.Textbox(
                            label="Enter text to synthesize",
                            placeholder="Type or paste your text here...",
                            lines=5,
                            max_lines=10
                        )
                        
                        # Model selection for TTS
                        gr.Markdown("### ⚙️ Model Settings")
                        tts_model_name = gr.Dropdown(
                            choices=list(AVAILABLE_MODELS.keys()),
                            value="default",
                            label="Select Model"
                        )
                        
                        # TTS button
                        tts_btn = gr.Button(
                            "🎵 Synthesize Speech",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        # TTS Results section
                        gr.Markdown("### 🎵 Results")
                        
                        tts_output_audio = gr.Audio(
                            label="Synthesized Audio Output",
                            type="filepath",
                            interactive=False
                        )
                        
                        tts_info_text = gr.Markdown(
                            value="Upload a voice sample and enter text to start...",
                            elem_classes=["success-box"]
                        )
                        
                        tts_download_link = gr.Markdown(
                            value="",
                            elem_classes=["success-box"]
                        )
        
        # Instructions
        with gr.Accordion("📖 How to Use", open=False):
            gr.Markdown("""
            ## 🎤 Voice Cloning
            
            1. **Upload Source Audio**: Choose an audio file containing speech you want to convert
            2. **Upload Target Voice**: Provide a 10-30 second sample of the target voice
            3. **Select Model**: Choose the appropriate RVC model for your needs
            4. **Convert**: Click the convert button and wait for processing
            5. **Download**: Preview the result and download your cloned audio
            
            ## 📝 Text-to-Speech
            
            1. **Upload Voice Sample**: Provide a 10-30 second sample of the voice you want to use
            2. **Enter Text**: Type or paste the text you want to synthesize
            3. **Select Model**: Choose the appropriate model for synthesis
            4. **Synthesize**: Click the synthesize button and wait for processing
            5. **Download**: Preview the result and download your synthesized audio
            
            ## 💡 Tips for Best Results
            
            - **Voice Samples**: High-quality recording, consistent speaking style, 10-30 seconds
            - **Source Audio**: Clear speech, minimal background noise, 1-5 minutes length
            - **Text Input**: Keep TTS text under 1000 characters for best results
            - **Format**: WAV, MP3, M4A, FLAC, or OGG files supported
            - **Processing**: Time depends on audio length and model complexity
            """)
        
        # Technical details
        with gr.Accordion("🔧 Technical Details", open=False):
            gr.Markdown(f"""
            ## RVC Technology
            
            This app uses **Retrieval-based Voice Conversion (RVC)** technology:
            
            - **Content Extraction**: Analyzes linguistic content from source audio
            - **Speaker Embedding**: Extracts voice characteristics from target sample
            - **Voice Conversion**: Applies target voice characteristics to source content
            - **Audio Generation**: Produces high-quality cloned audio output
            
            ## Model Information
            
            - **Sample Rate**: {voice_converter.sample_rate} Hz
            - **Hop Length**: {voice_converter.hop_length} samples
            - **Window Length**: {voice_converter.win_length} samples
            - **FFT Size**: {voice_converter.n_fft} samples
            
            ## Supported Formats
            
            - Input: WAV, MP3, M4A, FLAC, OGG
            - Output: WAV (high quality)
            """)
        
        # Event handlers for Voice Cloning
        convert_btn.click(
            fn=clone_voice,
            inputs=[source_audio, target_voice, model_name],
            outputs=[output_audio, info_text, download_link],
            show_progress=True
        )
        
        # Event handlers for Text-to-Speech
        tts_btn.click(
            fn=synthesize_speech,
            inputs=[tts_target_voice, text_input, tts_model_name],
            outputs=[tts_output_audio, tts_info_text, tts_download_link],
            show_progress=True
        )
        
        # File validation on upload for Voice Cloning
        source_audio.change(
            fn=lambda x: validate_upload(x, "source audio")[1] if x else "No file uploaded",
            inputs=source_audio,
            outputs=gr.Textbox(label="Source Audio Status", interactive=False)
        )
        
        target_voice.change(
            fn=lambda x: validate_upload(x, "target voice")[1] if x else "No file uploaded",
            inputs=target_voice,
            outputs=gr.Textbox(label="Target Voice Status", interactive=False)
        )
        
        # File validation on upload for TTS
        tts_target_voice.change(
            fn=lambda x: validate_upload(x, "target voice")[1] if x else "No file uploaded",
            inputs=tts_target_voice,
            outputs=gr.Textbox(label="TTS Voice Status", interactive=False)
        )
    
    return interface

def main():
    """Main function to run the app."""
    try:
        # Create interface
        interface = create_interface()
        
        # Launch app
        print("🚀 Starting Voice Cloner App...")
        print("📱 Opening web interface...")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start app: {e}")
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()

