import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import make_chunks
import tempfile
import shutil
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio processing, conversion, and preprocessing for voice cloning."""
    
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to target sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            logger.info(f"Loaded audio: {file_path}, shape: {audio.shape}, sr: {sr}")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def save_audio(self, audio: np.ndarray, file_path: str, sr: int = None) -> str:
        """
        Save audio data to file.
        
        Args:
            audio: Audio data as numpy array
            file_path: Output file path
            sr: Sample rate (uses target_sr if None)
            
        Returns:
            Path to saved file
        """
        try:
            if sr is None:
                sr = self.target_sr
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save with soundfile
            sf.write(file_path, audio, sr)
            logger.info(f"Saved audio: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving audio {file_path}: {e}")
            raise
    
    def convert_format(self, input_path: str, output_format: str = 'wav') -> str:
        """
        Convert audio file to specified format.
        
        Args:
            input_path: Input audio file path
            output_format: Desired output format (without dot)
            
        Returns:
            Path to converted file
        """
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(input_path)
            
            # Create output path
            output_path = input_path.rsplit('.', 1)[0] + f'.{output_format}'
            
            # Export to new format
            audio.export(output_path, format=output_format)
            logger.info(f"Converted {input_path} to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error converting format: {e}")
            raise
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to prevent clipping.
        
        Args:
            audio: Input audio data
            
        Returns:
            Normalized audio data
        """
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            # Normalize to prevent clipping
            if rms > 0:
                normalized = audio / (rms * 1.5)
                # Clip to prevent overflow
                normalized = np.clip(normalized, -1.0, 1.0)
                return normalized
            return audio
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return audio
    
    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        """
        Remove silence from beginning and end of audio.
        
        Args:
            audio: Input audio data
            threshold_db: Silence threshold in dB
            
        Returns:
            Trimmed audio data
        """
        try:
            # Convert threshold to amplitude
            threshold = 10**(threshold_db / 20)
            
            # Find non-silent regions
            non_silent = np.where(np.abs(audio) > threshold)[0]
            
            if len(non_silent) > 0:
                start = non_silent[0]
                end = non_silent[-1]
                return audio[start:end+1]
            
            return audio
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return audio
    
    def segment_audio(self, audio: np.ndarray, segment_length: int = 10) -> list:
        """
        Split audio into segments for processing.
        
        Args:
            audio: Input audio data
            segment_length: Length of each segment in seconds
            
        Returns:
            List of audio segments
        """
        try:
            samples_per_segment = int(self.target_sr * segment_length)
            segments = []
            
            for i in range(0, len(audio), samples_per_segment):
                segment = audio[i:i + samples_per_segment]
                if len(segment) >= samples_per_segment // 2:  # Only add if at least half length
                    segments.append(segment)
            
            logger.info(f"Created {len(segments)} segments")
            return segments
        except Exception as e:
            logger.error(f"Error segmenting audio: {e}")
            return [audio]
    
    def preprocess_for_rvc(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio specifically for RVC model input.
        
        Args:
            audio: Input audio data
            
        Returns:
            Preprocessed audio data
        """
        try:
            # Normalize
            audio = self.normalize_audio(audio)
            
            # Trim silence
            audio = self.trim_silence(audio)
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            logger.info(f"Preprocessed audio shape: {audio.shape}")
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing for RVC: {e}")
            return audio
    
    def validate_audio_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate audio file for voice cloning.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "File is empty"
            
            # Check format
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                return False, f"Unsupported format: {file_ext}"
            
            # Try to load audio
            try:
                audio, sr = self.load_audio(file_path)
                duration = len(audio) / sr
                
                if duration < 1.0:
                    return False, "Audio too short (less than 1 second)"
                if duration > 300:  # 5 minutes
                    return False, "Audio too long (more than 5 minutes)"
                
                return True, f"Valid audio: {duration:.1f}s, {sr}Hz"
                
            except Exception as e:
                return False, f"Failed to load audio: {str(e)}"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def create_temp_file(self, suffix: str = '.wav') -> str:
        """
        Create a temporary file for processing.
        
        Args:
            suffix: File extension
            
        Returns:
            Path to temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_file.close()
        return temp_file.name
    
    def cleanup_temp_files(self, file_paths: list):
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of file paths to remove
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

