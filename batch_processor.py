#!/usr/bin/env python3
"""
Batch Voice Cloning Processor

Process multiple source audio files with a single target voice sample.
Useful for bulk voice conversion tasks.
"""

import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import json

from voice_converter import VoiceCloningPipeline
from audio_processor import AudioProcessor
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchVoiceProcessor:
    """Process multiple audio files for voice cloning."""
    
    def __init__(self, target_voice_path: str, output_dir: str = "batch_outputs"):
        self.target_voice_path = target_voice_path
        self.output_dir = output_dir
        self.pipeline = VoiceCloningPipeline()
        self.audio_processor = AudioProcessor()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load configuration
        self.config = get_config()
        
        # Results tracking
        self.results = []
        self.failed_files = []
        
    def validate_target_voice(self) -> bool:
        """Validate the target voice file."""
        try:
            is_valid, message = self.audio_processor.validate_audio_file(self.target_voice_path)
            if is_valid:
                logger.info(f"Target voice validated: {message}")
                return True
            else:
                logger.error(f"Target voice validation failed: {message}")
                return False
        except Exception as e:
            logger.error(f"Target voice validation error: {e}")
            return False
    
    def get_audio_files(self, input_dir: str, recursive: bool = False) -> List[str]:
        """Get list of audio files from input directory."""
        audio_files = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return audio_files
        
        # Supported audio formats
        supported_extensions = self.config["file"]["supported_formats"]
        
        if recursive:
            # Recursive search
            for ext in supported_extensions:
                audio_files.extend(input_path.rglob(f"*{ext}"))
        else:
            # Non-recursive search
            for ext in supported_extensions:
                audio_files.extend(input_path.glob(f"*{ext}"))
        
        # Convert to strings and sort
        audio_files = [str(f) for f in audio_files]
        audio_files.sort()
        
        logger.info(f"Found {len(audio_files)} audio files in {input_dir}")
        return audio_files
    
    def process_single_file(self, source_file: str, model_name: str = "default") -> Dict[str, Any]:
        """Process a single audio file."""
        try:
            logger.info(f"Processing: {os.path.basename(source_file)}")
            
            # Process voice cloning
            result = self.pipeline.process_voice_cloning(
                source_file, 
                self.target_voice_path, 
                model_name
            )
            
            if result["success"]:
                # Move output to batch output directory
                original_output = result["output_path"]
                filename = os.path.basename(original_output)
                new_output = os.path.join(self.output_dir, filename)
                
                # Copy file to batch output directory
                import shutil
                shutil.copy2(original_output, new_output)
                
                # Update result
                result["output_path"] = new_output
                result["source_file"] = source_file
                result["target_file"] = self.target_voice_path
                
                logger.info(f"✅ Completed: {filename}")
                return result
            else:
                logger.error(f"❌ Failed: {os.path.basename(source_file)} - {result['error']}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Error processing {os.path.basename(source_file)}: {e}")
            return {
                "success": False,
                "error": str(e),
                "source_file": source_file,
                "target_file": self.target_voice_path
            }
    
    def process_batch(self, input_dir: str, model_name: str = "default", 
                     recursive: bool = False, max_files: int = None) -> Dict[str, Any]:
        """Process all audio files in the input directory."""
        start_time = time.time()
        
        # Validate target voice
        if not self.validate_target_voice():
            return {"success": False, "error": "Target voice validation failed"}
        
        # Get audio files
        audio_files = self.get_audio_files(input_dir, recursive)
        
        if not audio_files:
            return {"success": False, "error": "No audio files found"}
        
        # Limit files if specified
        if max_files and len(audio_files) > max_files:
            audio_files = audio_files[:max_files]
            logger.info(f"Limited to {max_files} files")
        
        # Process files
        total_files = len(audio_files)
        successful = 0
        failed = 0
        
        logger.info(f"Starting batch processing of {total_files} files...")
        
        for i, source_file in enumerate(audio_files, 1):
            logger.info(f"Progress: {i}/{total_files}")
            
            result = self.process_single_file(source_file, model_name)
            
            if result["success"]:
                self.results.append(result)
                successful += 1
            else:
                self.failed_files.append(result)
                failed += 1
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_time_per_file = total_time / total_files if total_files > 0 else 0
        
        batch_result = {
            "success": True,
            "total_files": total_files,
            "successful": successful,
            "failed": failed,
            "total_time": total_time,
            "avg_time_per_file": avg_time_per_file,
            "output_directory": self.output_dir,
            "model_used": model_name,
            "target_voice": self.target_voice_path
        }
        
        logger.info(f"Batch processing completed: {successful}/{total_files} successful")
        logger.info(f"Total time: {total_time:.2f}s, Average: {avg_time_per_file:.2f}s per file")
        
        return batch_result
    
    def save_results(self, output_file: str = None):
        """Save processing results to JSON file."""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "batch_results.json")
        
        try:
            results_data = {
                "summary": {
                    "total_files": len(self.results) + len(self.failed_files),
                    "successful": len(self.results),
                    "failed": len(self.failed_files),
                    "output_directory": self.output_dir
                },
                "successful_results": self.results,
                "failed_results": self.failed_files
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None
    
    def generate_report(self, output_file: str = None):
        """Generate a human-readable report."""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "batch_report.txt")
        
        try:
            with open(output_file, 'w') as f:
                f.write("Voice Cloning Batch Processing Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Target Voice: {self.target_voice_path}\n")
                f.write(f"Output Directory: {self.output_dir}\n")
                f.write(f"Total Files: {len(self.results) + len(self.failed_files)}\n")
                f.write(f"Successful: {len(self.results)}\n")
                f.write(f"Failed: {len(self.failed_files)}\n\n")
                
                if self.results:
                    f.write("Successful Conversions:\n")
                    f.write("-" * 25 + "\n")
                    for result in self.results:
                        f.write(f"• {os.path.basename(result['source_file'])} -> {os.path.basename(result['output_path'])}\n")
                        f.write(f"  Processing time: {result['processing_time']:.2f}s\n")
                    f.write("\n")
                
                if self.failed_files:
                    f.write("Failed Conversions:\n")
                    f.write("-" * 20 + "\n")
                    for result in self.failed_files:
                        f.write(f"• {os.path.basename(result['source_file'])}: {result['error']}\n")
                    f.write("\n")
            
            logger.info(f"Report generated: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Batch Voice Cloning Processor")
    
    parser.add_argument("input_dir", help="Input directory containing audio files")
    parser.add_argument("target_voice", help="Path to target voice sample file")
    parser.add_argument("-o", "--output", default="batch_outputs", 
                       help="Output directory for processed files")
    parser.add_argument("-m", "--model", default="default", 
                       help="RVC model to use for conversion")
    parser.add_argument("-r", "--recursive", action="store_true", 
                       help="Search subdirectories recursively")
    parser.add_argument("--max-files", type=int, 
                       help="Maximum number of files to process")
    parser.add_argument("--save-results", action="store_true", 
                       help="Save detailed results to JSON file")
    parser.add_argument("--generate-report", action="store_true", 
                       help="Generate human-readable report")
    
    args = parser.parse_args()
    
    # Create processor
    processor = BatchVoiceProcessor(args.target_voice, args.output)
    
    # Process batch
    result = processor.process_batch(
        args.input_dir, 
        args.model, 
        args.recursive, 
        args.max_files
    )
    
    if result["success"]:
        # Save results if requested
        if args.save_results:
            processor.save_results()
        
        # Generate report if requested
        if args.generate_report:
            processor.generate_report()
        
        print(f"\n🎉 Batch processing completed!")
        print(f"✅ Successful: {result['successful']}/{result['total_files']}")
        print(f"❌ Failed: {result['failed']}/{result['total_files']}")
        print(f"⏱️  Total time: {result['total_time']:.2f}s")
        print(f"📁 Output directory: {result['output_directory']}")
    else:
        print(f"❌ Batch processing failed: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

