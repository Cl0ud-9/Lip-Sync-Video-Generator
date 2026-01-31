"""
Main entry point for the Lip Sync Video Generator.

This script coordinates the process of generating lip-synced videos by:
1. Performing pre-flight checks for models and API keys.
2. Determining input files (face video/image and audio/script).
3. Utilizing ElevenLabs for text-to-speech generation if needed.
4. Calling the Wav2Lip inference script to generate the final video.
"""

import glob
import os
import subprocess
import sys
from typing import Optional, Tuple

from dotenv import load_dotenv

# --- Constants & Configuration ---
# ANSI Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def log(message: str, level: str = "info") -> None:
    """
    Prints a formatted log message to the console.

    Args:
        message (str): The message to print.
        level (str): The severity level ('info', 'warn', 'error'). Defaults to "info".
    """
    if level == "info":
        print(f"{GREEN}[INFO]{RESET} {message}")
    elif level == "warn":
        print(f"{YELLOW}[WARN]{RESET} {message}")
    elif level == "error":
        print(f"{RED}[ERROR]{RESET} {message}")


def check_pre_flight() -> bool:
    """
    Verifies that all necessary models, keys, and directories are present.

    Returns:
        bool: True if checks pass (or are non-fatal), False if a critical dependency is missing.
    """
    log("Running pre-flight checks...")

    # 1. Check for Environment Variables
    load_dotenv()
    if not os.getenv("ELEVENLABS_API_KEY"):
        log("ELEVENLABS_API_KEY not found in .env or environment variables.", "warn")
        # Non-fatal: User might provide their own audio file.

    # 2. Check for Wav2Lip Models
    wav2lip_model_path = os.path.join("Wav2Lip", "checkpoints", "wav2lip.pth")
    if not os.path.exists(wav2lip_model_path):
        log(f"Model not found: {wav2lip_model_path}", "error")
        log("Please download it using the link in README.md", "error")
        return False

    face_detector_path = os.path.join("Wav2Lip", "face_detection", "detection", "sfd", "s3fd.pth")
    if not os.path.exists(face_detector_path):
        log(f"Model not found: {face_detector_path}", "error")
        log("Please download it using the link in README.md", "error")
        return False

    # 3. Check/Create Input and Output Directories
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    return True


def get_input_files() -> Tuple[Optional[str], Optional[str]]:
    """
    Determines the face image/video and audio source to use.

    Scans the 'input/' directory for compatible face files and audio/script files.
    If a script is found but no audio, it invokes the ElevenLabs script to generate audio.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing (face_path, audio_path).
                                             Returns (None, None) if inputs are invalid.
    """
    # 1. Find Face Image/Video
    face_extensions = ['*.jpg', '*.png', '*.jpeg', '*.mp4', '*.avi', '*.mov']
    face_files = []
    
    # Use glob to find files matching extensions in the input directory
    for ext in face_extensions:
        face_files.extend(glob.glob(os.path.join("input", ext)))

    if not face_files:
        log("No face image or video found in input/ folder!", "error")
        return None, None

    # Default to the first valid file found
    face_path = face_files[0]
    log(f"Using face input: {face_path}")

    # 2. Determine Audio Source
    audio_path = os.path.join("input", "audio.wav")
    script_path = os.path.join("input", "script.txt")
    final_audio_path = os.path.join("output", "audio.wav")

    if os.path.exists(audio_path):
        log(f"Found custom audio: {audio_path}. Skipping Text-to-Speech.")
        final_audio_path = audio_path
        
    elif os.path.exists(script_path):
        log(f"No custom audio found. Found script: {script_path}. Running ElevenLabs TTS...")

        # Execute the ElevenLabs generation script
        # We use sys.executable to ensure we use the same Python interpreter
        cmd = [
            sys.executable, 
            "Elevenlab.py", 
            "--script", script_path, 
            "--output", final_audio_path
        ]
        
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                log("Text-to-Speech generation failed!", "error")
                return None, None
        except Exception as e:
            log(f"Failed to run ElevenLabs script: {e}", "error")
            return None, None
            
    else:
        log("No input found! Please put either 'audio.wav' OR 'script.txt' in the input folder.", "error")
        return None, None

    return face_path, final_audio_path


def main():
    """
    Main execution flow.
    """
    if not check_pre_flight():
        sys.exit(1)

    face_path, audio_path = get_input_files()
    if not face_path or not audio_path:
        sys.exit(1)

    output_video = os.path.join("output", "output_video.mp4")

    log("Starting Wav2Lip inference...")
    log(f"Face   : {face_path}")
    log(f"Audio  : {audio_path}")
    log(f"Output : {output_video}")

    # Construct the inference command
    # We use subprocess to call the separate Wav2Lip script to ensure environment separation
    wav2lip_inference_script = os.path.join("Wav2Lip", "inference.py")
    wav2lip_checkpoint = os.path.join("Wav2Lip", "checkpoints", "wav2lip.pth")
    
    cmd = [
        sys.executable,
        wav2lip_inference_script,
        "--checkpoint_path", wav2lip_checkpoint,
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", output_video,
        "--resize_factor", "2",
        "--nosmooth",
        "--wav2lip_batch_size", "256"
    ]

    try:
        subprocess.run(cmd, check=True)
        log(f"Success! Video saved to: {output_video}")
    except subprocess.CalledProcessError:
        log("Wav2Lip inference failed.", "error")
        sys.exit(1)
    except KeyboardInterrupt:
        log("Process interrupted by user.", "warn")
        sys.exit(0)


if __name__ == "__main__":
    main()
