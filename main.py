import os
import sys
import subprocess
import glob
from dotenv import load_dotenv

# ANSI Colors for nicer output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def log(message, type="info"):
    if type == "info":
        print(f"{GREEN}[INFO]{RESET} {message}")
    elif type == "warn":
        print(f"{YELLOW}[WARN]{RESET} {message}")
    elif type == "error":
        print(f"{RED}[ERROR]{RESET} {message}")

def check_pre_flight():
    """Verifies that all necessary models and keys are present."""
    log("Running pre-flight checks...")
    
    # 1. Check .env
    load_dotenv()
    if not os.getenv("ELEVENLABS_API_KEY"):
        log("ELEVENLABS_API_KEY not found in .env or environment variables.", "warn")
        # We don't exit here because the user might just be providing audio.wav directly

    # 2. Check Models
    if not os.path.exists("Wav2Lip/checkpoints/wav2lip.pth"):
        log("Model not found: Wav2Lip/checkpoints/wav2lip.pth", "error")
        log("Please download it using the link in README.md", "error")
        return False

    if not os.path.exists("Wav2Lip/face_detection/detection/sfd/s3fd.pth"):
        log("Model not found: Wav2Lip/face_detection/detection/sfd/s3fd.pth", "error")
        log("Please download it using the link in README.md", "error")
        return False

    # 3. Check Folders
    if not os.path.exists("input"):
        os.makedirs("input")
    if not os.path.exists("output"):
        os.makedirs("output")

    return True

def get_input_files():
    """Intelligently determines the face and audio input."""
    
    # 1. Find Face Image/Video
    face_extensions = ['*.jpg', '*.png', '*.jpeg', '*.mp4', '*.avi', '*.mov']
    face_files = []
    for ext in face_extensions:
        face_files.extend(glob.glob(os.path.join("input", ext)))
    
    if not face_files:
        log("No face image or video found in input/ folder!", "error")
        return None, None
    
    face_path = face_files[0] # Pick the first one
    log(f"Using face input: {face_path}")

    # 2. Determine Audio Source
    audio_path = "input/audio.wav"
    script_path = "input/script.txt"
    final_audio_path = "output/audio.wav"

    if os.path.exists(audio_path):
        log(f"Found custom audio: {audio_path}. Skipping Text-to-Speech.")
        final_audio_path = audio_path
    elif os.path.exists(script_path):
        log(f"No custom audio found. Found script: {script_path}. Running ElevenLabs...")
        
        # Run Elevenlab.py
        result = subprocess.run([sys.executable, "Elevenlab.py", "--script", script_path, "--output", final_audio_path])
        
        if result.returncode != 0:
            log("Text-to-Speech generation failed!", "error")
            return None, None
    else:
        log("No input found! Please put either 'audio.wav' OR 'script.txt' in the input folder.", "error")
        return None, None

    return face_path, final_audio_path

def main():
    if not check_pre_flight():
        sys.exit(1)

    face_path, audio_path = get_input_files()
    if not face_path or not audio_path:
        sys.exit(1)

    output_video = "output/output_video.mp4"
    
    log("Starting Wav2Lip inference...")
    log(f"Face: {face_path}")
    log(f"Audio: {audio_path}")
    log(f"Output: {output_video}")

    # Construct the inference command using the robust logic we built
    # We use subprocess to call the other script to ensure environment separation and clean execution
    cmd = [
        sys.executable,
        "Wav2Lip/inference.py",
        "--checkpoint_path", "Wav2Lip/checkpoints/wav2lip.pth",
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

if __name__ == "__main__":
    main()
