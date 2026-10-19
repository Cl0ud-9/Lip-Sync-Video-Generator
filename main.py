"""
Main entry point for the Lip Sync Video Generator.

This script coordinates the process of generating lip-synced videos by:
1. Performing pre-flight checks for models, API keys, and required tools.
2. Determining input files (face video/image and audio/script).
3. Utilizing ElevenLabs for text-to-speech generation if needed.
4. Calling the Wav2Lip inference script to generate the final video.
"""

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
from typing import Optional, Tuple

from dotenv import load_dotenv

# --- Constants & Configuration ---
# ANSI Colors for terminal output, disabled when stdout isn't a real terminal (e.g. redirected to a file).
_USE_COLOR = sys.stdout.isatty()
GREEN = "\033[92m" if _USE_COLOR else ""
YELLOW = "\033[93m" if _USE_COLOR else ""
RED = "\033[91m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""

# Defaults are anchored to this file's folder (not the current directory) so `python path/to/main.py` works from anywhere.
# Paths the user passes on the command line stay relative to wherever they ran the command.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
WAV2LIP_DIR = os.path.join(BASE_DIR, "Wav2Lip")
DEFAULT_CHECKPOINT = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip.pth")
DEFAULT_FACE_DETECTOR = os.path.join(WAV2LIP_DIR, "face_detection", "detection", "sfd", "s3fd.pth")
DEFAULT_OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "output_video.mp4")
FACE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov")


def _enable_windows_ansi_support() -> None:
    """Best-effort enable of ANSI escape codes on older Windows consoles; no-op elsewhere or on failure."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        enable_virtual_terminal_processing = 0x0004
        std_output_handle = -11
        handle = kernel32.GetStdHandle(std_output_handle)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing)
    except Exception:
        pass


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


def positive_int(value: str) -> int:
    """argparse type for flags that must be an integer >= 1 (0 or negatives would crash or misbehave downstream)."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments; every flag is optional and falls back to auto-detection from input/."""
    parser = argparse.ArgumentParser(
        description="Generate a lip-synced video from a face image/video and either a "
                     "script (via ElevenLabs TTS) or a pre-recorded audio file."
    )
    parser.add_argument("--face", type=str, default=None,
                         help="Path to a face image/video. Auto-detected from input/ if omitted.")
    parser.add_argument("--audio", type=str, default=None,
                         help="Path to a WAV audio file. If omitted, input/audio.wav or "
                              "input/script.txt (via ElevenLabs) is used instead.")
    parser.add_argument("--script", type=str, default=os.path.join(INPUT_DIR, "script.txt"),
                         help="Path to a text script for ElevenLabs TTS (used only if --audio is not given "
                              "and no input/audio.wav is found).")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_VIDEO,
                         help="Path to save the generated video.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                         help="Path to the Wav2Lip model checkpoint.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                         help="Force a compute device instead of auto-detecting CUDA availability.")
    parser.add_argument("--resize-factor", type=positive_int, default=2,
                         help="Downscale input frames by this factor before inference (speeds things up "
                              "and reduces memory use; try 4 on CPU-only machines).")
    parser.add_argument("--batch-size", type=positive_int, default=128,
                         help="Wav2Lip inference batch size. Lower this if you hit out-of-memory errors.")
    parser.add_argument("--smooth", action="store_true",
                         help="Enable temporal smoothing of detected face boxes (off by default, matching "
                              "the previous --nosmooth behavior).")
    parser.add_argument("--keep-temp", action="store_true",
                         help="Keep intermediate files in Wav2Lip/temp/ after a successful run "
                              "(useful for debugging).")
    return parser.parse_args()


def check_pre_flight(args: argparse.Namespace) -> bool:
    """
    Verifies that all necessary models, keys, tools, and directories are present.

    Returns:
        bool: True if checks pass (or are non-fatal), False if a critical dependency is missing.
    """
    log("Running pre-flight checks...")

    # 1. Check for Environment Variables
    # The key is only needed if text-to-speech ends up being used, so it is checked there rather than here.
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    # 2. Check for ffmpeg on PATH (required by both ElevenLabs conversion and Wav2Lip muxing)
    if shutil.which("ffmpeg") is None:
        log("ffmpeg was not found on PATH.", "error")
        log("Install it (e.g. 'winget install Gyan.FFmpeg' on Windows) and restart your terminal.", "error")
        return False

    # 3. Check for Wav2Lip Models
    if not os.path.exists(args.checkpoint):
        log(f"Model not found: {args.checkpoint}", "error")
        log("Please download it using the link in README.md", "error")
        return False

    if not os.path.exists(DEFAULT_FACE_DETECTOR):
        log(f"Model not found: {DEFAULT_FACE_DETECTOR}", "error")
        log("Please download it using the link in README.md", "error")
        return False

    # 4. Check/Create Input and Output Directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return True


def get_input_files(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    """
    Determines the face image/video and audio source to use, auto-detecting from input/ unless overridden.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing (face_path, audio_path).
                                             Returns (None, None) if inputs are invalid.
    """
    # 1. Find Face Image/Video
    if args.face:
        if not os.path.isfile(args.face):
            log(f"--face path does not exist: {args.face}", "error")
            return None, None
        face_path = args.face
    else:
        # Sorted for deterministic behavior when multiple candidate files are present. Matched
        # case-insensitively so e.g. PHOTO.JPG is found on Linux/macOS too.
        face_files = sorted(
            path for path in glob.glob(os.path.join(INPUT_DIR, "*"))
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in FACE_EXTENSIONS
        )

        if not face_files:
            log("No face image or video found in input/ folder!", "error")
            return None, None

        if len(face_files) > 1:
            log(f"Multiple face files found in input/: {face_files}. Using the first "
                f"one alphabetically. Pass --face to pick a specific one.", "warn")

        face_path = face_files[0]

    log(f"Using face input: {face_path}")

    # 2. Determine Audio Source
    if args.audio:
        if not os.path.isfile(args.audio):
            log(f"--audio path does not exist: {args.audio}", "error")
            return None, None
        log(f"Using custom audio: {args.audio}. Skipping Text-to-Speech.")
        return face_path, args.audio

    audio_path = os.path.join(INPUT_DIR, "audio.wav")
    final_audio_path = os.path.join(OUTPUT_DIR, "audio.wav")

    if os.path.isfile(audio_path):
        log(f"Found custom audio: {audio_path}. Skipping Text-to-Speech.")
        return face_path, audio_path

    if os.path.isfile(args.script):
        if not os.getenv("ELEVENLABS_API_KEY"):
            log("ELEVENLABS_API_KEY not found in .env or environment variables, which is needed to turn "
                f"{args.script} into speech. Add it to .env, or provide your own audio via --audio.", "error")
            return None, None

        log(f"No custom audio found. Found script: {args.script}. Running ElevenLabs TTS...")

        # Execute the ElevenLabs generation script using the same Python interpreter.
        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "Elevenlab.py"),
            "--script", args.script,
            "--output", final_audio_path,
            "--output_mp3", os.path.join(OUTPUT_DIR, "audio.mp3"),
        ]

        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                log("Text-to-Speech generation failed!", "error")
                return None, None
        except Exception as e:
            log(f"Failed to run ElevenLabs script: {e}", "error")
            return None, None

        return face_path, final_audio_path

    log("No input found! Please put either 'audio.wav' OR 'script.txt' in the input folder "
        "(or pass --audio / --script explicitly).", "error")
    return None, None


def cleanup_temp_files(keep_temp: bool) -> None:
    """Removes intermediate Wav2Lip/temp/ artifacts left over from a run, unless --keep-temp was passed."""
    if keep_temp:
        return

    temp_dir = os.path.join(WAV2LIP_DIR, "temp")
    if not os.path.isdir(temp_dir):
        return

    for name in os.listdir(temp_dir):
        if name == ".gitkeep":
            continue
        path = os.path.join(temp_dir, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as e:
            log(f"Could not remove temp file {path}: {e}", "warn")


def main():
    """
    Main execution flow.
    """
    _enable_windows_ansi_support()
    args = parse_args()

    if not check_pre_flight(args):
        sys.exit(1)

    face_path, audio_path = get_input_files(args)
    if not face_path or not audio_path:
        sys.exit(1)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    log("Starting Wav2Lip inference...")
    log(f"Face   : {face_path}")
    log(f"Audio  : {audio_path}")
    log(f"Output : {args.output}")

    # Call the separate Wav2Lip script via subprocess to keep environments isolated.
    wav2lip_inference_script = os.path.join(WAV2LIP_DIR, "inference.py")

    cmd = [
        sys.executable,
        wav2lip_inference_script,
        "--checkpoint_path", args.checkpoint,
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", args.output,
        "--resize_factor", str(args.resize_factor),
        "--wav2lip_batch_size", str(args.batch_size),
        "--device", args.device,
    ]
    if not args.smooth:
        cmd.append("--nosmooth")

    try:
        subprocess.run(cmd, check=True)
        log(f"Success! Video saved to: {args.output}")
        cleanup_temp_files(args.keep_temp)
    except subprocess.CalledProcessError:
        log("Wav2Lip inference failed.", "error")
        sys.exit(1)
    except KeyboardInterrupt:
        log("Process interrupted by user.", "warn")
        sys.exit(130)  # conventional exit status for SIGINT, so scripts can tell this apart from success


if __name__ == "__main__":
    main()
