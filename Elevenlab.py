"""
ElevenLabs Text-to-Speech Integration.

This module handles the interaction with the ElevenLabs API to generate speech from text.
It also includes utility functions to convert the generated audio formats if necessary.

Dependencies:
    - requests
    - python-dotenv
    - ffmpeg (must be installed and on PATH; used directly for MP3->WAV conversion)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

# Load environment variables once at module level
load_dotenv()

# Network timeout (seconds) for the ElevenLabs API call, so a stalled connection can't hang forever.
REQUEST_TIMEOUT = 60

# Transient failures (network blips, 5xx from ElevenLabs) are retried; auth/quota errors (401/429) never are.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2


def generate_audio_with_elevenlabs(
    script_path: str,
    output_audio_path: str,
    voice_id: str,
    api_key: str,
    model_id: str = "eleven_turbo_v2_5",
    stability: float = 0.6,
    similarity_boost: float = 0.5
) -> bool:
    """
    Reads text from the given script file and sends a request to the ElevenLabs API to generate speech.
    The generated audio (MP3) is saved to the specified output path.

    Args:
        script_path (str): Path to the text file containing your script.
        output_audio_path (str): Path where the generated MP3 audio will be saved.
        voice_id (str): ElevenLabs voice ID (e.g., 'Sm1seazb4gs7RSlUVw7c').
        api_key (str): Your ElevenLabs API key.
        model_id (str, optional): The ElevenLabs model ID to use. Defaults to "eleven_turbo_v2_5".
        stability (float, optional): Controls voice stability. Defaults to 0.6.
        similarity_boost (float, optional): Controls similarity boost. Defaults to 0.5.

    Returns:
        bool: True if the audio file was generated and written successfully, False otherwise.
    """
    # Read the input text
    try:
        # utf-8-sig so a BOM (Windows Notepad adds one) doesn't end up as a stray character in the spoken text.
        with open(script_path, "r", encoding="utf-8-sig") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print(f"[ERROR] Script file not found: {script_path}")
        return False
    except UnicodeDecodeError:
        print(f"[ERROR] Script file is not valid UTF-8 text: {script_path}")
        return False

    if not text:
        print("[ERROR] Script file is empty.")
        return False

    # Set up headers and payload for the API request
    headers = {
        "Accept": "audio/mpeg",
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    print(f"[INFO] Sending request to ElevenLabs (Voice ID: {voice_id})...")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        retryable = True
        try:
            response = requests.post(url, json=data, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                if not response.content:
                    print("[ERROR] ElevenLabs returned an empty audio response.")
                    return False
                # Written to a temp file first so a failed write can't leave a truncated MP3 behind.
                os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok=True)
                tmp_path = output_audio_path + ".part"
                with open(tmp_path, "wb") as out_file:
                    out_file.write(response.content)
                os.replace(tmp_path, output_audio_path)
                print(f"[INFO] Audio saved at: {output_audio_path}")
                return True
            elif response.status_code == 401:
                retryable = False
                print("[ERROR] ElevenLabs rejected the API key (401 Unauthorized). "
                      "Check ELEVENLABS_API_KEY in your .env file.")
            elif response.status_code == 429:
                retryable = False
                print("[ERROR] ElevenLabs rate limit / quota exceeded (429). "
                      "Wait a bit or check your plan usage before retrying.")
            else:
                # 4xx other than the above are the caller's fault (bad voice id, text too long...): don't retry.
                retryable = response.status_code >= 500
                print(f"[ERROR] ElevenLabs API returned code {response.status_code}: {response.text[:500]}")

        except requests.Timeout:
            print(f"[ERROR] Request to ElevenLabs timed out after {REQUEST_TIMEOUT}s. "
                  "Check your network connection and try again.")
        except requests.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
        except OSError as e:
            print(f"[ERROR] Could not write audio to {output_audio_path}: {e}")
            return False

        if not retryable or attempt == MAX_ATTEMPTS:
            break
        delay = RETRY_BACKOFF_SECONDS * attempt
        print(f"[INFO] Retrying in {delay}s (attempt {attempt + 1}/{MAX_ATTEMPTS})...")
        time.sleep(delay)

    return False


def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """
    Converts an MP3 file to WAV format by shelling out to ffmpeg directly.

    Uses ffmpeg instead of pydub, since pydub's audioop dependency was removed in Python 3.13+.

    Args:
        mp3_path (str): Path to the source MP3 file.
        wav_path (str): Path to the destination WAV file.

    Returns:
        bool: True if conversion succeeded, False otherwise.
    """
    if not os.path.isfile(mp3_path):
        print(f"[ERROR] MP3 file not found: {mp3_path}")
        return False

    if shutil.which("ffmpeg") is None:
        print("[ERROR] ffmpeg was not found on PATH. Install it and make sure it is "
              "accessible from this terminal (see README.md).")
        return False

    print(f"[INFO] Converting {mp3_path} to {wav_path}...")
    wav_dir = os.path.dirname(wav_path)
    if wav_dir:
        os.makedirs(wav_dir, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path, wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"[ERROR] Failed to convert MP3 to WAV (ffmpeg exit code {result.returncode}):")
        print(result.stderr.strip())
        return False

    print("[INFO] Conversion successful.")
    return True


def main():
    """
    Main entry point for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Generate speech using ElevenLabs API")

    parser.add_argument("--script", type=str, default="input/script.txt", help="Path to the text script file")
    parser.add_argument("--output", type=str, default="output/audio.wav", help="Path to save the result audio (WAV)")
    parser.add_argument("--output_mp3", type=str, default="output/audio.mp3", help="Path to save the intermediate MP3")
    parser.add_argument("--voice_id", type=str, default="Sm1seazb4gs7RSlUVw7c", help="ElevenLabs Voice ID")
    parser.add_argument("--api_key", type=str, default=None, help="ElevenLabs API Key (optional, defaults to .env)")

    args = parser.parse_args()

    # Load API Key
    api_key = args.api_key or os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        print("[ERROR] ELEVENLABS_API_KEY not found in environment variables or arguments.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 1. Generate Audio (MP3) -- use the return value, not file existence, to avoid stale-file false positives
    generated = generate_audio_with_elevenlabs(args.script, args.output_mp3, args.voice_id, api_key)
    if not generated:
        print("[ERROR] Audio generation failed, cannot convert to WAV.")
        sys.exit(1)

    # 2. Convert to WAV (Wav2Lip expects WAV)
    if not convert_mp3_to_wav(args.output_mp3, args.output):
        sys.exit(1)


if __name__ == "__main__":
    main()
