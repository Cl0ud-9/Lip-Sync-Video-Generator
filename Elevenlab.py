"""
ElevenLabs Text-to-Speech Integration.

This module handles the interaction with the ElevenLabs API to generate speech from text.
It also includes utility functions to convert the generated audio formats if necessary.

Dependencies:
    - requests
    - pydub
    - python-dotenv
    - ffmpeg (installed systematically)
"""

import argparse
import os
import requests
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables once at module level
load_dotenv()


def generate_audio_with_elevenlabs(
    script_path: str,
    output_audio_path: str,
    voice_id: str,
    api_key: str,
    model_id: str = "eleven_turbo_v2_5",
    stability: float = 0.6,
    similarity_boost: float = 0.5
) -> None:
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
    """
    # Read the input text
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        print(f"[ERROR] Script file not found: {script_path}")
        return

    if not text:
        print("[ERROR] Script file is empty.")
        return

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
    
    try:
        print(f"[INFO] Sending request to ElevenLabs (Voice ID: {voice_id})...")
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            with open(output_audio_path, "wb") as out_file:
                out_file.write(response.content)
            print(f"[INFO] Audio saved at: {output_audio_path}")
        else:
            print(f"[ERROR] ElevenLabs API returned code {response.status_code}: {response.text}")
            
    except requests.RequestException as e:
        print(f"[ERROR] Request failed: {e}")


def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    """
    Converts an MP3 file to WAV format using pydub.

    Args:
        mp3_path (str): Path to the source MP3 file.
        wav_path (str): Path to the destination WAV file.
    """
    try:
        print(f"[INFO] Converting {mp3_path} to {wav_path}...")
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
        print(f"[INFO] Conversion successful.")
    except Exception as e:
        print(f"[ERROR] Failed to convert MP3 to WAV: {e}")
        # Ensure 'ffmpeg' is installed and in PATH usually requires user action, but we log the error clearly.


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
        exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Generate Audio (MP3)
    generate_audio_with_elevenlabs(args.script, args.output_mp3, args.voice_id, api_key)

    # 2. Convert to WAV (Wav2Lip expects WAV)
    if os.path.exists(args.output_mp3):
        convert_mp3_to_wav(args.output_mp3, args.output)
    else:
        print("[ERROR] Audio generation failed, cannot convert to WAV.")
        exit(1)


if __name__ == "__main__":
    main()
