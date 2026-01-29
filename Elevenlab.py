#!pip install requests pydub
#!apt-get install ffmpeg

import requests
import os
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

def generate_audio_with_elevenlabs(script_path, output_audio_path, voice_id, api_key,model_id="eleven_turbo_v2_5", stability=0.6, similarity_boost=0.5):
    """
    Reads text from the given script file and sends a request to the ElevenLabs API to generate speech.
    The generated audio (MP3) is saved to output_audio_path.

    Parameters:
      - script_path: Path to the text file containing your script.
      - output_audio_path: Path where the generated MP3 audio will be saved.
      - voice_id: ElevenLabs voice ID (choose one with an Indian accent).
      - api_key: Your ElevenLabs API key.
      - stability: (Optional) Controls voice stability (default: 0.5).
      - similarity_boost: (Optional) Controls similarity boost (default: 0.5).
    """
    # Read the input text
    with open(script_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

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
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        with open(output_audio_path, "wb") as out_file:
            out_file.write(response.content)
        print(f"[INFO] Audio saved at: {output_audio_path}")
    else:
        print("Error:", response.status_code, response.text)

def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Converts an MP3 file to WAV format.
    """
    sound = AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")
    print(f"[INFO] Converted MP3 to WAV at: {wav_path}")

# Usage Example
if __name__ == "__main__":
    import argparse

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
        print("Error: ELEVENLABS_API_KEY not found in environment variables.")
        exit(1)

    # Ensure output directory exists (for robust CLI usage)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Generate audio
    generate_audio_with_elevenlabs(args.script, args.output_mp3, args.voice_id, api_key)

    # Convert to WAV
    convert_mp3_to_wav(args.output_mp3, args.output)
