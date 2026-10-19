# 🎬 Lip Sync Video Generator

An AI-powered pipeline that transforms text into realistic lip-synced talking face videos using **ElevenLabs Text-to-Speech** and **Wav2Lip**.

Perfect for AI demos, virtual presenters, educational content, and speech-driven facial animation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> ⚠️ **License note:** the top-level MIT license covers this repo's own code
> (`main.py`, `Elevenlab.py`, project scaffolding). The vendored `Wav2Lip/`
> directory is **not** MIT - it remains under the original authors'
> personal/research/non-commercial-only terms. See
> [`Wav2Lip/NOTICE.md`](Wav2Lip/NOTICE.md) before any commercial use.

---

## 🎥 What This Does

Transform a simple text script and a face image into a fully lip-synced video:

```text
📝 Text Script → 🎙️ AI Speech → 👄 Lip Sync → 🎬 Final Video
```

**Input:** Text + Face Image  
**Output:** Realistic talking face video with synchronized lips

---

## ✨ Features

- 🗣️ **Natural Speech Synthesis** - Powered by ElevenLabs TTS API  
- 👄 **Accurate Lip Synchronization** - Using state-of-the-art Wav2Lip  
- 🤖 **Smart Pipeline** - Auto-detects audio or script inputs  
- ⚡ **GPU Acceleration** - CUDA support for faster processing (with a `--device` override)  
- 📂 **Organized Workflow** - Clean input/output structure  
- 🚀 **One-Click Execution** - Run `main.py` and you're done  
- 🛠️ **Configurable CLI** - Override face/audio paths, batch size, resize factor, and more  
- ✅ **Pre-flight Checks** - Verifies ffmpeg and model files *before* doing any work  

---

## 🧠 Pipeline Overview

```text
[Input: Script or Audio] → 🤖 main.py (Auto-Pipeline) → [Output: Lip-Synced Video]
```

---

## 📁 Project Structure

```text
lip-sync-video-generator/
│
├── input/                               # Your input files
│   ├── script.txt                       # Text to convert into speech
│   └── face.jpg                         # Face image (front-facing)
│
├── output/                              # Generated results
│   ├── audio.wav                        # Generated speech audio
│   └── output_video.mp4                 # Final lip-synced video
│
├── Wav2Lip/                             # Wav2Lip model and scripts (vendored, own license -- see NOTICE.md)
│   ├── checkpoints/
│   │   └── wav2lip.pth                  # Wav2Lip model file
│   ├── face_detection/detection/sfd/
│   │   └── s3fd.pth                     # Face detection model
│   └── NOTICE.md                        # Upstream license terms & citation
│
├── main.py                              # 🚀 Unified Pipeline (Run this!)
├── Elevenlab.py                         # Text-to-speech generator
├── requirements.txt                     # Python dependencies (pinned)
├── .env                                 # API key (you create this)
└── README.md
```

---

## 📦 Requirements

| Requirement | Purpose |
|------------|---------|
| **Python 3.9+** | Core runtime |
| **FFmpeg** | Video processing |
| **ElevenLabs API Key** | Speech generation |
| **NVIDIA GPU + CUDA** *(Optional)* | Faster processing |

---

## 🎞️ Install FFmpeg (Windows)

```bash
winget install Gyan.FFmpeg
```

Restart your terminal after installation.

### Verify Installation

```bash
ffmpeg -version
```

---

## 🛠️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Cl0ud-9/lip-sync-video-generator.git
```

Then open it in your code editor.

---

### 2️⃣ Set Up a Virtual Environment

#### Create a Virtual Environment
```bash
python -m venv .venv
```
#### Activate Environment
```bash
.\.venv\Scripts\activate
```

---

### 3️⃣ Install PyTorch

#### CPU Device Only
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### NVIDIA GPU Device (CUDA 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### 4️⃣ Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

### 5️⃣ Configure ElevenLabs API Key

Get your API key from: 
https://elevenlabs.io/developers

Create a `.env` file in the root directory:

( Rename the `.env.example` file to `.env` and add your API key )

```
ELEVENLABS_API_KEY=your_api_key_here
```

---

## ⬇️ Download Required Model Files

### 🔹 Wav2Lip Model

Download:  
https://drive.google.com/uc?id=1fQtBSYEyuai9MjBOF8j7zZ4oQ9W2N64q  

Place here:

```
Wav2Lip/checkpoints/wav2lip.pth
```

---

### 🔹 Face Detection Model (S3FD)

Download:  
https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth

Rename to:

```
s3fd.pth
```

Place here:

```
Wav2Lip/face_detection/detection/sfd/s3fd.pth
```

---

## 🎯 Usage (The Easy Way)

### 1️⃣ Prepare Input
- **Face**: Put your image or video in `input/` (e.g., `input/face.jpg` or `input/face.mp4`).
- **Audio**:
    - **Option A (Text Script)**: Put your script in `input/script.txt`.
    - **Option B (Audio File)**: Put your audio in `input/audio.wav`.

### 2️⃣ Run
```bash
python main.py
```
That's it! The script will automatically detect your input and generate the video.

**Output:** `output/output_video.mp4`

### ⚙️ Optional Flags

`main.py` also accepts flags to override the auto-detected defaults -- useful if you
want to keep multiple face/audio files in `input/` or tune performance:

| Flag | Default | Description |
|------|---------|--------------|
| `--face PATH` | auto-detected | Explicit face image/video path |
| `--audio PATH` | auto-detected | Explicit audio path (WAV, or anything ffmpeg can read, e.g. MP3); skips ElevenLabs |
| `--script PATH` | `input/script.txt` | Script to use for ElevenLabs TTS |
| `--output PATH` | `output/output_video.mp4` | Where to save the final video |
| `--checkpoint PATH` | `Wav2Lip/checkpoints/wav2lip.pth` | Wav2Lip model checkpoint |
| `--device {auto,cuda,cpu}` | `auto` | Force a compute device |
| `--resize-factor N` | `2` | Downscale input frames (try `4` on CPU-only machines) |
| `--batch-size N` | `128` | Wav2Lip batch size (lower this on out-of-memory errors) |
| `--smooth` | off | Enable temporal smoothing of face boxes |
| `--keep-temp` | off | Keep `Wav2Lip/temp/` intermediate files after a run |

Example:
```bash
python main.py --face input/my_face.mp4 --resize-factor 4 --batch-size 32 --device cpu
```

---

## 🔧 Advanced / Manual Usage

If you want even more control, you can run the scripts individually.

### 🎙️ Step 1 - Generate Speech Audio (Optional)
```bash
python Elevenlab.py --script input/script.txt --output output/audio.wav
```

### 🎬 Step 2 - Generate Lip-Synced Video
```bash
python Wav2Lip/inference.py --checkpoint_path Wav2Lip/checkpoints/wav2lip.pth --face input/face.jpg --audio input/audio.wav --outfile output/output_video.mp4 --resize_factor 2 --nosmooth --wav2lip_batch_size 128 --device auto
```

---

## ⚙️ Verify PyTorch GPU Access

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|------|----------|
| FFmpeg not found | `main.py` now checks for this up front and tells you before running anything; restart your terminal after installing |
| CUDA not detected | Install GPU PyTorch build, or pass `--device cpu` explicitly |
| Out of memory / CUDA OOM | Lower `--batch-size` (try 32 or 16) and/or raise `--resize-factor` |
| Blurry lips | Use a better face crop |
| Model not found | Check file paths under `Wav2Lip/checkpoints/` and `Wav2Lip/face_detection/detection/sfd/` |
| Slow processing | Use GPU (`--device cuda`) |
| API key error | Verify `.env` file has `ELEVENLABS_API_KEY` set (only needed when generating speech from `script.txt`; not needed with `--audio` / `input/audio.wav`) |
| `has no video stream` | `--face` must be an image (`.jpg`/`.jpeg`/`.png`) or a video file, not an audio file |
| `Face not detected!` | The offending frame is saved to `Wav2Lip/temp/faulty_frame.jpg`; use a clearer, front-facing face or a lower `--resize-factor` |
| `ModuleNotFoundError: audioop` / `pyaudioop` | Shouldn't happen anymore -- MP3→WAV conversion now shells out to ffmpeg directly instead of the `pydub` package, which broke on Python 3.13+ |

---

## 📜 Acknowledgements

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)
- [ElevenLabs](https://elevenlabs.io)

---

## 📚 Technical Overview

1. ElevenLabs generates speech  
2. S3FD detects face  
3. Wav2Lip generates lip motion  
4. FFmpeg renders final video  

---

## ⚖️ License

This project's own code is licensed under the **MIT License** - see [LICENSE](LICENSE).

The vendored `Wav2Lip/` directory is **not** covered by that MIT license - it
remains under the original authors' personal/research/non-commercial-only
terms. See [Wav2Lip/NOTICE.md](Wav2Lip/NOTICE.md) for the full text and
citation requirements before any commercial use.

---

## 📌 Disclaimer

For educational and research use only.
Ensure consent before using any person's face or voice. `input/script.txt`
ships with placeholder demo text only - replace it with your own script
before generating anything, and never use this pipeline to impersonate a
real person (e.g. fake payment-collection or "urgent action needed" calls)
without their explicit consent.

---

**Made with ❤️ for the AI community**








