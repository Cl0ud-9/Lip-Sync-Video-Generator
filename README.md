# 🎬 Lip Sync Video Generator

An AI-powered pipeline that transforms text into realistic lip-synced talking face videos using **ElevenLabs Text-to-Speech** and **Wav2Lip**.

Perfect for AI demos, virtual presenters, educational content, and speech-driven facial animation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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
- ⚡ **GPU Acceleration** - CUDA support for faster processing  
- 📂 **Organized Workflow** - Clean input/output structure  
- 🚀 **One-Click Execution** - Run `main.py` and you're done  

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
├── Wav2Lip/                             # Wav2Lip model and scripts
│   ├── checkpoints/
│   │   └── wav2lip.pth                  # Wav2Lip model file
│   └── face_detection/detection/sfd/
│       └── s3fd.pth                     # Face detection model
│
├── main.py                              # 🚀 Unified Pipeline (Run this!)
├── Elevenlab.py                         # Text-to-speech generator
├── requirements.txt                     # Python dependencies
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

(Rename the `.env.example` file to `.env` and add your API key)

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
- **Face**: Put your image or video in `input/` (e.g., `input/face.jpg`).
- **Audio**:
    - **Option A (Text Script)**: Put your script in `input/script.txt`.
    - **Option B (Audio File)**: Put your audio in `input/audio.wav`.

### 2️⃣ Run
```bash
python main.py
```
That's it! The script will automatically detect your input and generate the video.

**Output:** `output/output_video.mp4`

---

## 🔧 Advanced / Manual Usage

If you want more control (like specific resize factors or specific file paths), you can run the scripts individually.

### 🎙️ Step 1 — Generate Speech Audio (Optional)
```bash
python Elevenlab.py --script input/script.txt --output output/audio.wav
```

### 🎬 Step 2 — Generate Lip-Synced Video
```bash
python Wav2Lip/inference.py --checkpoint_path Wav2Lip/checkpoints/wav2lip.pth --face input/image.jpg --audio input/audio.wav --outfile output/output_video.mp4 --resize_factor 2 --nosmooth --wav2lip_batch_size 256
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
| FFmpeg not found | Restart terminal |
| CUDA not detected | Install GPU PyTorch build |
| Blurry lips | Use a better face crop |
| Model not found | Check file paths |
| Slow processing | Use GPU |
| API key error | Verify `.env` file |

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

Licensed under the **MIT License** — see [LICENSE](LICENSE).

---

## 📌 Disclaimer

For educational and research use only.  
Ensure consent before using any person's face or voice.

---

**Made with ❤️ for the AI community**






