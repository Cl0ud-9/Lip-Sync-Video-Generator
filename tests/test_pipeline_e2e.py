"""
End-to-end runs of Wav2Lip/inference.py as a subprocess, exactly as main.py invokes it.

Random-weight models are used (see conftest.random_models), so these verify the plumbing -- frame reading, mel
chunking, batching, encoding, error handling -- not the visual quality of the lip sync.
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import soundfile as sf

from conftest import WAV2LIP_DIR

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("ffmpeg_tools")]

INFERENCE = os.path.join(WAV2LIP_DIR, "inference.py")


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("media")
    rng = np.random.default_rng(0)

    image = (rng.random((193, 255, 3)) * 255).astype(np.uint8)  # odd dimensions on purpose
    cv2.imwrite(str(root / "odd.jpg"), image)
    cv2.imwrite(str(root / "UPPER.PNG"), image)

    writer = cv2.VideoWriter(str(root / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (256, 192))
    for i in range(40):
        writer.write(np.roll(cv2.resize(image, (256, 192)), i * 3, axis=1))
    writer.release()

    def tone(name, seconds, sr=16000):
        t = np.arange(int(sr * seconds)) / sr
        sf.write(str(root / name), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)

    tone("tone2s.wav", 2.0)
    tone("tone4s.wav", 4.0)  # longer than the 1.6 s / 40-frame clip, so the video has to loop
    tone("tone44k.wav", 1.0, sr=44100)
    tone("short.wav", 0.1)
    sf.write(str(root / "empty.wav"), np.zeros(0, dtype=np.float32), 16000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(root / "tone2s.wav"), str(root / "tone.mp3")], check=True)
    return root


@pytest.fixture
def infer(random_models, tmp_path):
    def run(face, audio, *extra, name="out.mp4", check=True):
        outfile = tmp_path / name
        cmd = [sys.executable, INFERENCE, "--checkpoint_path", random_models["checkpoint"], "--face", str(face),
               "--audio", str(audio), "--outfile", str(outfile), "--device", "cpu", *map(str, extra)]
        env = dict(os.environ, S3FD_MODEL_PATH=random_models["s3fd"])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if check:
            assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc, outfile
    return run


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,pix_fmt",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return {s["codec_type"]: s for s in json.loads(out.stdout)["streams"]}


def test_static_image_with_odd_dimensions_produces_playable_h264(media, infer):
    _, out = infer(media / "odd.jpg", media / "tone2s.wav")
    streams = probe(out)
    assert streams["video"]["codec_name"] == "h264" and streams["video"]["pix_fmt"] == "yuv420p"
    assert streams["video"]["width"] % 2 == 0 and streams["video"]["height"] % 2 == 0
    assert streams["audio"]["codec_name"] == "aac"


def test_uppercase_extension_and_non_16k_audio(media, infer):
    _, out = infer(media / "UPPER.PNG", media / "tone44k.wav")
    assert probe(out)["video"]["codec_name"] == "h264"


def test_mp3_audio_is_converted(media, infer):
    _, out = infer(media / "odd.jpg", media / "tone.mp3")
    assert out.exists() and out.stat().st_size > 0


def test_video_input_with_smoothing_and_resize(media, infer):
    _, out = infer(media / "clip.avi", media / "tone2s.wav", "--resize_factor", 2, "--nosmooth")
    assert probe(out)["video"]["width"] == 128
    _, out2 = infer(media / "clip.avi", media / "tone2s.wav", "--resize_factor", 2, name="smooth.mp4")
    assert out2.exists()


def test_very_short_audio_still_yields_a_video(media, infer):
    _, out = infer(media / "odd.jpg", media / "short.wav")
    assert out.exists()


def test_static_flag_on_video_uses_first_frame_only(media, infer):
    _, out = infer(media / "clip.avi", media / "tone2s.wav", "--static")
    assert out.exists()


def test_explicit_box_skips_face_detection(media, infer):
    proc, out = infer(media / "odd.jpg", media / "tone2s.wav", "--box", 10, 150, 10, 200)
    assert "Using the specified bounding box" in proc.stdout and out.exists()


def test_output_directory_is_created(media, infer, tmp_path):
    _, out = infer(media / "odd.jpg", media / "tone2s.wav", name=os.path.join("deep", "er", "o.mp4"))
    assert out.exists()


@pytest.mark.parametrize("face, audio, message", [
    ("odd.jpg", "empty.wav", "No audio samples"),
    ("tone2s.wav", "tone2s.wav", "must be an image or a video file"),   # audio-only --face used to hang forever
    ("missing.jpg", "tone2s.wav", "valid path"),
    ("odd.jpg", "missing.wav", "valid path"),
])
def test_bad_inputs_fail_cleanly(media, infer, face, audio, message):
    proc, out = infer(media / face, media / audio, check=False)
    assert proc.returncode == 1 and message in proc.stdout
    assert not out.exists()


def test_corrupt_image_fails_cleanly(media, infer, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    proc, _ = infer(bad, media / "tone2s.wav", check=False)
    assert proc.returncode == 1 and "Could not decode image" in proc.stdout


def test_non_checkpoint_file_fails_before_any_heavy_work(media, infer, random_models):
    cmd = [sys.executable, INFERENCE, "--checkpoint_path", str(media / "tone2s.wav"), "--face", str(media / "odd.jpg"),
           "--audio", str(media / "tone2s.wav"), "--device", "cpu"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, S3FD_MODEL_PATH=random_models["s3fd"]))
    assert proc.returncode == 1
    assert "Face Detection" not in proc.stderr  # checkpoint is loaded (and rejected) before the slow detection pass


def test_mux_handles_a_genuinely_odd_sized_intermediate_video(media, tmp_path):
    """
    OpenCV usually rounds the intermediate AVI to even dimensions itself, but not every build does, and libx264 with
    yuv420p rejects odd sizes. Build a real odd-sized AVI with ffmpeg to prove the final encode copes either way.
    """
    sys.path.insert(0, WAV2LIP_DIR)
    import inference

    odd_avi = tmp_path / "odd.avi"
    # Raw frames with an explicit size on stdin: nothing (a lavfi source, a filter, a yuv420 encoder) gets the chance
    # to quietly round 127x95 down to 126x94, which some ffmpeg builds do and which would make this test vacuous.
    frames = bytes(127 * 95 * 3) * 25
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "127x95",
                    "-r", "25", "-i", "-", "-c:v", "rawvideo", "-pix_fmt", "bgr24", str(odd_avi)],
                   input=frames, check=True)
    assert probe(odd_avi)["video"]["width"] == 127  # really odd

    out = tmp_path / "muxed.mp4"
    inference.mux_audio_video(str(media / "tone2s.wav"), str(odd_avi), str(out))
    video = probe(out)["video"]
    assert video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p"
    assert (video["width"], video["height"]) == (126, 94)


def test_mux_never_trims_the_audio_track(media, tmp_path):
    """The generated video is slightly shorter than the speech; the end of the last word must not be cut off."""
    sys.path.insert(0, WAV2LIP_DIR)
    import inference

    short_avi = tmp_path / "short.avi"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=1:r=25",
                    "-c:v", "mpeg4", str(short_avi)], check=True)  # 1 s of video vs a 2 s tone
    out = tmp_path / "muxed.mp4"
    inference.mux_audio_video(str(media / "tone2s.wav"), str(short_avi), str(out))

    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration",
                             "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True)
    assert float(result.stdout.strip()) >= 1.95


def test_video_shorter_than_audio_loops_and_covers_the_whole_audio(media, infer):
    """clip.avi has 40 frames; 4 s of audio needs ~97, so the (streamed) video must wrap around."""
    _, out = infer(media / "clip.avi", media / "tone4s.wav", "--resize_factor", 2, "--nosmooth")
    capture = cv2.VideoCapture(str(out))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert 95 <= frames <= 100, frames
