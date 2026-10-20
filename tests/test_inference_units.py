"""Unit tests for the helpers in Wav2Lip/inference.py."""

import argparse
import sys

import cv2
import numpy as np
import pytest
import soundfile as sf

import inference


# --- small helpers -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [("true", True), ("T", True), ("1", True), ("no", False), ("0", False)])
def test_str2bool(value, expected):
    assert inference._str2bool(value) is expected


def test_str2bool_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        inference._str2bool("maybe")


@pytest.mark.parametrize("path, expected", [("a.jpg", True), ("A.JPEG", True), ("x.PNG", True), ("v.mp4", False), ("noext", False)])
def test_is_image(path, expected):
    assert inference._is_image(path) is expected


def test_is_out_of_memory():
    assert inference._is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert inference._is_out_of_memory(RuntimeError("DefaultCPUAllocator: can't allocate memory"))
    assert not inference._is_out_of_memory(RuntimeError("size mismatch for conv1.weight"))


# --- mel chunking -------------------------------------------------------------------------------------------------

def test_mel_chunks_normal_length():
    mel = np.random.rand(80, 161).astype(np.float32)  # 2 s of audio
    chunks = inference.build_mel_chunks(mel, fps=25)
    assert len(chunks) == 47
    assert all(c.shape == (80, 16) for c in chunks)
    np.testing.assert_array_equal(chunks[0], mel[:, :16])
    np.testing.assert_array_equal(chunks[-1], mel[:, -16:])  # last window is clamped to the end, never past it


def test_mel_chunks_shorter_than_one_window_is_padded():
    mel = np.random.rand(80, 5).astype(np.float32)
    chunks = inference.build_mel_chunks(mel, fps=25)
    # (as upstream, a final clamped window is always appended after the last full one, hence 2 for a 16-step mel)
    assert len(chunks) == 2 and all(c.shape == (80, 16) for c in chunks)
    np.testing.assert_array_equal(chunks[0][:, :5], mel)
    np.testing.assert_array_equal(chunks[0][:, 5:], np.repeat(mel[:, -1:], 11, axis=1))


@pytest.mark.parametrize("fps", [12.5, 24, 29.97, 60])
def test_mel_chunks_always_full_width(fps):
    chunks = inference.build_mel_chunks(np.random.rand(80, 400).astype(np.float32), fps=fps)
    assert chunks and all(c.shape == (80, 16) for c in chunks)


# --- box smoothing --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 9])
def test_smoothing_constant_boxes_unchanged(n):
    boxes = np.tile(np.array([[10, 20, 110, 120]]), (n, 1))
    np.testing.assert_array_equal(inference.get_smoothened_boxes(boxes.copy(), T=5), boxes)


def test_smoothing_short_clip_uses_all_frames():
    boxes = np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]])
    smoothed = inference.get_smoothened_boxes(boxes.copy(), T=5)
    assert smoothed[0].tolist() == [10, 10, 20, 20]  # mean of all three, not just the tail


# --- face_detect (with a fake detector, so no weights needed) -------------------------------------------------------

class FakeDetector:
    """Stands in for face_detection.FaceAlignment; raises OOM for batches larger than `max_batch`."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        FakeDetector.instances.append(self)

    max_batch = 10**9
    fail_with = None
    result = (5, 5, 60, 70)

    def get_detections_for_batch(self, images):
        self.calls.append(len(images))
        if self.fail_with is not None:
            raise self.fail_with
        if len(images) > FakeDetector.max_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
        return [FakeDetector.result] * len(images)


@pytest.fixture
def fake_detector(monkeypatch):
    FakeDetector.instances = []
    FakeDetector.max_batch, FakeDetector.fail_with, FakeDetector.result = 10**9, None, (5, 5, 60, 70)
    monkeypatch.setattr(inference.face_detection, "FaceAlignment", FakeDetector)
    monkeypatch.setattr(inference.cv2, "imwrite", lambda *a, **k: True)  # don't litter Wav2Lip/temp
    return FakeDetector


def _frames(n, h=100, w=120):
    return [np.zeros((h, w, 3), np.uint8) for _ in range(n)]


def test_face_detect_crops_with_padding_and_clamps(fake_detector):
    out = inference.face_detect(_frames(2), "cpu", 16, [0, 10, 0, 0], nosmooth=True)
    (crop, coords), _ = out
    # detector box is (x1=5, y1=5, x2=60, y2=70); bottom pad of 10 -> y2=80; coords are (y1, y2, x1, x2)
    assert coords == (5, 80, 5, 60)
    assert crop.shape == (75, 55, 3)


def test_face_detect_pads_never_leave_the_image(fake_detector):
    fake_detector.result = (0, 0, 120, 100)
    (_, (y1, y2, x1, x2)), = inference.face_detect(_frames(1), "cpu", 16, [50, 50, 50, 50], nosmooth=True)
    assert (y1, y2, x1, x2) == (0, 100, 0, 120)


def test_face_detect_halves_batch_on_real_oom(fake_detector):
    fake_detector.max_batch = 2
    out = inference.face_detect(_frames(8), "cpu", 8, [0, 0, 0, 0], nosmooth=True)
    assert len(out) == 8
    assert fake_detector.instances[0].calls[:2] == [8, 4]  # 8 -> OOM, 4 -> OOM, then succeeds at 2


def test_face_detect_does_not_mask_unrelated_runtime_errors(fake_detector):
    fake_detector.fail_with = RuntimeError("size mismatch for conv1.weight")
    with pytest.raises(RuntimeError, match="size mismatch"):
        inference.face_detect(_frames(4), "cpu", 4, [0, 0, 0, 0], nosmooth=True)
    assert fake_detector.instances[0].calls == [4]  # no pointless retry loop


def test_face_detect_gives_up_at_batch_size_one(fake_detector):
    fake_detector.max_batch = 0
    with pytest.raises(RuntimeError, match="resize_factor"):
        inference.face_detect(_frames(2), "cpu", 2, [0, 0, 0, 0], nosmooth=True)


def test_face_detect_no_face_raises(fake_detector):
    fake_detector.result = None
    with pytest.raises(ValueError, match="Face not detected"):
        inference.face_detect(_frames(2), "cpu", 2, [0, 0, 0, 0], nosmooth=True)


# --- reading inputs ---------------------------------------------------------------------------------------------------

def _args(**kw):
    base = dict(face="", resize_factor=1, rotate=False, crop=[0, -1, 0, -1])
    base.update(kw)
    return argparse.Namespace(**base)


def test_read_image_from_non_ascii_path(tmp_path):
    folder = tmp_path / "façé-ünï"
    folder.mkdir()
    path = folder / "phóto.jpg"
    cv2.imencode(".jpg", np.full((40, 60, 3), 128, np.uint8))[1].tofile(str(path))
    frames = inference.read_frames(_args(face=str(path)), max_frames=99)
    assert len(frames) == 1 and frames[0].shape == (40, 60, 3)


def test_read_image_resize_never_reaches_zero(tmp_path):
    path = tmp_path / "tiny.png"
    cv2.imwrite(str(path), np.zeros((3, 3, 3), np.uint8))
    frames = inference.read_frames(_args(face=str(path), resize_factor=10), max_frames=1)
    assert frames[0].shape[:2] == (1, 1)


def test_read_corrupt_image_raises(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_text("definitely not a jpeg")
    with pytest.raises(ValueError, match="Could not decode"):
        inference.read_frames(_args(face=str(path)), max_frames=1)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, np.uint8))
    writer.release()
    return str(path)


def test_read_video_stops_at_max_frames(video):
    assert len(inference.read_frames(_args(face=video), max_frames=7)) == 7
    assert len(inference.read_frames(_args(face=video), max_frames=1000)) == 30


def test_read_video_resize_rotate_crop(video):
    frames = inference.read_frames(_args(face=video, resize_factor=2, rotate=True, crop=[0, 20, 0, -1]), max_frames=1)
    # 64x48 -> 32x24 after resize -> rotated to 24x32 (w x h) -> cropped to 20 rows
    assert frames[0].shape[:2] == (20, 24)


def test_video_fps_read(video):
    assert inference.get_video_fps(video, default_fps=99) == pytest.approx(25)


def test_video_fps_falls_back_when_container_reports_none(video, monkeypatch):
    real = cv2.VideoCapture

    class NoFps:
        def __init__(self, path):
            self._cap = real(path)

        def get(self, prop):
            return 0.0 if prop == cv2.CAP_PROP_FPS else self._cap.get(prop)

        def isOpened(self):
            return self._cap.isOpened()

        def release(self):
            self._cap.release()

    monkeypatch.setattr(inference.cv2, "VideoCapture", NoFps)
    assert inference.get_video_fps(video, default_fps=30) == 30


def test_audio_only_file_is_rejected_instead_of_hanging(tmp_path):
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="no video stream"):
        inference.get_video_fps(str(wav), default_fps=25)


def test_missing_video_raises(tmp_path):
    with pytest.raises(ValueError):
        inference.get_video_fps(str(tmp_path / "nope.mp4"), default_fps=25)


# --- writer + CLI validation -------------------------------------------------------------------------------------------

def test_open_video_writer_returns_open_writer(tmp_path):
    writer = inference.open_video_writer(str(tmp_path / "o.avi"), 25, (64, 48))
    try:
        assert writer.isOpened()
    finally:
        writer.release()


@pytest.mark.parametrize("extra, message", [
    (["--pads", "1", "2"], "exactly 4"),
    (["--crop", "1", "2", "3"], "exactly 4"),
    (["--box", "1", "2", "3", "4", "5"], "exactly 4"),
    (["--wav2lip_batch_size", "0"], "at least 1"),
    (["--face_det_batch_size", "-3"], "at least 1"),
    (["--resize_factor", "0"], "at least 1"),
    (["--fps", "0"], "--fps"),
    (["--fps", "nan"], "--fps"),
])
def test_parse_args_rejects_bad_values(monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["inference.py", "--checkpoint_path", "c", "--face", "f", "--audio", "a", *extra])
    with pytest.raises(SystemExit) as exc:
        inference.parse_args()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["inference.py", "--checkpoint_path", "c", "--face", "f", "--audio", "a"])
    args = inference.parse_args()
    assert args.img_size == 96 and args.pads == [0, 10, 0, 0] and args.static is False
