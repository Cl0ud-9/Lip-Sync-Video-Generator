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


# --- detect_face_boxes (with a fake detector, so no weights needed) -----------------------------------------------

class FakeDetector:
    """Stands in for face_detection.FaceAlignment; raises OOM for batches larger than `max_batch`."""
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        FakeDetector.instances.append(self)

    max_batch = 10**9
    fail_with = None
    result = (5, 5, 60, 70)
    on_call = None

    def get_detections_for_batch(self, images):
        self.calls.append(len(images))
        if FakeDetector.on_call:
            FakeDetector.on_call(images)
        if self.fail_with is not None:
            raise self.fail_with
        if len(images) > FakeDetector.max_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
        return [FakeDetector.result] * len(images)


@pytest.fixture
def fake_detector(monkeypatch):
    FakeDetector.instances = []
    FakeDetector.max_batch, FakeDetector.fail_with, FakeDetector.result = 10**9, None, (5, 5, 60, 70)
    FakeDetector.on_call = None
    monkeypatch.setattr(inference.face_detection, "FaceAlignment", FakeDetector)
    monkeypatch.setattr(inference.cv2, "imwrite", lambda *a, **k: True)  # don't litter Wav2Lip/temp
    return FakeDetector


def _frames(n, h=100, w=120):
    return [np.zeros((h, w, 3), np.uint8) for _ in range(n)]


def test_boxes_include_padding_and_use_y1_y2_x1_x2_order(fake_detector):
    boxes = inference.detect_face_boxes(iter(_frames(2)), "cpu", 16, [0, 10, 0, 0], nosmooth=True)
    # detector box is (x1=5, y1=5, x2=60, y2=70); bottom pad of 10 -> y2=80
    assert boxes == [(5, 80, 5, 60)] * 2


def test_boxes_never_leave_the_image(fake_detector):
    fake_detector.result = (0, 0, 120, 100)
    boxes = inference.detect_face_boxes(iter(_frames(1)), "cpu", 16, [50, 50, 50, 50], nosmooth=True)
    assert boxes == [(0, 100, 0, 120)]


def test_one_box_per_frame_across_batches(fake_detector):
    boxes = inference.detect_face_boxes(iter(_frames(37)), "cpu", 8, [0, 0, 0, 0], nosmooth=True)
    assert len(boxes) == 37
    assert fake_detector.instances[0].calls == [8, 8, 8, 8, 5]


def test_detection_streams_instead_of_holding_every_frame(fake_detector):
    """Memory must not grow with video length: only about one detection batch of frames may be alive at a time."""
    import gc
    import weakref

    live = []
    peak = [0]

    def source():
        for _ in range(100):
            frame = np.zeros((50, 50, 3), np.uint8)
            live.append(weakref.ref(frame))
            yield frame

    def on_call(images):
        gc.collect()
        peak[0] = max(peak[0], sum(1 for ref in live if ref() is not None))

    fake_detector.on_call = on_call
    inference.detect_face_boxes(source(), "cpu", 4, [0, 0, 0, 0], nosmooth=True)
    assert peak[0] <= 4 + 2, f"{peak[0]} frames alive at once"


def test_oom_halves_batch_and_remembers_it(fake_detector):
    fake_detector.max_batch = 2
    boxes = inference.detect_face_boxes(iter(_frames(8)), "cpu", 8, [0, 0, 0, 0], nosmooth=True)
    assert len(boxes) == 8
    assert fake_detector.instances[0].calls == [8, 4, 2, 2, 2, 2]  # 8 and 4 fail; sizes known to fail aren't retried


def test_later_chunks_start_from_the_reduced_batch_size(fake_detector):
    fake_detector.max_batch = 2
    inference.detect_face_boxes(iter(_frames(12)), "cpu", 4, [0, 0, 0, 0], nosmooth=True)
    assert fake_detector.instances[0].calls == [4, 2, 2, 2, 2, 2, 2]  # one failure, then 2 for everything else


def test_unrelated_runtime_errors_are_not_masked(fake_detector):
    fake_detector.fail_with = RuntimeError("size mismatch for conv1.weight")
    with pytest.raises(RuntimeError, match="size mismatch"):
        inference.detect_face_boxes(iter(_frames(4)), "cpu", 4, [0, 0, 0, 0], nosmooth=True)
    assert fake_detector.instances[0].calls == [4]  # no pointless retry loop


def test_gives_up_at_batch_size_one(fake_detector):
    fake_detector.max_batch = 0
    with pytest.raises(RuntimeError, match="resize_factor"):
        inference.detect_face_boxes(iter(_frames(2)), "cpu", 2, [0, 0, 0, 0], nosmooth=True)


def test_no_face_raises(fake_detector):
    fake_detector.result = None
    with pytest.raises(ValueError, match="Face not detected"):
        inference.detect_face_boxes(iter(_frames(2)), "cpu", 2, [0, 0, 0, 0], nosmooth=True)


def test_empty_input_gives_no_boxes(fake_detector):
    assert inference.detect_face_boxes(iter([]), "cpu", 4, [0, 0, 0, 0], nosmooth=False) == []


# --- reading inputs ---------------------------------------------------------------------------------------------------

def read_frames(args, max_frames):
    return list(inference.iter_frames(args, max_frames))


def _args(**kw):
    base = dict(face="", resize_factor=1, rotate=False, crop=[0, -1, 0, -1])
    base.update(kw)
    return argparse.Namespace(**base)


def test_read_image_from_non_ascii_path(tmp_path):
    folder = tmp_path / "façé-ünï"
    folder.mkdir()
    path = folder / "phóto.jpg"
    cv2.imencode(".jpg", np.full((40, 60, 3), 128, np.uint8))[1].tofile(str(path))
    frames = read_frames(_args(face=str(path)), max_frames=99)
    assert len(frames) == 1 and frames[0].shape == (40, 60, 3)


def test_read_image_resize_never_reaches_zero(tmp_path):
    path = tmp_path / "tiny.png"
    cv2.imwrite(str(path), np.zeros((3, 3, 3), np.uint8))
    frames = read_frames(_args(face=str(path), resize_factor=10), max_frames=1)
    assert frames[0].shape[:2] == (1, 1)


def test_read_corrupt_image_raises(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_text("definitely not a jpeg")
    with pytest.raises(ValueError, match="Could not decode"):
        read_frames(_args(face=str(path)), max_frames=1)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, np.uint8))
    writer.release()
    return str(path)


def test_read_video_stops_at_max_frames(video):
    assert len(read_frames(_args(face=video), max_frames=7)) == 7
    assert len(read_frames(_args(face=video), max_frames=1000)) == 30


def test_read_video_resize_rotate_crop(video):
    frames = read_frames(_args(face=video, resize_factor=2, rotate=True, crop=[0, 20, 0, -1]), max_frames=1)
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


# --- streaming: frame_stream / datagen ----------------------------------------------------------------------------------

def _stream_args(face, **kw):
    base = dict(face=face, resize_factor=1, rotate=False, crop=[0, -1, 0, -1], static=False, box=[-1, -1, -1, -1],
                img_size=96, wav2lip_batch_size=8)
    base.update(kw)
    return argparse.Namespace(**base)


def test_frame_stream_loops_back_to_the_start(video):
    stream = inference.frame_stream(_stream_args(video), limit=10)
    items = [next(stream) for _ in range(25)]
    assert [i for i, _ in items] == list(range(10)) * 2 + list(range(5))
    # frame i of the clip is a flat grey of value ~i*8, so the second lap really is the start of the video again
    assert abs(float(items[12][1].mean()) - 2 * 8) < 6 and abs(float(items[2][1].mean()) - 2 * 8) < 6


def test_frame_stream_static_repeats_one_frame(tmp_path):
    path = tmp_path / "f.png"
    cv2.imwrite(str(path), np.full((20, 30, 3), 77, np.uint8))
    stream = inference.frame_stream(_stream_args(str(path), static=True), limit=1)
    (i1, f1), (i2, f2) = next(stream), next(stream)
    assert i1 == i2 == 0 and f1 is f2  # shared on purpose; main() copies before pasting the generated mouth


def test_frame_stream_with_unreadable_video_raises(tmp_path):
    empty = tmp_path / "empty.avi"
    empty.write_bytes(b"")
    with pytest.raises((RuntimeError, ValueError)):
        next(inference.frame_stream(_stream_args(str(empty)), limit=5))


def _mels(n):
    return [np.zeros((80, 16), np.float32) for _ in range(n)]


def test_datagen_pairs_each_frame_with_its_own_box_and_loops_with_the_audio(video):
    boxes = [(i, i + 20, i, i + 30) for i in range(30)]  # 30-frame clip, each frame gets a distinct box
    batches = list(inference.datagen(boxes, _mels(35), _stream_args(video)))
    coords = [c for _, _, _, cs in batches for c in cs]
    frames = [f for _, _, fs, _ in batches for f in fs]
    assert [len(b[2]) for b in batches] == [8, 8, 8, 8, 3]
    assert coords == [boxes[i % 30] for i in range(35)]
    # the frame content follows the same loop as the boxes
    assert abs(float(frames[31].mean()) - 1 * 8) < 6 and abs(float(frames[20].mean()) - 20 * 8) < 6


def test_datagen_static_resizes_the_face_once_and_shares_the_frame(tmp_path, monkeypatch):
    path = tmp_path / "f.png"
    cv2.imwrite(str(path), np.full((100, 120, 3), 90, np.uint8))
    resizes = []
    real_resize = cv2.resize
    monkeypatch.setattr(inference.cv2, "resize", lambda *a, **k: resizes.append(1) or real_resize(*a, **k))
    batches = list(inference.datagen([(10, 80, 10, 100)], _mels(20), _stream_args(str(path), static=True)))
    assert sum(len(b[2]) for b in batches) == 20
    assert len(resizes) == 1
    frames = batches[0][2]
    assert all(f is frames[0] for f in frames)


def test_datagen_uses_the_constant_box_when_given_no_detections(video):
    batches = list(inference.datagen(None, _mels(5), _stream_args(video, box=[2, 40, 3, 50])))
    assert all(c == (2, 40, 3, 50) for _, _, _, cs in batches for c in cs)


def test_datagen_batch_shapes(video):
    img, mel, frames, coords = next(inference.datagen([(0, 40, 0, 60)] * 30, _mels(10), _stream_args(video)))
    assert img.shape == (8, 96, 96, 6) and img.dtype == np.float32
    assert mel.shape == (8, 80, 16, 1)
    assert float(img[..., 3:].max()) <= 1.0
    assert float(img[:, 48:, :, :3].max()) == 0.0  # lower half of the "masked" copy is blanked
