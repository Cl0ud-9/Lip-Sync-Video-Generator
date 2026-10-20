"""
Shared test setup.

Nothing here needs the real Wav2Lip / S3FD weights or network access: tests that need a model build one with
random weights, which is enough to exercise the pipeline's mechanics (not its output quality).
"""

import os
import shutil
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAV2LIP_DIR = os.path.join(REPO_ROOT, "Wav2Lip")

# main.py / Elevenlab.py live at the repo root; inference.py's own imports (audio, models, face_detection)
# are resolved relative to Wav2Lip/, exactly as when it is run as a script.
for path in (REPO_ROOT, WAV2LIP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end tests that build random-weight models (tens of seconds)")


@pytest.fixture(scope="session")
def ffmpeg_tools():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")


@pytest.fixture(scope="session")
def random_models(tmp_path_factory):
    """Random-weight Wav2Lip checkpoint + S3FD weights (the detector then 'finds' faces in anything)."""
    import torch
    from face_detection.detection.sfd.net_s3fd import s3fd
    from models import Wav2Lip

    root = tmp_path_factory.mktemp("models")
    torch.manual_seed(0)
    checkpoint = root / "wav2lip.pth"
    detector = root / "s3fd.pth"
    torch.save({"state_dict": Wav2Lip().state_dict(), "global_step": 1}, checkpoint)
    torch.save(s3fd().state_dict(), detector)
    return {"checkpoint": str(checkpoint), "s3fd": str(detector)}
