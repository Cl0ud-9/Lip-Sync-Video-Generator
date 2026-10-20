"""Tests for Elevenlab.py with the HTTP layer mocked (no network, no API key, no credits spent)."""

import os

import pytest
import requests

import Elevenlab as E


class FakeResponse:
    def __init__(self, status=200, content=b"MP3DATA", text="error body"):
        self.status_code, self.content, self.text = status, content, text


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(E.time, "sleep", lambda s: None)


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "script.txt"
    path.write_bytes(b"\xef\xbb\xbfhello world")  # UTF-8 BOM, as Windows Notepad writes it
    return path


def run(monkeypatch, script, out, responses):
    queue, sent = list(responses), []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(E.requests, "post", fake_post)
    return E.generate_audio_with_elevenlabs(str(script), str(out), "voice123", "secret"), sent


def test_success_writes_audio_and_strips_bom(monkeypatch, script, tmp_path):
    out = tmp_path / "o.mp3"
    ok, sent = run(monkeypatch, script, out, [FakeResponse()])
    assert ok and out.read_bytes() == b"MP3DATA"
    assert sent[0]["json"]["text"] == "hello world"
    assert sent[0]["url"].endswith("/voice123") and sent[0]["headers"]["xi-api-key"] == "secret"
    assert sent[0]["timeout"] == E.REQUEST_TIMEOUT
    assert not os.path.exists(str(out) + ".part")


@pytest.mark.parametrize("responses, calls, ok", [
    ([FakeResponse(500), FakeResponse(503), FakeResponse()], 3, True),         # transient server errors are retried
    ([FakeResponse(500)] * 3, 3, False),                                        # ...but only so many times
    ([requests.Timeout(), FakeResponse()], 2, True),
    ([requests.ConnectionError("down")] * 3, 3, False),
    ([FakeResponse(401)], 1, False),                                            # bad key: retrying can't help
    ([FakeResponse(429)], 1, False),                                            # quota: retrying burns nothing but time
    ([FakeResponse(404)], 1, False),                                            # bad voice id etc.
    ([FakeResponse(200, content=b"")], 1, False),                               # empty body is not "success"
])
def test_retry_policy(monkeypatch, script, tmp_path, responses, calls, ok):
    result, sent = run(monkeypatch, script, tmp_path / "o.mp3", responses)
    assert result is ok and len(sent) == calls


def test_failed_request_leaves_no_output_file(monkeypatch, script, tmp_path):
    out = tmp_path / "o.mp3"
    run(monkeypatch, script, out, [FakeResponse(401)])
    assert not out.exists()


def test_output_path_without_directory(monkeypatch, script, tmp_path):
    monkeypatch.chdir(tmp_path)
    ok, _ = run(monkeypatch, script, "bare.mp3", [FakeResponse()])
    assert ok and (tmp_path / "bare.mp3").exists()


def test_creates_missing_output_directories(monkeypatch, script, tmp_path):
    out = tmp_path / "a" / "b" / "o.mp3"
    ok, _ = run(monkeypatch, script, out, [FakeResponse()])
    assert ok and out.exists()


def test_bad_scripts(monkeypatch, tmp_path):
    out = tmp_path / "o.mp3"
    never = lambda *a, **k: pytest.fail("API must not be called")
    monkeypatch.setattr(E.requests, "post", never)

    empty = tmp_path / "empty.txt"; empty.write_text("  \n")
    binary = tmp_path / "binary.txt"; binary.write_bytes(b"\xff\xfe\x00bad\x80")
    for path in (empty, binary, tmp_path / "missing.txt"):
        assert E.generate_audio_with_elevenlabs(str(path), str(out), "v", "k") is False


def test_convert_missing_input(tmp_path):
    assert E.convert_mp3_to_wav(str(tmp_path / "nope.mp3"), str(tmp_path / "o.wav")) is False


def test_convert_without_ffmpeg(monkeypatch, tmp_path):
    src = tmp_path / "a.mp3"; src.write_bytes(b"x")
    monkeypatch.setattr(E.shutil, "which", lambda name: None)
    assert E.convert_mp3_to_wav(str(src), str(tmp_path / "o.wav")) is False


@pytest.mark.usefixtures("ffmpeg_tools")
def test_convert_real_mp3_to_wav(tmp_path):
    import subprocess
    mp3 = tmp_path / "tone.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(mp3)],
                   check=True)
    wav = tmp_path / "nested" / "tone.wav"
    assert E.convert_mp3_to_wav(str(mp3), str(wav)) is True
    assert wav.stat().st_size > 1000
