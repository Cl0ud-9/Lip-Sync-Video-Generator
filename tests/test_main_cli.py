"""Tests for main.py: argument handling, input discovery, pre-flight checks and temp cleanup."""

import argparse
import os
import shutil

import pytest

import main


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point main.py's input/output/Wav2Lip folders at an empty temp tree."""
    for name in ("input", "output", "Wav2Lip"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(main, "INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(main, "WAV2LIP_DIR", str(tmp_path / "Wav2Lip"))
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    return tmp_path


def make_args(workspace, **overrides):
    values = dict(face=None, audio=None, script=str(workspace / "input" / "script.txt"), keep_temp=False,
                  checkpoint=str(workspace / "wav2lip.pth"))
    values.update(overrides)
    return argparse.Namespace(**values)


# --- positive_int -------------------------------------------------------------------------------------------------

def test_positive_int_accepts_valid():
    assert main.positive_int("4") == 4


@pytest.mark.parametrize("bad", ["0", "-2", "abc", "1.5", ""])
def test_positive_int_rejects_invalid(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        main.positive_int(bad)


def test_defaults_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert os.path.isabs(main.DEFAULT_OUTPUT_VIDEO) and os.path.isabs(main.DEFAULT_CHECKPOINT)
    assert os.path.isdir(main.WAV2LIP_DIR)


# --- get_input_files ------------------------------------------------------------------------------------------------

def test_explicit_missing_face(workspace, capsys):
    assert main.get_input_files(make_args(workspace, face=str(workspace / "nope.jpg"))) == (None, None)
    assert "does not exist" in capsys.readouterr().out


def test_no_face_in_input(workspace, capsys):
    assert main.get_input_files(make_args(workspace)) == (None, None)
    assert "No face image or video" in capsys.readouterr().out


def test_face_discovery_is_case_insensitive_sorted_and_ignores_other_files(workspace, capsys):
    (workspace / "input" / "b.MP4").write_bytes(b"x")
    (workspace / "input" / "a.JPG").write_bytes(b"x")
    (workspace / "input" / "notes.txt").write_text("x")
    (workspace / "input" / "audio.wav").write_bytes(b"x")
    face, audio = main.get_input_files(make_args(workspace))
    assert os.path.basename(face) == "a.JPG"
    assert os.path.basename(audio) == "audio.wav"
    assert "Multiple face files" in capsys.readouterr().out


def test_explicit_audio_skips_tts(workspace):
    face, audio = workspace / "f.png", workspace / "a.mp3"
    face.write_bytes(b"x"); audio.write_bytes(b"x")
    assert main.get_input_files(make_args(workspace, face=str(face), audio=str(audio))) == (str(face), str(audio))


def test_explicit_missing_audio(workspace, capsys):
    face = workspace / "f.png"; face.write_bytes(b"x")
    assert main.get_input_files(make_args(workspace, face=str(face), audio=str(workspace / "nope.wav"))) == (None, None)
    assert "--audio path does not exist" in capsys.readouterr().out


def test_script_without_api_key_fails_early_and_clearly(workspace, capsys, monkeypatch):
    (workspace / "input" / "a.jpg").write_bytes(b"x")
    (workspace / "input" / "script.txt").write_text("hello")
    spawned = []
    monkeypatch.setattr(main.subprocess, "run", lambda *a, **k: spawned.append(a))
    assert main.get_input_files(make_args(workspace)) == (None, None)
    assert "ELEVENLABS_API_KEY" in capsys.readouterr().out
    assert spawned == []  # never launched the TTS subprocess


def test_script_with_key_runs_tts_with_absolute_paths(workspace, monkeypatch):
    (workspace / "input" / "a.jpg").write_bytes(b"x")
    (workspace / "input" / "script.txt").write_text("hello")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    seen = {}

    def fake_run(cmd, check):
        seen["cmd"] = cmd
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    face, audio = main.get_input_files(make_args(workspace))
    assert audio == str(workspace / "output" / "audio.wav")
    assert os.path.isabs(seen["cmd"][1]) and seen["cmd"][1].endswith("Elevenlab.py")
    assert str(workspace / "output" / "audio.mp3") in seen["cmd"]


def test_tts_failure_is_reported(workspace, monkeypatch, capsys):
    (workspace / "input" / "a.jpg").write_bytes(b"x")
    (workspace / "input" / "script.txt").write_text("hello")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setattr(main.subprocess, "run", lambda cmd, check: argparse.Namespace(returncode=1))
    assert main.get_input_files(make_args(workspace)) == (None, None)
    assert "Text-to-Speech generation failed" in capsys.readouterr().out


def test_no_audio_and_no_script(workspace, capsys):
    (workspace / "input" / "a.jpg").write_bytes(b"x")
    assert main.get_input_files(make_args(workspace)) == (None, None)
    assert "No input found" in capsys.readouterr().out


# --- pre-flight + cleanup ---------------------------------------------------------------------------------------------

def test_preflight_requires_ffmpeg(workspace, monkeypatch):
    monkeypatch.setattr(main.shutil, "which", lambda name: None)
    assert main.check_pre_flight(make_args(workspace)) is False


def test_preflight_requires_checkpoint(workspace, monkeypatch):
    monkeypatch.setattr(main.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert main.check_pre_flight(make_args(workspace)) is False  # checkpoint file does not exist


def test_preflight_passes_when_everything_is_present(workspace, monkeypatch):
    monkeypatch.setattr(main.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    checkpoint = workspace / "wav2lip.pth"; checkpoint.write_bytes(b"x")
    detector = workspace / "s3fd.pth"; detector.write_bytes(b"x")
    monkeypatch.setattr(main, "DEFAULT_FACE_DETECTOR", str(detector))
    assert main.check_pre_flight(make_args(workspace, checkpoint=str(checkpoint))) is True


def test_cleanup_removes_temp_files_but_keeps_gitkeep(workspace):
    temp = workspace / "Wav2Lip" / "temp"
    temp.mkdir()
    (temp / ".gitkeep").write_text("")
    (temp / "result.avi").write_bytes(b"x")
    (temp / "temp.wav").write_bytes(b"x")
    main.cleanup_temp_files(keep_temp=False)
    assert sorted(os.listdir(temp)) == [".gitkeep"]


def test_cleanup_respects_keep_temp(workspace):
    temp = workspace / "Wav2Lip" / "temp"
    temp.mkdir()
    (temp / "result.avi").write_bytes(b"x")
    main.cleanup_temp_files(keep_temp=True)
    assert os.listdir(temp) == ["result.avi"]


def test_cleanup_with_missing_temp_dir_is_a_noop(workspace):
    main.cleanup_temp_files(keep_temp=False)  # must not raise
