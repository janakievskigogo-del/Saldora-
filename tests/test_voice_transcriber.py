"""Tests for Groq Whisper voice transcription (fully offline)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from config import Settings
from voice_transcriber import VoiceTranscriptionError, transcribe_audio


def _settings() -> Settings:
    return Settings(groq_api_key="gsk_test-key")


def test_transcribe_audio_uses_groq_whisper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"RIFF-fake-audio")
    captured: dict[str, object] = {}

    class FakeTranscriptions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(text="  Create invoice for Acme  ")

    class FakeClient:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    transcript = transcribe_audio(audio, settings=_settings())

    assert transcript == "Create invoice for Acme"
    assert captured["api_key"] == "gsk_test-key"
    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["model"] == "whisper-large-v3"
    assert captured["response_format"] == "json"
    assert captured["temperature"] == 0.0
    file_parts = captured["file"]
    assert isinstance(file_parts, tuple)
    assert file_parts[0] == "note.wav"
    assert file_parts[2] == "audio/wav"


def test_transcribe_audio_requires_groq_key(tmp_path: Path) -> None:
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"RIFF-fake-audio")

    with pytest.raises(VoiceTranscriptionError, match="GROQ_API_KEY"):
        transcribe_audio(audio, settings=Settings())


def test_transcribe_audio_rejects_unsupported_format(tmp_path: Path) -> None:
    audio = tmp_path / "note.txt"
    audio.write_bytes(b"not-audio")

    with pytest.raises(VoiceTranscriptionError, match="Unsupported audio format"):
        transcribe_audio(audio, settings=_settings())
