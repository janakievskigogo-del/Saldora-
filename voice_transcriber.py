"""Groq audio transcription for FakturaBot.mk voice prompts.

The MVP accepts a short voice-note file, sends it to the configured Groq Whisper
model for transcription, and then hands the *text* to the deterministic invoice
parser. Keeping transcription separate means a future UI can replace it with any
speech-to-text provider without touching invoice extraction.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from config import GROQ_BASE_URL, ConfigError, Settings
from utils import strip_code_fences

_MAX_AUDIO_BYTES = 18 * 1024 * 1024  # stay below the inline request limit

_SUPPORTED_AUDIO_TYPES = {
    ".flac": "audio/flac",
    ".mp3": "audio/mp3",
    ".mp4": "audio/mp4",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


class VoiceTranscriptionError(RuntimeError):
    """Raised when a voice prompt cannot be transcribed."""


def _audio_mime_type(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix in _SUPPORTED_AUDIO_TYPES:
        return _SUPPORTED_AUDIO_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("audio/"):
        return guessed
    raise VoiceTranscriptionError(
        f"Unsupported audio format {suffix or '(none)'}. "
        "Use flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm."
    )


def transcribe_audio(audio_path: str | Path, settings: Settings | None = None) -> str:
    """Transcribe a short voice note containing an invoice request.

    The returned transcript is intentionally not parsed here; callers should pass
    it to ``invoice_parser.parse_invoice_prompt``.
    """

    settings = settings or Settings.from_env()
    path = Path(audio_path).expanduser()
    if not path.exists():
        raise VoiceTranscriptionError(f"Voice prompt file not found: {path}")
    if not path.is_file():
        raise VoiceTranscriptionError(f"Voice prompt path is not a file: {path}")
    if path.stat().st_size == 0:
        raise VoiceTranscriptionError(f"Voice prompt file is empty: {path}")
    if path.stat().st_size > _MAX_AUDIO_BYTES:
        raise VoiceTranscriptionError(
            f"Voice prompt is larger than {_MAX_AUDIO_BYTES // (1024 * 1024)} MB. "
            "Use a shorter voice note."
        )

    try:
        api_key = settings.require_groq_api_key()
    except ConfigError as exc:
        raise VoiceTranscriptionError(str(exc)) from exc

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise VoiceTranscriptionError(
            "The openai package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    mime_type = _audio_mime_type(path)
    model = settings.groq_transcription_model

    prompt = (
        "Transcribe the following short business voice note faithfully. "
        "It is an invoice creation request in Macedonian, Albanian, or English. "
        "Preserve names, company names, tax IDs, amounts, percentages, dates, "
        "email addresses, and phone numbers exactly as spoken. Do not summarize, "
        "translate, explain, or add anything. Return only the transcript."
    )

    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        with path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=(path.name, audio_file, mime_type),
                prompt=prompt,
                temperature=0.0,
                response_format="json",
            )
        transcript = getattr(response, "text", None)
        if not isinstance(transcript, str):
            transcript = str(response) if response is not None else ""
    except Exception as exc:
        raise VoiceTranscriptionError(
            "Groq could not transcribe the voice note. Check the API key, quota, "
            f"model name ({model}), and network connection. Error: {exc}"
        ) from exc

    transcript = strip_code_fences(transcript).strip()
    if not transcript:
        raise VoiceTranscriptionError(
            "Groq returned an empty transcript. Check the audio file and try again."
        )
    return transcript
