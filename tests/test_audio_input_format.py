"""ASR audio input format: only WAV/MP3/FLAC reach llama.cpp.

The audio test tab records in the browser and converts to WAV. If that
conversion fails (e.g. Firefox handing back WebM/Opus it cannot decode), the
raw clip must be rejected with a clear 400 instead of being forwarded
mislabeled as WAV — which used to surface as the opaque server error
"Failed to load image or audio file".
"""
import pytest
from fastapi import HTTPException

from routes.inference import _audio_data_uri_format


@pytest.mark.parametrize("data_uri,expected", [
    ("data:audio/wav;base64,AAAA", "wav"),
    ("data:audio/x-wav;base64,AAAA", "wav"),
    ("data:audio/wave;base64,AAAA", "wav"),
    ("data:audio/mpeg;base64,AAAA", "mp3"),
    ("data:audio/mp3;base64,AAAA", "mp3"),
    ("data:audio/flac;base64,AAAA", "flac"),
])
def test_supported_formats(data_uri, expected):
    assert _audio_data_uri_format(data_uri) == expected


@pytest.mark.parametrize("data_uri", [
    "data:audio/webm;codecs=opus;base64,AAAA",
    "data:audio/ogg;codecs=opus;base64,AAAA",
    "data:audio/mp4;base64,AAAA",   # AAC container miniaudio can't read
    "data:nonsense",
    "",
])
def test_unsupported_containers_are_rejected(data_uri):
    with pytest.raises(HTTPException) as exc:
        _audio_data_uri_format(data_uri)
    assert exc.value.status_code == 400
    assert "WAV" in exc.value.detail
