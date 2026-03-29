from __future__ import annotations

import json

from fastapi.testclient import TestClient

from pronunciation_backend import frontend


def test_create_frontend_app_requires_http_scheme() -> None:
    try:
        frontend.create_frontend_app(backend_base_url="example.com:8000")
    except ValueError as exc:
        assert "http://" in str(exc)
    else:
        raise AssertionError("Expected create_frontend_app to reject backend URLs without a scheme")


def test_frontend_index_and_config() -> None:
    client = TestClient(frontend.create_frontend_app(backend_base_url="http://backend.internal:8000"))

    index_response = client.get("/")
    assert index_response.status_code == 200
    assert "Pronunciation Scorer Debug UI" in index_response.text

    config_response = client.get("/api/config")
    assert config_response.status_code == 200
    assert config_response.json()["backend_base_url"] == "http://backend.internal:8000"


def test_frontend_words_proxy(monkeypatch) -> None:
    def fake_proxy(method: str, url: str, **kwargs):
        del kwargs
        assert method == "GET"
        assert url == "http://backend.internal:8000/v1/words"
        return 200, json.dumps({"words": ["thought", "through"]}).encode("utf-8"), {"Content-Type": "application/json"}

    monkeypatch.setattr(frontend, "_proxy_request", fake_proxy)
    client = TestClient(frontend.create_frontend_app(backend_base_url="http://backend.internal:8000"))

    response = client.get("/api/words")
    assert response.status_code == 200
    assert response.json()["words"] == ["thought", "through"]


def test_frontend_score_proxy_forwards_audio(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_proxy(method: str, url: str, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["data"] = kwargs["data"]
        payload = {
            "word": "thought",
            "accent_target": "en-US",
            "ipa": "theta",
            "overall_score": 80.0,
            "confidence": 0.9,
            "audio_quality": {
                "status": "ok",
                "snr_estimate": 20.0,
                "duration_ms": 600,
                "rms": 0.2,
                "clipping_ratio": 0.0,
                "silence_ratio": 0.1,
                "original_duration_ms": 600,
                "trim_start_ms": 0,
                "trim_end_ms": 600,
                "trim_applied": False,
            },
            "phonemes": [],
            "primary_issue": {"phoneme": "TH", "type": "accented", "message": "debug"},
            "reference": {"ipa": "theta", "audio_id": "thought_en_us_01", "asset_path": None},
            "model_info": {
                "runtime_backend": "scorer_v2",
                "model_version": "v2",
                "checkpoint_name": "fake.pt",
                "backbone_id": "facebook/hubert-base-ls960",
                "device": "cpu",
                "class_labels": ["wrong_or_missed", "accented", "correct"],
            },
        }
        return 200, json.dumps(payload).encode("utf-8"), {"Content-Type": "application/json"}

    monkeypatch.setattr(frontend, "_proxy_request", fake_proxy)
    client = TestClient(frontend.create_frontend_app(backend_base_url="http://backend.internal:8000"))

    response = client.post(
        "/api/score",
        data={"word": "thought", "speaker_id": "spk-1", "noTrim": "true"},
        files={"audio": ("sample.wav", b"RIFFdemo", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["word"] == "thought"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://backend.internal:8000/v1/pronunciation/score"
    assert "multipart/form-data" in str(captured["headers"])
    body = captured["data"]
    assert isinstance(body, bytes)
    assert b'thought' in body
    assert b'spk-1' in body
    assert b'noTrim' in body
    assert b'true' in body
    assert b'sample.wav' in body
    assert b'RIFFdemo' in body


def test_frontend_main_uses_env_backend_url(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app, host: str, port: int):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setenv(frontend.BACKEND_URL_ENV, "http://backend.internal:8000")
    monkeypatch.setattr(frontend.uvicorn, "run", fake_run)

    exit_code = frontend.main([])

    assert exit_code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 3000
