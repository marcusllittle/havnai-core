"""ACE-Step provider contract tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from engines.ace_step import AceStepError, AceStepProvider


class FakeResponse:
    def __init__(self, payload: Any = None, *, body: bytes = b"", content_type: str = "application/json") -> None:
        self.payload = payload
        self.body = body
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload

    def iter_content(self, chunk_size: int = 0):
        del chunk_size
        yield self.body


class FakeSession:
    def __init__(self, posts: list[FakeResponse], gets: list[FakeResponse]) -> None:
        self.posts = posts
        self.gets = gets
        self.post_calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.post_calls.append((url, kwargs))
        return self.posts.pop(0)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        del url, kwargs
        return self.gets.pop(0)


def wrapped(data: Any, *, error: str | None = None) -> FakeResponse:
    return FakeResponse({"data": data, "code": 500 if error else 200, "error": error})


class AceStepProviderTests(unittest.TestCase):
    def test_probe_reports_loaded_models(self) -> None:
        session = FakeSession(
            posts=[],
            gets=[
                wrapped({"status": "ok", "service": "ACE-Step API", "version": "1.0"}),
                wrapped({
                    "models": [{"name": "acestep-v15-turbo", "is_default": True}],
                    "default_model": "acestep-v15-turbo",
                }),
            ],
        )
        probe = AceStepProvider(session=session).probe()
        self.assertEqual(probe["models"], ["acestep-v15-turbo"])
        self.assertEqual(probe["default_model"], "acestep-v15-turbo")

    def test_probe_reports_lazy_configured_model_from_health(self) -> None:
        session = FakeSession(
            posts=[],
            gets=[
                wrapped({
                    "status": "ok",
                    "service": "ACE-Step API",
                    "loaded_model": "acestep-v15-turbo",
                    "models_initialized": False,
                }),
                wrapped({"models": [], "default_model": "acestep-v15-turbo"}),
            ],
        )
        probe = AceStepProvider(session=session).probe()
        self.assertEqual(probe["models"], ["acestep-v15-turbo"])
        self.assertEqual(probe["default_model"], "acestep-v15-turbo")

    def test_generates_and_downloads_audio_with_metadata(self) -> None:
        result = [{
            "file": "/v1/audio?path=result.mp3",
            "prompt": "night drive",
            "lyrics": "",
            "metas": {"duration": 30, "bpm": 112, "keyscale": "D Minor"},
            "seed_value": "42",
            "dit_model": "acestep-v15-turbo",
        }]
        session = FakeSession(
            posts=[
                wrapped({"task_id": "ace-1", "status": "queued"}),
                wrapped([{"task_id": "ace-1", "status": 0, "result": ""}]),
                wrapped([{"task_id": "ace-1", "status": 1, "result": json.dumps(result)}]),
            ],
            gets=[FakeResponse(body=b"real-audio", content_type="audio/mpeg")],
        )
        progress: list[tuple[float, str]] = []
        provider = AceStepProvider(session=session, poll_interval=0.001)
        with tempfile.TemporaryDirectory() as directory:
            generated = provider.generate(
                {
                    "prompt": "night drive",
                    "style": "synthwave",
                    "lyrics": "ignored",
                    "instrumental": True,
                    "duration": 30,
                    "bpm": 112,
                    "key": "D Minor",
                    "seed": 42,
                },
                Path(directory),
                progress=lambda value, stage: progress.append((value, stage)),
            )
            self.assertEqual(generated.path.read_bytes(), b"real-audio")
        release_payload = session.post_calls[0][1]["json"]
        self.assertEqual(release_payload["task_type"], "text2music")
        self.assertEqual(release_payload["lyrics"], "")
        self.assertIn("Style: synthwave", release_payload["prompt"])
        self.assertFalse(release_payload["thinking"])
        self.assertFalse(release_payload["use_random_seed"])
        self.assertEqual(generated.metadata["bpm"], 112)
        self.assertEqual(progress[-1][1], "finishing")

    def test_service_error_is_not_treated_as_success(self) -> None:
        provider = AceStepProvider(
            session=FakeSession(posts=[wrapped(None, error="queue full")], gets=[])
        )
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(AceStepError, "queue full"):
            provider.generate({"prompt": "test", "duration": 30}, Path(directory))

    def test_metadata_sentinels_fall_back_to_requested_values(self) -> None:
        self.assertEqual(AceStepProvider._metadata_value("N/A", 108), 108)
        self.assertEqual(AceStepProvider._metadata_value("none", "C Minor"), "C Minor")
        self.assertIsNone(AceStepProvider._metadata_value("N/A"))


if __name__ == "__main__":
    unittest.main()
