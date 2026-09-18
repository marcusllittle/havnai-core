"""ACE-Step provider contract tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from engines.ace_step import (
    TURBO_TASK_TYPES,
    AceStepCapabilityError,
    AceStepError,
    AceStepModel,
    AceStepModelMismatch,
    AceStepProvider,
)


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
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0].path.read_bytes(), b"real-audio")
            self.assertTrue(generated[0].is_primary)
        release_payload = session.post_calls[0][1]["json"]
        self.assertEqual(release_payload["task_type"], "text2music")
        self.assertEqual(release_payload["lyrics"], "")
        self.assertIn("Style: synthwave", release_payload["prompt"])
        self.assertFalse(release_payload["thinking"])
        self.assertFalse(release_payload["use_random_seed"])
        self.assertEqual(generated[0].metadata["bpm"], 112)
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




class AceStepTaskSurfaceTests(unittest.TestCase):
    """The provider must expose the whole ACE-Step task surface, not just text2music."""

    def test_cover_payload_carries_source_and_strength(self) -> None:
        payload = AceStepProvider._generation_payload({
            "task_type": "cover",
            "prompt": "make it a bossa nova",
            "duration": 45,
            "audio_cover_strength": 0.4,
            "cover_noise_strength": 0.2,
        })
        self.assertEqual(payload["task_type"], "cover")
        self.assertAlmostEqual(payload["audio_cover_strength"], 0.4)
        self.assertAlmostEqual(payload["cover_noise_strength"], 0.2)

    def test_repaint_payload_sets_explicit_mask(self) -> None:
        payload = AceStepProvider._generation_payload({
            "task_type": "repaint",
            "prompt": "fix the second verse",
            "duration": 90,
            "repainting_start": 30.0,
            "repainting_end": 45.0,
            "repaint_mode": "aggressive",
        })
        self.assertEqual(payload["chunk_mask_mode"], "explicit")
        self.assertEqual(payload["repainting_start"], 30.0)
        self.assertEqual(payload["repainting_end"], 45.0)
        self.assertEqual(payload["repaint_mode"], "aggressive")
        # repaint_strength only applies to balanced mode
        self.assertNotIn("repaint_strength", payload)

    def test_extract_requires_a_known_track_name(self) -> None:
        payload = AceStepProvider._generation_payload({
            "task_type": "extract",
            "duration": 60,
            "track_name": "Drums",
        })
        self.assertEqual(payload["track_name"], "drums")
        with self.assertRaises(AceStepCapabilityError):
            AceStepProvider._generation_payload({
                "task_type": "extract",
                "duration": 60,
                "track_name": "kazoo",
            })

    def test_complete_requires_track_classes(self) -> None:
        payload = AceStepProvider._generation_payload({
            "task_type": "complete",
            "duration": 60,
            "track_classes": ["bass", "Drums", "kazoo"],
        })
        self.assertEqual(payload["track_classes"], ["bass", "drums"])
        with self.assertRaises(AceStepCapabilityError):
            AceStepProvider._generation_payload({
                "task_type": "complete",
                "duration": 60,
                "track_classes": ["kazoo"],
            })

    def test_batch_size_is_clamped(self) -> None:
        payload = AceStepProvider._generation_payload({
            "task_type": "text2music",
            "prompt": "x",
            "duration": 30,
            "batch_size": 99,
        })
        self.assertEqual(payload["batch_size"], 8)

    def test_unknown_task_type_rejected(self) -> None:
        with self.assertRaises(AceStepCapabilityError):
            AceStepProvider._generation_payload({"task_type": "stemify", "duration": 30})

    def test_validate_rejects_source_tasks_without_audio(self) -> None:
        provider = AceStepProvider(session=FakeSession(posts=[], gets=[]))
        with self.assertRaisesRegex(AceStepCapabilityError, "source_audio_required"):
            provider.validate({"task_type": "repaint", "duration": 30})

    def test_validate_rejects_task_the_checkpoint_cannot_run(self) -> None:
        provider = AceStepProvider(session=FakeSession(posts=[], gets=[]))
        models = [AceStepModel(name="acestep-v15-turbo", supported_task_types=TURBO_TASK_TYPES)]
        with self.assertRaisesRegex(AceStepCapabilityError, "task_unsupported_by_model"):
            provider.validate(
                {
                    "task_type": "extract",
                    "duration": 30,
                    "src_audio_path": "/tmp/x.mp3",
                    "engine_model": "acestep-v15-turbo",
                },
                models=models,
            )

    def test_batch_results_are_written_to_distinct_files(self) -> None:
        items = [
            {"file": "/v1/audio?path=a.mp3", "seed_value": "1", "metas": {"duration": 30}},
            {"file": "/v1/audio?path=b.mp3", "seed_value": "2", "metas": {"duration": 30}},
        ]
        session = FakeSession(
            posts=[
                wrapped({"task_id": "ace-2"}),
                wrapped([{"task_id": "ace-2", "status": 1, "result": json.dumps(items)}]),
            ],
            gets=[
                FakeResponse(body=b"take-one", content_type="audio/mpeg"),
                FakeResponse(body=b"take-two", content_type="audio/mpeg"),
            ],
        )
        provider = AceStepProvider(session=session, poll_interval=0.001)
        with tempfile.TemporaryDirectory() as directory:
            results = provider.generate(
                {"prompt": "two takes", "duration": 30, "batch_size": 2},
                Path(directory),
            )
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].path.name, "music.mp3")
            self.assertEqual(results[1].path.name, "music-2.mp3")
            self.assertEqual(results[0].path.read_bytes(), b"take-one")
            self.assertEqual(results[1].path.read_bytes(), b"take-two")
            self.assertTrue(results[0].is_primary)
            self.assertFalse(results[1].is_primary)
            self.assertEqual(results[1].metadata["variation"], 2)




class AceStepFallbackGuardTests(unittest.TestCase):
    """ACE-Step answers an unknown/unloaded `model` with primary-model audio at
    HTTP 200. That must never reach rewards or publication as the real thing."""

    def test_result_from_a_different_checkpoint_is_rejected(self) -> None:
        item = {
            "file": "/v1/audio?path=a.mp3",
            "dit_model": "acestep-v15-turbo",
            "metas": {"duration": 30},
        }
        session = FakeSession(
            posts=[
                wrapped({"task_id": "ace-3"}),
                wrapped([{"task_id": "ace-3", "status": 1, "result": json.dumps([item])}]),
            ],
            gets=[FakeResponse(body=b"wrong-model", content_type="audio/mpeg")],
        )
        provider = AceStepProvider(session=session, poll_interval=0.001)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AceStepModelMismatch, "model_fallback"):
                provider.generate(
                    {"prompt": "x", "duration": 30, "engine_model": "acestep-v15-base"},
                    Path(directory),
                )

    def test_matching_checkpoint_passes(self) -> None:
        item = {
            "file": "/v1/audio?path=a.mp3",
            "dit_model": "acestep-v15-base",
            "metas": {"duration": 30},
        }
        session = FakeSession(
            posts=[
                wrapped({"task_id": "ace-4"}),
                wrapped([{"task_id": "ace-4", "status": 1, "result": json.dumps([item])}]),
            ],
            gets=[FakeResponse(body=b"right-model", content_type="audio/mpeg")],
        )
        provider = AceStepProvider(session=session, poll_interval=0.001)
        with tempfile.TemporaryDirectory() as directory:
            results = provider.generate(
                {"prompt": "x", "duration": 30, "engine_model": "acestep-v15-base"},
                Path(directory),
            )
            self.assertEqual(results[0].path.read_bytes(), b"right-model")

    def test_service_without_dit_model_reporting_is_tolerated(self) -> None:
        AceStepProvider._verify_engine_model(
            [{"file": "a.mp3"}], {"engine_model": "acestep-v15-base"}
        )

    def test_downloaded_but_unloaded_checkpoint_is_not_selectable(self) -> None:
        provider = AceStepProvider(session=FakeSession(posts=[], gets=[]))
        models = [
            AceStepModel(name="acestep-v15-turbo", is_default=True, is_loaded=True),
            # On disk (so the inventory lists it) but not loaded into any slot.
            AceStepModel(name="acestep-v15-base", is_loaded=False),
        ]
        with self.assertRaisesRegex(AceStepCapabilityError, "model_not_loaded"):
            provider.validate(
                {"task_type": "text2music", "duration": 30, "engine_model": "acestep-v15-base"},
                models=models,
            )




class AceStepOnDemandTests(unittest.TestCase):
    """With the service in on-demand mode an on-disk checkpoint is servable."""

    def setUp(self) -> None:
        os.environ.pop("HAVNAI_ACESTEP_ON_DEMAND", None)

    def tearDown(self) -> None:
        os.environ.pop("HAVNAI_ACESTEP_ON_DEMAND", None)

    def test_off_by_default(self) -> None:
        self.assertFalse(AceStepProvider(session=FakeSession(posts=[], gets=[])).on_demand)

    def test_unloaded_checkpoint_allowed_when_enabled(self) -> None:
        os.environ["HAVNAI_ACESTEP_ON_DEMAND"] = "1"
        provider = AceStepProvider(session=FakeSession(posts=[], gets=[]))
        self.assertTrue(provider.on_demand)
        models = [
            AceStepModel(name="acestep-v15-turbo", is_default=True, is_loaded=True),
            AceStepModel(name="acestep-v15-base", is_loaded=False),
        ]
        # No raise: the service will swap its primary slot on the request.
        provider.validate(
            {"task_type": "text2music", "duration": 30, "engine_model": "acestep-v15-base"},
            models=models,
        )

    def test_mismatch_guard_still_applies_in_on_demand_mode(self) -> None:
        os.environ["HAVNAI_ACESTEP_ON_DEMAND"] = "1"
        with self.assertRaises(AceStepModelMismatch):
            AceStepProvider._verify_engine_model(
                [{"file": "a.mp3", "dit_model": "acestep-v15-turbo"}],
                {"engine_model": "acestep-v15-base"},
            )


if __name__ == "__main__":
    unittest.main()
