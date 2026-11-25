"""HAVN-71: five high-risk checkpoints stay in the registry but cannot be served.

``enabled: false`` is the coordinator's off switch. The rows remain so legal
can turn a model back on by setting the flag to true or deleting it. Until
then the model is absent from catalogs, job submission, auto routing, downloads,
and node leasing.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

import app as app_module
from common.prompt_enhancers import enhance_prompt_for_positions

REGISTRY_PATH = ROOT / "server" / "manifests" / "registry.json"
WALLET = "0x1111111111111111111111111111111111111111"

DISABLED = {
    "disneyPixarCartoon_v10": "Disney/Pixar trademark",
    "uberRealisticPornMerge_v23Final": "No provenance found",
    "unstablePornhwa_beta": "No provenance found",
    "kizukiAnimeHentai_animeHentaiV4": "Civitai 22364",
    "copaxTimeless_xplus2BNSFW1": "Civitai 118111",
}


def _names(payload: dict) -> set[str]:
    models = payload.get("models") or []
    found = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model", "model_key", "id"):
            value = item.get(key)
            if value:
                found.add(str(value).lower())
    return found


class RegistryFlagTests(unittest.TestCase):
    def test_five_rows_stay_in_the_file_and_record_havn_71(self) -> None:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        by_name = {item["name"]: item for item in payload["models"]}
        self.assertEqual(len(by_name), 30)
        for name, marker in DISABLED.items():
            entry = by_name[name]
            self.assertIs(entry["enabled"], False)
            self.assertEqual(entry["license_status"], "blocked_pending_clearance")
            self.assertIn("HAVN-71", entry["disabled_reason"])
            self.assertIn(marker, entry["disabled_reason"])
        enabled = [name for name, entry in by_name.items() if entry.get("enabled") is not False]
        self.assertEqual(len(enabled), 25)
        self.assertIn("juggernautXL_ragnarokBy", enabled)
        self.assertTrue(set(DISABLED).isdisjoint(enabled))

    def test_only_json_false_disables_an_entry(self) -> None:
        original_models = copy.deepcopy(app_module.MANIFEST_MODELS)
        original_weights = copy.deepcopy(app_module.MODEL_WEIGHTS)
        original_stats = copy.deepcopy(app_module.MODEL_STATS)
        original_file = app_module.MANIFEST_FILE
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "registry.json"
                path.write_text(
                    json.dumps(
                        {
                            "models": [
                                {"name": "kept-omitted", "pipeline": "sdxl", "task_type": "IMAGE_GEN"},
                                {"name": "kept-true", "pipeline": "sdxl", "enabled": True},
                                {"name": "held-back", "pipeline": "sd15", "enabled": False, "disabled_reason": "HAVN-71"},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                app_module.MANIFEST_FILE = path
                app_module.load_manifest()
            self.assertIn("kept-omitted", app_module.MANIFEST_MODELS)
            self.assertIn("kept-true", app_module.MANIFEST_MODELS)
            self.assertNotIn("held-back", app_module.MANIFEST_MODELS)
            self.assertNotIn("held-back", app_module.MODEL_WEIGHTS)
        finally:
            app_module.MANIFEST_FILE = original_file
            app_module.MANIFEST_MODELS.clear()
            app_module.MANIFEST_MODELS.update(original_models)
            app_module.MODEL_WEIGHTS.clear()
            app_module.MODEL_WEIGHTS.update(original_weights)
            app_module.MODEL_STATS.clear()
            app_module.MODEL_STATS.update(original_stats)


class ServingGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()
        self._models = copy.deepcopy(app_module.MANIFEST_MODELS)
        self._weights = copy.deepcopy(app_module.MODEL_WEIGHTS)
        self._stats = copy.deepcopy(app_module.MODEL_STATS)
        self._nodes = copy.deepcopy(app_module.NODES)
        self._file = app_module.MANIFEST_FILE
        self._db_path = app_module.DB_PATH
        self._db_conn = app_module.DB_CONN
        self._credits = app_module.credits.CREDITS_ENABLED
        self._tmpdir = tempfile.TemporaryDirectory()
        app_module.MANIFEST_FILE = REGISTRY_PATH
        app_module.load_manifest()
        app_module.DB_PATH = Path(self._tmpdir.name) / "ledger.db"
        app_module.DB_CONN = None
        app_module.credits.CREDITS_ENABLED = False
        app_module.init_db()

    def tearDown(self) -> None:
        app_module.MANIFEST_FILE = self._file
        app_module.MANIFEST_MODELS.clear()
        app_module.MANIFEST_MODELS.update(self._models)
        app_module.MODEL_WEIGHTS.clear()
        app_module.MODEL_WEIGHTS.update(self._weights)
        app_module.MODEL_STATS.clear()
        app_module.MODEL_STATS.update(self._stats)
        app_module.NODES.clear()
        app_module.NODES.update(self._nodes)
        app_module.DB_PATH = self._db_path
        app_module.DB_CONN = self._db_conn
        app_module.credits.CREDITS_ENABLED = self._credits
        self._tmpdir.cleanup()

    def test_live_catalog_omits_the_five_and_keeps_the_studio_default(self) -> None:
        loaded = {str(meta.get("name")) for meta in app_module.MANIFEST_MODELS.values()}
        self.assertTrue(set(DISABLED).isdisjoint(loaded))
        self.assertIn("juggernautXL_ragnarokBy", loaded)
        self.assertEqual(len(loaded), 25)
        self.assertNotIn("unstablepornhwa_beta", app_module.MODEL_WEIGHTS)
        for name in DISABLED:
            self.assertIsNone(app_module.get_model_config(name))
            self.assertIsNone(app_module.get_model_config(name.lower()))
            self.assertIsNone(app_module._resolve_hosted_model(name))

    def test_public_and_private_catalogs_hide_them(self) -> None:
        listed = _names(self.client.get("/models/list").get_json())
        feed = _names(self.client.get("/feed").get_json())
        nodes = _names({"models": self.client.get("/nodes").get_json()["models_catalog"]})
        with patch.object(app_module, "_require_studio_user", return_value=None):
            capabilities = _names(self.client.get("/v1/capabilities").get_json())
            account_capabilities = _names(self.client.get("/v2/capabilities").get_json())
        disabled = {name.lower() for name in DISABLED}
        for catalog in (listed, feed, nodes, capabilities, account_capabilities):
            self.assertIn("juggernautxl_ragnarokby", catalog)
            self.assertTrue(disabled.isdisjoint(catalog))

    def test_submit_paths_reject_an_explicit_request(self) -> None:
        prompt = "a ceramic bowl on a wooden table"
        for name in DISABLED:
            job = self.client.post(
                "/submit-job",
                json={"wallet": WALLET, "model": name, "prompt": prompt},
            )
            self.assertEqual(job.status_code, 400, name)
            self.assertEqual(job.get_json()["error"], "unknown model")
            video = self.client.post(
                "/generate-video",
                json={"wallet": WALLET, "model": name, "prompt": prompt},
            )
            self.assertEqual(video.status_code, 400, name)
            self.assertEqual(video.get_json()["error"], "unknown model")
            swap = self.client.post(
                "/submit-faceswap-job",
                json={"wallet": WALLET, "model": name, "prompt": prompt},
            )
            self.assertEqual(swap.status_code, 400, name)
            self.assertEqual(swap.get_json()["error"], "unknown model")

    def test_auto_routing_ignores_a_node_that_still_advertises_them(self) -> None:
        app_module.NODES["node-havn71"] = {
            "role": "creator",
            "last_seen_unix": app_module.unix_now(),
            "models": ["juggernautXL_ragnarokBy", *DISABLED],
            "pipelines": ["sdxl", "sd15"],
            "supports": ["image"],
        }
        seen: dict[str, list[str]] = {}

        def capture(names, weights, k):
            seen["names"] = list(names)
            return [names[0]]

        with patch.object(app_module.random, "choices", capture):
            response = self.client.post(
                "/submit-job",
                json={"wallet": WALLET, "model": "auto", "prompt": "a ceramic bowl on a wooden table"},
            )
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(seen["names"], ["juggernautXL_ragnarokBy"])
        self.assertTrue(set(DISABLED).isdisjoint(seen["names"]))

    def test_download_and_lease_do_not_serve_a_disabled_checkpoint(self) -> None:
        app_module.NODES["node-havn71"] = {
            "role": "creator",
            "last_seen_unix": app_module.unix_now(),
            "models": ["juggernautXL_ragnarokBy", *DISABLED],
            "pipelines": ["sdxl", "sd15"],
            "supports": ["image"],
        }
        blocked = app_module.job_helpers.enqueue_job(
            WALLET, "disneyPixarCartoon_v10", "IMAGE_GEN", "{}", 8.0
        )
        self.assertIsNone(app_module.job_helpers.fetch_next_job_for_node("node-havn71"))
        allowed = app_module.job_helpers.enqueue_job(
            WALLET, "juggernautXL_ragnarokBy", "IMAGE_GEN", "{}", 8.0
        )
        leased = app_module.job_helpers.fetch_next_job_for_node("node-havn71")
        self.assertIsNotNone(leased)
        self.assertEqual(leased["id"], allowed)
        self.assertNotEqual(leased["id"], blocked)

        with patch.object(app_module, "NODE_API_TOKEN", "havn71-token"):
            response = self.client.get(
                "/models/download/disneyPixarCartoon_v10",
                headers={"Authorization": "Bearer havn71-token"},
            )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"], "model artifact not available")


class RerouteTests(unittest.TestCase):
    def test_astra_preference_uses_the_studio_default_instead_of_copax(self) -> None:
        lowered = [name.lower() for name in app_module.astra_gen.PREFERRED_MODELS]
        self.assertNotIn("copaxtimeless_xplus2bnsfw1", lowered)
        self.assertEqual(lowered[-1], "juggernautxl_ragnarokby")
        self.assertEqual(lowered[:2], ["amixillustrious_amix", "perfectdeliberate_v60"])

    def test_style_router_no_longer_names_the_disabled_checkpoints(self) -> None:
        constants = " ".join(
            item
            for item in enhance_prompt_for_positions.__code__.co_consts
            if isinstance(item, str)
        )
        for name in DISABLED:
            self.assertNotIn(name, constants)
        self.assertIn("juggernautXL_ragnarokBy", constants)
        _enhanced, model, _lora, _negative = enhance_prompt_for_positions("anime hentai from behind")
        self.assertEqual(model, "lazymixRealAmateur_v40")
        self.assertNotIn(model, DISABLED)

    def test_node_manifest_parser_skips_a_disabled_entry(self) -> None:
        # Load the node modules by file so this test does not bind the ``client``
        # package name. The worker tests import ``client.py`` as ``client``.
        import importlib.util

        def load(module_name: str, path: Path):
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            # dataclass looks the class module up in sys.modules during class creation.
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        registry = load("havn71_node_registry", ROOT / "client" / "registry.py")
        sources = load("havn71_model_sources", ROOT / "client" / "model_sources.py")
        manifest = registry.ModelRegistry._parse_manifest(
            {
                "models": [
                    {"name": "kept", "pipeline": "sdxl", "type": "checkpoint"},
                    {"name": "disneyPixarCartoon_v10", "pipeline": "sd15", "enabled": False},
                ]
            }
        )
        self.assertEqual([entry.name for entry in manifest.models], ["kept"])
        plan = sources.requirements_from_manifest(
            [
                {"name": "kept", "pipeline": "sdxl", "source": {"kind": "coordinator"}},
                {"name": "copaxTimeless_xplus2BNSFW1", "pipeline": "sdxl", "enabled": False},
            ],
            Path("/tmp/havn71-plan"),
        )
        self.assertEqual([item.name for item in plan], ["kept"])


if __name__ == "__main__":
    unittest.main()
