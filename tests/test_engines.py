import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tempfile
from pathlib import Path

from engines import CosyvoiceEngine, DotsEngine, QwenEngine, VoxcpmEngine, present_engines


class EnginePresenceTest(unittest.TestCase):
    """zz-whitby 2026-11-21: polytts advertised `qwen` with no models/ dir. The
    host broker evicted flash_next to warm it, the warm answered 500 ("Qwen model
    not found") and the card was held for 900 s. An engine that cannot load must
    not be offered as a unit at all."""

    def test_qwen_with_no_model_dir_is_not_advertised(self):
        with tempfile.TemporaryDirectory() as d:
            present, missing = present_engines(
                {"qwen": QwenEngine(Path(d) / "models"), "voxcpm": VoxcpmEngine()})
        self.assertEqual(list(present), ["voxcpm"])
        self.assertIn("Qwen model not found", missing["qwen"])

    def test_qwen_with_its_model_dir_is_advertised(self):
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {"POLYTTS_MODEL": "1.7B"}):
            (Path(d) / "Qwen3-TTS-12Hz-1.7B-Base").mkdir()
            present, missing = present_engines({"qwen": QwenEngine(Path(d))})
        self.assertEqual(list(present), ["qwen"])
        self.assertEqual(missing, {})

    # zz-whitby / zz-leaky 2026-11-23: cosyvoice and dots were declared units on
    # hosts holding neither model. Each check below is offline: an empty
    # HF_HUB_CACHE stands for "never downloaded".

    def _hf_env(self, cache):
        return patch.dict("os.environ", {"HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1"})

    def _snapshot(self, cache, repo, files):
        org, name = repo.split("/")
        root = Path(cache) / f"models--{org}--{name}"
        snap = root / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (root / "refs").mkdir()
        (root / "refs" / "main").write_text("abc123")
        for f in files:
            (snap / f).write_text("x")

    def _present(self, engines_by_name, cache):
        import huggingface_hub.constants as c
        with self._hf_env(cache), patch.object(c, "HF_HUB_CACHE", str(cache)):
            return present_engines(engines_by_name)

    def test_hf_engines_absent_from_the_cache_are_not_advertised(self):
        with tempfile.TemporaryDirectory() as d:
            present, missing = self._present({"voxcpm": VoxcpmEngine(), "dots": DotsEngine()}, d)
        self.assertEqual(present, {})
        self.assertIn("openbmb/VoxCPM2 not in the local Hugging Face cache", missing["voxcpm"])
        self.assertIn("dots-studio/dots.tts-mf not in the local Hugging Face cache", missing["dots"])

    def test_hf_engine_in_the_cache_is_advertised(self):
        with tempfile.TemporaryDirectory() as d:
            self._snapshot(d, "openbmb/VoxCPM2", ["config.json", "model.safetensors"])
            present, missing = self._present({"voxcpm": VoxcpmEngine(), "dots": DotsEngine()}, d)
        self.assertEqual(list(present), ["voxcpm"])
        self.assertEqual(list(missing), ["dots"])

    def test_a_snapshot_without_weights_is_a_partial_download(self):
        with tempfile.TemporaryDirectory() as d:
            self._snapshot(d, "openbmb/VoxCPM2", ["config.json"])
            _present, missing = self._present({"voxcpm": VoxcpmEngine()}, d)
        self.assertIn("has no weight file", missing["voxcpm"])

    def test_an_engine_pointed_at_a_local_directory_is_advertised(self):
        with tempfile.TemporaryDirectory() as d, patch.dict("os.environ", {"POLYTTS_DOTS_MODEL": d}):
            present, _missing = present_engines({"dots": DotsEngine()})
        self.assertEqual(list(present), ["dots"])

    def test_cosyvoice_follows_the_sidecars_model_dir(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict("os.environ", {"COSYVOICE_REPO": d}):
                _p, missing = present_engines({"cosyvoice": CosyvoiceEngine()})
                self.assertIn("CosyVoice model not found", missing["cosyvoice"])
                (Path(d) / "pretrained_models" / "Fun-CosyVoice3-0.5B").mkdir(parents=True)
                present, _m = present_engines({"cosyvoice": CosyvoiceEngine()})
        self.assertEqual(list(present), ["cosyvoice"])

    def test_the_presence_probe_prints_a_json_list_without_importing_torch(self):
        import json
        import subprocess
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as d:
            self._snapshot(d, "openbmb/VoxCPM2", ["model.safetensors"])
            env = {**__import__("os").environ, "HF_HUB_CACHE": d, "HF_HUB_OFFLINE": "1",
                   "COSYVOICE_REPO": d, "POLYTTS_DOTS_MODEL": "dots-studio/dots.tts-mf"}
            code = ("import runpy,sys;runpy.run_path(sys.argv[1],run_name='__main__');"
                    "assert 'torch' not in sys.modules,'torch imported'")
            out = subprocess.run([sys.executable, "-c", code, str(root / "engines.py")],
                                 env=env, capture_output=True, text=True, timeout=60, check=True)
        self.assertEqual(json.loads(out.stdout), ["voxcpm"])
        self.assertIn("engine dots not offered", out.stderr)


class VoxcpmEngineTest(unittest.TestCase):
    def test_load_reclaims_cpu_staging_allocations_and_keeps_model_warm(self):
        events = []
        loaded_model = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=48_000))

        class FakeVoxCPM:
            @staticmethod
            def from_pretrained(model_name, load_denoiser):
                events.append(("load", model_name, load_denoiser))
                return loaded_model

        fake_module = SimpleNamespace(VoxCPM=FakeVoxCPM)
        with (
            patch.dict(sys.modules, {"voxcpm": fake_module}),
            patch("engines.gc.collect", side_effect=lambda: events.append(("gc",))),
            patch("engines._trim_ram", side_effect=lambda: events.append(("trim",))),
        ):
            engine = VoxcpmEngine()
            engine.load()

        self.assertIs(engine._model, loaded_model)
        self.assertTrue(engine.loaded)
        self.assertEqual(engine.sample_rate, 48_000)
        self.assertEqual(
            events,
            [
                ("load", "openbmb/VoxCPM2", False),
                ("gc",),
                ("trim",),
            ],
        )


if __name__ == "__main__":
    unittest.main()


class LanguageNamingTest(unittest.TestCase):
    """An ISO code must mean the same thing on every endpoint.

    `/tts` took `language` raw while the streaming path mapped it through
    `_lang_name`, so `en` — what the phone sends, and what anything speaking
    BCP-47 sends — reached the engine unmapped: `Unsupported languages:
    ['en']. Supported: ['auto', 'chinese', 'english', ...]`, a 500 on the
    first English synthesis this fleet ever asked for.
    """

    def _server(self):
        import importlib.util, os, sys
        from pathlib import Path
        here = Path(__file__).resolve().parents[1] / "server.py"
        cached = sys.modules.get("polytts_server_under_test")
        if cached is not None and hasattr(cached, "_lang_name"):
            return cached
        spec = importlib.util.spec_from_file_location("polytts_server_under_test", here)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:                  # torch / livestack_node absent
            # Cached only on SUCCESS. Registering the half-built module first
            # meant the second test got it back without the import ever having
            # finished, and failed with `has no attribute` instead of skipping.
            raise unittest.SkipTest(f"server not importable here: {e}")
        sys.modules["polytts_server_under_test"] = module
        return module

    def test_iso_codes_become_the_names_the_engine_knows(self):
        srv = self._server()
        self.assertEqual(srv._lang_name("en"), "English")
        self.assertEqual(srv._lang_name("zh"), "Chinese")

    def test_a_name_passes_through(self):
        srv = self._server()
        self.assertEqual(srv._lang_name("English"), "English")

    def test_nothing_is_english_for_terminal_content(self):
        srv = self._server()
        self.assertEqual(srv._lang_name(None), "English")
        self.assertEqual(srv._lang_name(""), "English")
