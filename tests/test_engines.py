import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from engines import VoxcpmEngine


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
