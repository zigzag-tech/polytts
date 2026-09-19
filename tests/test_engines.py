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
