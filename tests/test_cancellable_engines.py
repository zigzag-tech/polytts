import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from engines import QwenEngine, VoxcpmEngine
from synthesis_jobs import CancellationSignal, SynthesisCancelled


class CancellableEnginesTest(unittest.TestCase):
    def test_unknown_dependency_revisions_do_not_advertise_physical_cancellation(self):
        with patch('engines.importlib.metadata.version', return_value='unreviewed'):
            self.assertFalse(QwenEngine(Path('/unused')).cancellation_capability()['supported'])
            self.assertFalse(VoxcpmEngine().cancellation_capability()['supported'])

    def test_qwen_forwards_real_transformers_stopping_criteria_and_discards_cancelled_output(self):
        import torch
        cancellation = CancellationSignal()
        observed = []
        def generate_voice_clone(**kwargs):
            criteria = kwargs['stopping_criteria']
            input_ids = torch.zeros((1, 1), dtype=torch.int64)  # CPU only
            self.assertFalse(bool(criteria(input_ids, None)[0]))
            cancellation.event.set()
            self.assertTrue(bool(criteria(input_ids, None)[0]))
            observed.append(kwargs['text'])
            return [np.ones(2, dtype=np.float32)], 24000
        engine = QwenEngine(Path('/unused'))
        engine._model = SimpleNamespace(generate_voice_clone=generate_voice_clone)
        engine._prompts['public-test-voice'] = ['owned-prompt']
        with self.assertRaises(SynthesisCancelled):
            engine.generate_cancellable('Exact text', 'public-test-voice', Path('/unused'), {}, 'English', {}, cancellation)
        self.assertEqual(observed, ['Exact text'])

    def test_native_voxcpm_reads_no_voice_and_closes_generation_at_cancellation_boundary(self):
        cancellation = CancellationSignal()
        closed, calls = [], []
        def generate_streaming(**kwargs):
            calls.append(kwargs)
            try:
                yield np.ones(2, dtype=np.float32)
                cancellation.event.set()
                yield np.ones(2, dtype=np.float32)
                self.fail('generation continued after owned cancellation')
            finally:
                closed.append(True)
        engine = VoxcpmEngine()
        engine._model = SimpleNamespace(generate_streaming=generate_streaming)
        with self.assertRaises(SynthesisCancelled):
            engine.generate_cancellable('Exact English narration', None, None, None, 'English', {}, cancellation)
        self.assertEqual(closed, [True])
        self.assertEqual(calls[0]['text'], 'Exact English narration')
        self.assertFalse(calls[0]['normalize'])
        for field in ['reference_wav_path', 'prompt_wav_path', 'prompt_text']:
            self.assertNotIn(field, calls[0])


if __name__ == '__main__':
    unittest.main()
