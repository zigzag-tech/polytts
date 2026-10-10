import unittest
from pathlib import Path
from model_paths import resolve_models_dir


class ModelAssetBindingTests(unittest.TestCase):
    def test_default_keeps_existing_source_relative_model_root(self):
        default = Path('/original/service/models')
        self.assertEqual(resolve_models_dir(default, {}), default)

    def test_immutable_source_can_bind_existing_external_model_assets(self):
        self.assertEqual(resolve_models_dir(Path('/staged/release/models'), {'POLYTTS_MODELS_DIR': '/original/service/models'}), Path('/original/service/models'))

    def test_explicit_empty_and_relative_bindings_refuse_before_model_work(self):
        for value in ['', ' ', 'models', '../models']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'nonempty absolute path'):
                resolve_models_dir(Path('/staged/models'), {'POLYTTS_MODELS_DIR': value})
