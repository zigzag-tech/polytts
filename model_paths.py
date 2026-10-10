"""Explicit model asset binding for immutable service source; no model imports."""
import os
from pathlib import Path


def resolve_models_dir(default_dir, environ=None):
    value = (os.environ if environ is None else environ).get('POLYTTS_MODELS_DIR')
    if value is None:
        return Path(default_dir)
    path = Path(value)
    if not value.strip() or not path.is_absolute():
        raise ValueError('POLYTTS_MODELS_DIR must be a nonempty absolute path')
    return path
