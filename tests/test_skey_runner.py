import importlib.util
from pathlib import Path

import numpy as np


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "skey_predict.py"
    spec = importlib.util.spec_from_file_location("skey_predict", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_input_tiling_preserves_content_and_reaches_minimum():
    module = _runner_module()
    signal = np.asarray([[1.0, 2.0, 3.0]])
    tiled, changed = module.tile_short_signal(signal, sample_rate=4, min_seconds=2.0)
    assert changed is True
    assert tiled.shape == (1, 8)
    assert tiled.tolist() == [[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]]


def test_long_input_is_not_tiled():
    module = _runner_module()
    signal = np.ones((1, 20))
    output, changed = module.tile_short_signal(signal, sample_rate=4, min_seconds=2.0)
    assert changed is False
    assert output is signal
