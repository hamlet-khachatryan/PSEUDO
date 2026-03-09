import json
import numpy as np
import pytest

from quantify.api import load_ensemble


def test_load_ensemble_missing_first_map_raises(tmp_path):
    """load_ensemble must raise FileNotFoundError when map 0 is absent."""
    with pytest.raises(FileNotFoundError, match="First map not found"):
        load_ensemble(tmp_path / "results", "target", map_cap=5)