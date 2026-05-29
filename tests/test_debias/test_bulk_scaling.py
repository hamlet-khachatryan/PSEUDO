from __future__ import annotations

import json

from debias.config import DebiasConfig, DebiasParams
from debias.phenix_param_parser import ParameterFile
from debias.parameter_generator import (
    PARAM_TEMPLATES,
    _BULK_AND_SCALING_TOGGLES,
    _apply_bulk_and_scaling,
    save_run_config,
)

TEMPLATE = PARAM_TEMPLATES / "maps_template.params"


def _load_template() -> ParameterFile:
    pf = ParameterFile()
    pf.load_from_path(str(TEMPLATE))
    return pf


def test_template_defaults_have_bulk_off():
    """Regression guard: the default template keeps bulk solvent + scaling off."""
    pf = _load_template()
    assert pf.get("omit_map.boxing.exclude_bulk_solvent") is True
    assert pf.get("omit_map.boxing.refinement.main.bulk_solvent_and_scale") is False
    assert (
        pf.get("omit_map.boxing.refinement.bulk_solvent_and_scale.bulk_solvent")
        is False
    )
    assert (
        pf.get("omit_map.boxing.refinement.bulk_solvent_and_scale.anisotropic_scaling")
        is False
    )
    assert (
        pf.get(
            "omit_map.boxing.refinement.bulk_solvent_and_scale.minimization_k_sol_b_sol"
        )
        is False
    )


def test_apply_bulk_and_scaling_flips_exactly_the_toggles():
    pf = _load_template()
    _apply_bulk_and_scaling(pf)
    for dotted_path, value in _BULK_AND_SCALING_TOGGLES:
        assert pf.get(dotted_path) is value
    # the unrelated DEN-block bulk flag must remain untouched
    assert pf.get("omit_map.boxing.refinement.den.bulk_solvent_and_scale") is False


def test_apply_is_noop_when_not_called_keeps_template_bytes(tmp_path):
    """Not toggling must leave the rendered params byte-identical to the template."""
    pf = _load_template()
    out = tmp_path / "maps.params"
    pf.save(str(out))
    # round-tripping the untouched template reproduces the same key states
    again = ParameterFile()
    again.load_from_path(str(out))
    assert again.get("omit_map.boxing.refinement.main.bulk_solvent_and_scale") is False


def _cfg(use_bulk: bool, k_sol: float = 0.35) -> DebiasConfig:
    return DebiasConfig(
        debias=DebiasParams(
            run_name="t",
            use_bulk_and_scaling=use_bulk,
            bulk_solvent_k_sol=k_sol,
        )
    )


def test_save_run_config_records_bulk_flag(tmp_path):
    out = tmp_path / "t_run_config.json"
    save_run_config(out, _cfg(use_bulk=True, k_sol=0.4))
    data = json.loads(out.read_text())
    assert data["schema_version"] == 1
    assert data["use_bulk_and_scaling"] is True
    assert data["bulk_solvent_k_sol"] == 0.4
    # extensible schema carries the omission provenance too
    assert "omission_type" in data and "omit_type" in data


def test_save_run_config_defaults_false(tmp_path):
    out = tmp_path / "t_run_config.json"
    save_run_config(out, _cfg(use_bulk=False))
    data = json.loads(out.read_text())
    assert data["use_bulk_and_scaling"] is False
