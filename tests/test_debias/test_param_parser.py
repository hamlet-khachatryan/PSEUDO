import pytest
from src.debias import phenix_param_parser as parser


def test_parse_format_roundtrip_simple():
    content = """
# Example param file
alpha = 1
beta = 2.5
name = "hello world"
flag = True
nothing = None
items = a b c *special
nested {
  inner = 10
  s = 'x y'
}
"""
    parsed = parser.parse_parameters(content)
    assert parsed["alpha"] == 1
    assert parsed["beta"] == 2.5
    assert parsed["name"] == "hello world"
    assert parsed["flag"] is True
    assert parsed["nothing"] is None
    assert parsed["items"] == ["a", "b", "c", "*special"]
    assert isinstance(parsed["nested"], dict)
    assert parsed["nested"]["inner"] == 10
    out = parser.format_parameters(parsed)
    reparsed = parser.parse_parameters(out)
    assert reparsed == parsed


def test_parameterfile_save_and_load(tmp_path):
    pf = parser.ParameterFile()
    pf.params = {"a": 1, "block": {"x": "val with space", "y": [1, 2, 3]}}
    outp = tmp_path / "test_params.conf"
    pf.save(outp)
    pf2 = parser.ParameterFile()
    pf2.load_from_path(outp)
    assert pf2.get("a") == 1
    assert pf2.get("block.x") == "val with space"
    assert pf2.get("block.y") == [1, 2, 3]


def test_invalid_load_raises(tmp_path):
    pf = parser.ParameterFile()
    with pytest.raises(FileNotFoundError):
        pf.load_from_path(tmp_path / "nope.conf")
