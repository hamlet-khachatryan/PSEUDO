from __future__ import annotations

import re
import tempfile
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_KEY_VALUE_RE = re.compile(r"^([A-Za-z0-9_\-\.]+)\s*=\s*(.*)$")
_BLOCK_START_RE = re.compile(r"^([A-Za-z0-9_\-\.]+)\s*\{\s*$")
_QUOTED_STR_RE = re.compile(r"^(['\"])(.*)\1$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_BOOL_RE = re.compile(r"^(True|False)$", re.IGNORECASE)
_NONE_RE = re.compile(r"^None$", re.IGNORECASE)


def _tokenize_line_value(raw: str) -> List[str]:
    """Tokenize a right-hand side value into tokens."""
    raw = raw.strip()
    if raw == "":
        return []

    m = _QUOTED_STR_RE.match(raw)
    if m:
        return [m.group(2)]

    parts = re.split(r"\s+", raw)
    return [p for p in parts if p != ""]


def _parse_scalar_token(token: str) -> Any:
    """Parse a single token to int/float/bool/None/string according to patterns."""
    if _NONE_RE.match(token):
        return None
    if _BOOL_RE.match(token):
        v = token.lower()
        return True if v == "true" else False
    if _INT_RE.match(token):
        try:
            return int(token)
        except ValueError:
            pass
    if _FLOAT_RE.match(token):
        try:
            return float(token)
        except ValueError:
            pass
    return token


def parse_parameters(file_content: str) -> Dict[str, Any]:
    """
    Parse the textual content of a parameter file into a nested dictionary.

    - Supports `key = value` lines and nested blocks:
        key {
           nested_key = value
        }
    - Values can be quoted strings, ints, floats, booleans, None, or lists of tokens.
    - Comments (lines starting with '#') and blank lines are ignored.
    """
    lines = file_content.splitlines()
    i = 0
    stack: List[Dict[str, Any]] = [dict()]

    while i < len(lines):
        raw = lines[i].rstrip("\n")
        line = raw.strip()
        i += 1

        if not line or line.startswith("#"):
            continue

        m_block = _BLOCK_START_RE.match(line)
        if m_block:
            key = m_block.group(1)
            new_block: Dict[str, Any] = {}
            stack[-1][key] = new_block
            stack.append(new_block)
            continue

        if line == "}":
            if len(stack) == 1:
                logger.warning("Unmatched closing brace at line %d", i)
            else:
                stack.pop()
            continue

        m_kv = _KEY_VALUE_RE.match(line)
        if m_kv:
            k = m_kv.group(1)
            raw_val = m_kv.group(2).strip()

            tokens = _tokenize_line_value(raw_val)

            if len(tokens) == 0:
                parsed_value = ""
            elif len(tokens) == 1:
                parsed_value = _parse_scalar_token(tokens[0])
            else:
                parsed_value = [_parse_scalar_token(t) for t in tokens]

            stack[-1][k] = parsed_value
            continue

        logger.warning("Skipping unrecognized line %d: %r", i, raw)

    return stack[0]


def _format_value_for_output(value: Any) -> str:
    """
    Convert Python object into the textual representation.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        str_tokens: List[str] = []
        for v in value:
            if isinstance(v, str):
                str_tokens.append(v)
            else:
                str_tokens.append(str(v))
        return " ".join(str_tokens)
    if isinstance(value, str):
        if " " in value:
            return f'"{value}"'
        return value
    return str(value)


def format_parameters(params: Dict[str, Any], indent: int = 0) -> str:
    """
    Format the parameters dict back into a string matching the parameter file style.
    """
    lines: List[str] = []
    prefix = "  " * indent
    for key, val in params.items():
        if isinstance(val, dict):
            lines.append(f"{prefix}{key} {{")
            lines.append(format_parameters(val, indent=indent + 1))
            lines.append(f"{prefix}}}")
        else:
            formatted_value = _format_value_for_output(val)
            lines.append(f"{prefix}{key} = {formatted_value}")
    return "\n".join(lines)


@dataclass
class ParameterFile:
    """
    Impure wrapper to read/write phenix parameter files.
    """

    file_path: Optional[Path] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def load_from_path(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Parameter file not found: {p}")
        text = p.read_text(encoding="utf-8")
        self.params = parse_parameters(text)
        self.file_path = p

    def load_from_string(self, content: str) -> None:
        self.params = parse_parameters(content)

    def get(self, dotted_path: str, default: Any = None) -> Any:
        if dotted_path == "":
            return default
        parts = dotted_path.split(".")
        node: Any = self.params
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def set(self, dotted_path: str, value: Any) -> bool:
        parts = dotted_path.split(".")
        node = self.params
        for p in parts[:-1]:
            if p not in node:
                node[p] = {}
            if not isinstance(node[p], dict):
                logger.error("Cannot set path %s: %s is not a mapping", dotted_path, p)
                return False
            node = node[p]
        node[parts[-1]] = value
        return True

    def save(self, out_path: Optional[Path | str] = None, mode: int = 0o644) -> None:
        """
        Save current parameters to disk atomically using a temporary file.
        """
        if out_path is None:
            if self.file_path is None:
                raise ValueError(
                    "No output path provided and no file_path set on instance."
                )
            out_path = self.file_path
        out_path = Path(out_path)

        formatted = format_parameters(self.params) + "\n"
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp = tempfile.NamedTemporaryFile(
            prefix=".tmp_params_", dir=str(out_dir), delete=False
        )
        try:
            tmp_name = Path(tmp.name)
            tmp.write(formatted.encode("utf-8"))
            tmp.flush()
            tmp.close()
            tmp_name.chmod(mode)
            tmp_name.replace(out_path)
        except Exception:
            # Clean up temp file on failure
            try:
                tmp_name.unlink()
            except Exception:
                pass
            raise
        self.file_path = out_path
        logger.info("Saved parameters to %s", out_path)

    def __str__(self) -> str:
        return format_parameters(self.params)
