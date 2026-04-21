"""Tests for logoloom.data.loader"""

import json
from pathlib import Path

import pytest

from logoloom.data.loader import load_controls, _validate_controls


VALID_CONTROLS = [
    {"id": "SRP-1", "title": "Title A", "text": "Text A", "chapter": "1"},
    {"id": "SRP-2", "title": "Title B", "text": "Text B", "chapter": "1"},
]


def _write_catalog(tmp_path: Path, data: list[dict]) -> Path:
    p = tmp_path / "controls.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── load_controls ──────────────────────────────────────────────────────────────

def test_load_controls_returns_list(tmp_path):
    path = _write_catalog(tmp_path, VALID_CONTROLS)
    controls = load_controls(path)
    assert isinstance(controls, list)
    assert len(controls) == 2


def test_load_controls_returns_dicts_with_required_fields(tmp_path):
    path = _write_catalog(tmp_path, VALID_CONTROLS)
    controls = load_controls(path)
    for ctrl in controls:
        assert "id" in ctrl
        assert "title" in ctrl
        assert "text" in ctrl


def test_load_controls_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Controls catalog not found"):
        load_controls(tmp_path / "nonexistent.json")


def test_load_controls_empty_list_passes(tmp_path):
    """An empty catalog is technically valid at load time."""
    path = _write_catalog(tmp_path, [])
    controls = load_controls(path)
    assert controls == []


# ── _validate_controls ─────────────────────────────────────────────────────────

def test_validate_controls_passes_for_valid_data():
    _validate_controls(VALID_CONTROLS)   # should not raise


def test_validate_controls_missing_id_raises():
    bad = [{"title": "Title", "text": "Text"}]   # missing "id"
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_controls(bad)


def test_validate_controls_missing_title_raises():
    bad = [{"id": "SRP-1", "text": "Text"}]   # missing "title"
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_controls(bad)


def test_validate_controls_missing_text_raises():
    bad = [{"id": "SRP-1", "title": "Title"}]   # missing "text"
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_controls(bad)


def test_validate_controls_extra_fields_ignored():
    """Extra fields beyond required set should not cause validation failure."""
    controls = [{"id": "SRP-1", "title": "T", "text": "X", "chapter": "1",
                 "extra_field": "value"}]
    _validate_controls(controls)   # should not raise
