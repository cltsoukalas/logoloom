"""
Controls catalog loader.

Loads NUREG-0800 SRP criteria (or any framework catalog) from a
processed JSON file and returns them in the standard control dict format
expected by NexusRanker.

Control schema:
    {
        "id":        str,   # e.g. "SRP-15.2.1"
        "title":     str,   # Section title
        "text":      str,   # Acceptance criteria / review guidance text
        "chapter":   str,   # Chapter number/name
        "section":   str,   # Section identifier
        "framework": str,   # "NUREG-0800" | "10CFR53" etc.
    }
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CATALOG = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "nureg0800_controls.json"
)


def load_controls(path: str | Path | None = None) -> list[dict]:
    """
    Load the controls catalog from a JSON file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the controls JSON file. Defaults to the packaged
        NUREG-0800 catalog at data/processed/nureg0800_controls.json.

    Returns
    -------
    list[dict]
        List of control dicts ready for use with NexusRanker.
    """
    catalog_path = Path(path) if path else DEFAULT_CATALOG

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Controls catalog not found at {catalog_path}.\n"
            "Run scripts/build_nureg0800_catalog.py to generate it."
        )

    with open(catalog_path, encoding="utf-8") as f:
        controls = json.load(f)

    _validate_controls(controls)
    return controls


def _validate_controls(controls: list[dict]) -> None:
    """Basic schema validation on the controls list."""
    required = {"id", "title", "text"}
    for i, ctrl in enumerate(controls):
        missing = required - set(ctrl.keys())
        if missing:
            raise ValueError(
                f"Control at index {i} is missing required fields: {missing}"
            )
