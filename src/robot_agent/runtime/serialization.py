"""Helpers for safely persisting text received from terminals and providers."""

from __future__ import annotations

from typing import Any


def sanitize_text(value: str) -> str:
    """Replace lone UTF-16 surrogates while preserving valid Unicode text."""
    return "".join(
        "\ufffd" if 0xD800 <= ord(character) <= 0xDFFF else character
        for character in value
    )


def sanitize_json_value(value: Any) -> Any:
    """Recursively make JSON-compatible text safe for UTF-8 persistence."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {
            sanitize_text(key) if isinstance(key, str) else key: sanitize_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_json_value(item) for item in value)
    return value
