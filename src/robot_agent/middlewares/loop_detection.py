"""Detect identical tool calls before they become an uncontrolled ReAct loop.

This is a compact adaptation of DeerFlow's loop-detection principle. It runs
at the registry boundary, so it works with all models and tool backends.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopDecision:
    allowed: bool
    count: int
    warning: bool = False


class ToolLoopDetector:
    def __init__(self, warn_threshold: int, hard_limit: int) -> None:
        if warn_threshold < 1 or hard_limit <= warn_threshold:
            raise ValueError("Loop thresholds must satisfy 1 <= warn_threshold < hard_limit")
        self.warn_threshold = warn_threshold
        self.hard_limit = hard_limit
        self._calls: Counter[str] = Counter()

    @staticmethod
    def _key(tool_name: str, arguments: dict) -> str:
        return f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"

    def check(self, tool_name: str, arguments: dict) -> LoopDecision:
        key = self._key(tool_name, arguments)
        self._calls[key] += 1
        count = self._calls[key]
        return LoopDecision(
            allowed=count < self.hard_limit,
            count=count,
            warning=count == self.warn_threshold,
        )
