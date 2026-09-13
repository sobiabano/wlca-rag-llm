"""Shared result type returned by every carbon agent (Section 3.6.3)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AgentOutput:
    agent: str
    result: dict
    source_cases: list
    latency_ms: float
