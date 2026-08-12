"""Lead agent assembly modeled after DeerFlow's ``make_lead_agent`` factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from langchain.agents import create_agent

from robot_agent.agents.lead_agent.prompt import build_lead_agent_prompt


@dataclass(frozen=True)
class LeadAgentAssembly:
    agent: Any
    tools: list[Any]
    metadata: dict[str, Any]
    system_prompt: str


def _filter_tools(
    tools: Iterable[Any], allowlist: set[str] | None = None, blocklist: set[str] | None = None
) -> list[Any]:
    selected = []
    for tool in tools:
        name = getattr(tool, "name", "")
        if allowlist is not None and name not in allowlist:
            continue
        if blocklist and name in blocklist:
            continue
        selected.append(tool)
    return selected


def make_lead_agent(
    *,
    model: Any,
    tools: Iterable[Any],
    known_locations: list[str],
    max_tool_calls: int,
    middleware: list[Any] | None = None,
    allowlist: set[str] | None = None,
    blocklist: set[str] | None = None,
) -> LeadAgentAssembly:
    """Resolve final tools, metadata, prompt, then compile a LangChain v1 agent."""
    final_tools = _filter_tools(tools, allowlist=allowlist, blocklist=blocklist)
    if not final_tools:
        raise ValueError("Lead agent requires at least one enabled tool")
    metadata = {
        "agent_name": "turtlebot_ros2_lead_agent",
        "tool_names": [tool.name for tool in final_tools],
        "known_locations": known_locations,
        "architecture": "deerflow-inspired-harness",
        "max_tool_calls": max_tool_calls,
        "middleware_names": [type(item).__name__ for item in middleware or []],
    }
    prompt = build_lead_agent_prompt(
        known_locations=known_locations,
        tool_names=[tool.name for tool in final_tools],
        max_tool_calls=max_tool_calls,
    )
    agent = create_agent(
        model=model,
        tools=final_tools,
        system_prompt=prompt,
        middleware=middleware or [],
    )
    return LeadAgentAssembly(agent=agent, tools=final_tools, metadata=metadata, system_prompt=prompt)
