#!/usr/bin/env python3

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sim3d_dryrun.planner import BasePlanner, LLMPlanner
from sim3d_dryrun.prompts import render_executor_system_prompt
from sim3d_dryrun.tools import get_tool_summary, get_tools
from turtle_agent.scripts.llm import get_llm


def trace(label: str, payload: Any):
    if os.getenv("SIM3D_TRACE", "true").strip().lower() not in ("1", "true", "yes", "on"):
        return
    print(f"\n[SIM3D TRACE] {label}")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def compact_trace() -> bool:
    return os.getenv("SIM3D_TRACE_COMPACT", "true").strip().lower() in ("1", "true", "yes", "on")


class Sim3DDryRunAgent:
    def __init__(self, planner: BasePlanner, max_steps: int = 10):
        self._llm = get_llm(streaming=False)
        self._planner = planner
        self._tools = get_tools()
        self._tool_map = {tool_fn.name: tool_fn for tool_fn in self._tools}
        self._tool_summary = get_tool_summary()
        self._max_steps = max_steps

    def _build_executor_messages(
        self,
        user_request: str,
        approved_plan: str,
        last_called_tool: Optional[str],
        tool_history: List[Dict[str, Any]],
        step_index: int,
    ):
        execution_context = {
            "approved_plan": approved_plan,
            "last_called_tool": last_called_tool or "None",
            "tool_history": tool_history[-5:],
            "tool_catalog": self._tool_summary,
            "step_index": step_index,
        }
        if not compact_trace():
            trace("EXECUTOR CONTEXT", execution_context)
        return [
            SystemMessage(content=render_executor_system_prompt()),
            SystemMessage(
                content=(
                    "Execution context for this step:\n"
                    f"Approved plan:\n{approved_plan}\n\n"
                    f"Current step index: {step_index}\n\n"
                    f"Last called tool: {last_called_tool or 'None'}\n\n"
                    f"Recent tool history: {json.dumps(tool_history[-5:], ensure_ascii=True)}\n\n"
                    "You must follow the approved plan as a fixed meta instruction. "
                    "Choose the next tool that advances the plan. "
                    "Prefer the minimum number of tools needed. "
                    "Do not restart the plan from the beginning after every tool. "
                    "If the plan has been sufficiently covered, stop and provide the final response instead of "
                    "calling another tool."
                )
            ),
            HumanMessage(content=user_request),
        ]

    async def run(self):
        console = Console()
        console.print("3D ROS2 dry-run agent. Type `exit` to quit.\n")
        while True:
            user_request = input("> ").strip()
            if user_request.lower() == "exit":
                break
            self.execute_request(user_request)

    def execute_request(self, user_request: str) -> str:
        bound_llm = self._llm.bind_tools(self._tools)
        tool_history: List[Dict[str, Any]] = []
        last_called_tool = None
        llm_call_count = 0
        trace("USER REQUEST", user_request)
        plan = self._planner.plan(
            user_request=user_request,
            tool_summary=self._tool_summary,
        )
        llm_call_count += 1
        approved_plan = plan.to_prompt_block()

        planned_step_count = len(plan.steps)
        max_tool_steps = min(self._max_steps, planned_step_count) if planned_step_count else self._max_steps

        for step_index in range(1, max_tool_steps + 1):
            messages = self._build_executor_messages(
                user_request=user_request,
                approved_plan=approved_plan,
                last_called_tool=last_called_tool,
                tool_history=tool_history,
                step_index=step_index,
            )
            ai_message = bound_llm.invoke(messages)
            llm_call_count += 1
            if not compact_trace():
                trace(
                    "EXECUTOR RAW OUTPUT",
                    {
                        "content": getattr(ai_message, "content", ""),
                        "tool_calls": getattr(ai_message, "tool_calls", []),
                        "step_index": step_index,
                    },
                )

            tool_calls = getattr(ai_message, "tool_calls", [])
            if not tool_calls:
                break

            tool_call = tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            trace(
                "TOOL CALL SELECTED",
                {"step_index": step_index, "tool": tool_name, "args": tool_args},
            )
            tool_fn = self._tool_map[tool_name]
            tool_result = tool_fn.invoke(tool_args)
            if not compact_trace():
                trace(
                    "TOOL RESULT",
                    {"step_index": step_index, "tool": tool_name, "result": tool_result},
                )
            tool_history.append(
                {
                    "step_index": step_index,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                }
            )
            last_called_tool = tool_name

        if tool_history:
            tool_names = [entry["tool"] for entry in tool_history]
            final_response = (
                f"Tools run ({len(tool_history)}): " + " -> ".join(tool_names)
            )
        else:
            final_response = "No tools were run."

        trace("FINAL RESPONSE", final_response)
        trace(
            "REQUEST STATS",
            {"llm_api_calls": llm_call_count, "tool_calls": len(tool_history)},
        )
        return final_response


def main():
    dotenv.load_dotenv(dotenv.find_dotenv())
    os.environ.setdefault("SIM3D_TRACE", "true")
    os.environ.setdefault("SIM3D_TRACE_COMPACT", "true")
    llm = get_llm(streaming=False)
    planner = LLMPlanner(llm=llm)
    agent = Sim3DDryRunAgent(planner=planner)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
