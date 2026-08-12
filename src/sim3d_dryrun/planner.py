import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage


def _trace(label: str, payload):
    if os.getenv("SIM3D_TRACE", "true").strip().lower() not in ("1", "true", "yes", "on"):
        return
    print(f"\n[SIM3D TRACE] {label}")
    if isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def _compact_trace() -> bool:
    return os.getenv("SIM3D_TRACE_COMPACT", "true").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class PlanStep:
    step_id: str
    goal: str
    recommended_tool: str
    arguments_hint: Dict[str, Any]


@dataclass
class ExecutionPlan:
    objective: str
    assumptions: List[str]
    steps: List[PlanStep]

    def to_prompt_block(self) -> str:
        lines = [f"Objective: {self.objective}", "Assumptions:"]
        for item in self.assumptions:
            lines.append(f"- {item}")
        lines.append("Ordered steps:")
        for step in self.steps:
            lines.append(
                f"- {step.step_id}: {step.goal} | tool={step.recommended_tool} | "
                f"args_hint={json.dumps(step.arguments_hint, ensure_ascii=True)}"
            )
        return "\n".join(lines)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "assumptions": self.assumptions[:3],
            "steps": [
                {
                    "step_id": step.step_id,
                    "tool": step.recommended_tool,
                    "goal": step.goal,
                }
                for step in self.steps
            ],
        }


class BasePlanner(ABC):
    @abstractmethod
    def plan(
        self,
        user_request: str,
        tool_summary: str,
    ) -> ExecutionPlan:
        raise NotImplementedError


class LLMPlanner(BasePlanner):
    def __init__(self, llm):
        self._llm = llm

    def plan(
        self,
        user_request: str,
        tool_summary: str,
    ) -> ExecutionPlan:
        planner_prompt = (
            "You are a planning module for a ROS2 TurtleBot-style dry-run executor.\n"
            "Create a concise ordered execution plan.\n"
            "The plan must be valid for a tool-calling agent that can only call one tool at a time.\n"
            "Assume the default environment is a minimal mobile robot in Gazebo using ROS2 and standard Nav2-style "
            "interfaces.\n"
            "The user should only need to describe the task, not the ROS2, Gazebo, or Nav2 context.\n"
            "For short requests like 'go straight forward', 'turn left', or 'run in a circle of radius 1m', infer "
            "the intended robot command behavior from that default environment.\n"
            "For simple motion requests, prefer direct velocity-control steps and do not add inspection steps unless "
            "the user explicitly asks for verification.\n"
            "When the request contains a straight-line distance, prefer generate_drive_distance_command and pass "
            "distance_m directly. Do not convert the distance to duration yourself.\n"
            "When the request contains a circle radius or circular path, prefer generate_circle_motion_command and "
            "pass radius_m directly. Do not convert the circle to duration yourself.\n"
            "Do not repeat a completed motion primitive unless the user explicitly asks for repeated motion. "
            "For compound trajectories, decompose the request into the smallest ordered set of distinct motion "
            "primitives needed to describe the requested path, then stop.\n"
            "Prefer inspection before motion only when the task is ambiguous, diagnostic, or explicitly asks to check "
            "state first.\n"
            "Return JSON with keys: objective, assumptions, steps.\n"
            "Each step must contain: step_id, goal, recommended_tool, arguments_hint.\n"
            "Do not include markdown fences."
        )
        planner_input = {
            "user_request": user_request,
            "available_tools": tool_summary,
        }
        if not _compact_trace():
            _trace("PLANNER INPUT", planner_input)
        response = self._llm.invoke(
            [
                SystemMessage(content=planner_prompt),
                HumanMessage(content=json.dumps(planner_input, ensure_ascii=True, indent=2)),
            ]
        )
        raw_content = getattr(response, "content", "")
        if isinstance(raw_content, list):
            raw_content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw_content
            )
        if not _compact_trace():
            _trace("PLANNER RAW OUTPUT", raw_content)
        data = json.loads(raw_content)
        plan = ExecutionPlan(
            objective=data["objective"],
            assumptions=list(data.get("assumptions", [])),
            steps=[
                PlanStep(
                    step_id=step["step_id"],
                    goal=step["goal"],
                    recommended_tool=step["recommended_tool"],
                    arguments_hint=dict(step.get("arguments_hint", {})),
                )
                for step in data.get("steps", [])
            ],
        )
        if _compact_trace():
            _trace("PLAN SUMMARY", plan.to_summary())
        else:
            _trace(
                "PLANNER STRUCTURED PLAN",
                {
                    "objective": plan.objective,
                    "assumptions": plan.assumptions,
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "goal": step.goal,
                            "recommended_tool": step.recommended_tool,
                            "arguments_hint": step.arguments_hint,
                        }
                        for step in plan.steps
                    ],
                },
            )
        return plan
