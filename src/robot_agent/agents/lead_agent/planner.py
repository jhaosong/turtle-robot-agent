"""Optional LLM task planner, separate from the tool-calling lead agent.

The structured-output pattern is adapted from BTPG's ``GoalPlan`` generation.
The plan is context for the lead agent; it cannot issue ROS2 commands itself.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PlannedStep(BaseModel):
    description: str = Field(description="One high-level, observable task step")
    preferred_capability: Literal["navigation", "perception", "behavior_tree", "control"]


class TaskPlan(BaseModel):
    objective: str
    assumptions: list[str] = Field(default_factory=list)
    steps: list[PlannedStep] = Field(default_factory=list, max_length=8)
    requires_perception: bool = False
    requested_colors: list[Literal["red", "green", "blue"]] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_requirements(self) -> "TaskPlan":
        if self.requested_colors and not self.requires_perception:
            raise ValueError("requested_colors requires requires_perception=true")
        return self


PLANNER_PROMPT = """You plan high-level TurtleBot tasks. Produce the shortest
ordered plan that directly satisfies the goal. Choose only these capabilities:
navigation, perception, behavior_tree, control. Do not produce ROS commands,
tool arguments, raw sensor processing, or invented world facts.

The navigation capability already encapsulates localization, path planning,
obstacle handling, motion control, and arrival feedback through Nav2. Never split
those implementation details into separate perception, behavior-tree, or control
steps. If the user asks to navigate to an exact known location, produce exactly
one navigation step, set requires_perception=false, and do not add verification
or localization steps. Use behavior_tree only for an explicitly reusable or
multi-step procedure. Set requires_perception=true only when the user explicitly
asks to observe or find something. Populate requested_colors only when the user
explicitly requests red, green, or blue."""


class LeadTaskPlanner:
    def __init__(self, model: Any) -> None:
        self.model = model

    def plan(
        self,
        goal: str,
        robot_state: dict[str, Any],
        available_capabilities: set[str],
        known_locations: list[str] | None = None,
    ) -> TaskPlan:
        structured_model = self.model.with_structured_output(TaskPlan)
        response = structured_model.invoke(
            [
                ("system", PLANNER_PROMPT),
                (
                    "human",
                    "Goal:\n"
                    f"{goal}\n\nCurrent semantic state: "
                    f"{json.dumps(robot_state, ensure_ascii=False)}\n\n"
                    "Known navigation locations: "
                    f"{json.dumps(sorted(known_locations or []), ensure_ascii=False)}\n"
                    "Available capabilities: "
                    f"{json.dumps(sorted(available_capabilities))}",
                ),
            ]
        )
        plan = response if isinstance(response, TaskPlan) else TaskPlan.model_validate(response)
        unavailable = {step.preferred_capability for step in plan.steps} - available_capabilities
        if unavailable:
            raise ValueError(f"Planner selected unavailable capabilities: {sorted(unavailable)}")
        return plan
