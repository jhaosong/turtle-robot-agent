"""LLM-generated, validated Behavior Tree skill for TurtleBot navigation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Literal
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

from robot_agent.state import ToolResult, ToolStatus


class BehaviorTreeNode(BaseModel):
    type: Literal["GoToPose", "Wait", "Stop"] = Field(description="One supported behavior tree node type")
    location: str | None = Field(default=None, description="Known location for GoToPose")
    seconds: float | None = Field(default=None, description="Positive duration for Wait")


class BehaviorTreePlan(BaseModel):
    goal: str
    nodes: list[BehaviorTreeNode] = Field(max_length=12)


BT_GENERATION_PROMPT = """You generate a small BehaviorTree.CPP plan for a TurtleBot.

Use only these action node types:
- GoToPose(location): navigate to exactly one named location from the provided catalog.
- Wait(seconds): wait briefly.
- Stop: issue a safe stop.

The root is an ordered Sequence. Never invent locations, action names, recovery
nodes, raw velocity commands, topics, services, or ROS shell commands. Return a
plan that is minimal and directly advances the user goal.
"""


def behavior_tree_to_xml(
    plan: BehaviorTreePlan,
    tree_id: str = "GeneratedTurtleBotTree",
    navigation_retry_count: int = 1,
) -> str:
    """Export TurtleBot-demo-style navigation XML plus declared custom nodes.

    ``GoToPose`` follows the bundled TurtleBot XML contract. ``Wait`` and
    ``Stop`` need plugins in the old C++ demo; this harness remains their
    executor and does not claim the XML is independently deployable.
    """
    root = ET.Element("root", {"BTCPP_format": "4", "main_tree_to_execute": tree_id})
    behavior_tree = ET.SubElement(root, "BehaviorTree", {"ID": tree_id})
    sequence = ET.SubElement(behavior_tree, "Sequence", {"name": "GeneratedPlan"})
    if any(node.type == "GoToPose" for node in plan.nodes):
        ET.SubElement(
            sequence,
            "SetLocations",
            {
                "name": "set_locations",
                "num_locs": "{num_locs}",
                "loc_names": "{loc_names}",
                "loc_poses": "{loc_poses}",
            },
        )
    for node in plan.nodes:
        if node.type == "GoToPose":
            target = sequence
            if navigation_retry_count > 0:
                target = ET.SubElement(
                    sequence,
                    "Fallback",
                    {"name": f"navigate_with_recovery_{node.location}"},
                )
            for attempt in range(navigation_retry_count + 1):
                suffix = "primary" if attempt == 0 else f"retry_{attempt}"
                ET.SubElement(
                    target,
                    "GoToPose",
                    {
                        "name": f"go_to_{node.location}_{suffix}",
                        "loc_poses": "{loc_poses}",
                        "loc": node.location or "",
                    },
                )
        elif node.type == "Wait":
            ET.SubElement(sequence, "Wait", {"seconds": str(node.seconds)})
        elif node.type == "Stop":
            ET.SubElement(sequence, "Stop")
    models = ET.SubElement(root, "TreeNodesModel")
    set_locations = ET.SubElement(models, "Action", {"ID": "SetLocations"})
    ET.SubElement(set_locations, "output_port", {"name": "loc_poses"})
    go_to_pose = ET.SubElement(models, "Action", {"ID": "GoToPose"})
    ET.SubElement(go_to_pose, "input_port", {"name": "loc"})
    ET.SubElement(go_to_pose, "input_port", {"name": "loc_poses"})
    ET.SubElement(models, "Action", {"ID": "Wait"})
    ET.SubElement(models, "Action", {"ID": "Stop"})
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n"


@dataclass
class BehaviorTreeSkill:
    model: Any
    known_locations: set[str]
    output_directory: Path
    navigation_retry_count: int = 1

    def _validate(self, plan: BehaviorTreePlan) -> None:
        if not 0 <= self.navigation_retry_count <= 3:
            raise ValueError("Behavior tree navigation retries must be between 0 and 3")
        if not plan.nodes:
            raise ValueError("Behavior tree must contain at least one node")
        if len(plan.nodes) > 12:
            raise ValueError("Behavior tree may contain at most 12 nodes")
        for index, node in enumerate(plan.nodes):
            if node.type == "GoToPose" and node.location not in self.known_locations:
                raise ValueError(f"Unknown BT location: {node.location}")
            if node.type == "GoToPose" and node.seconds is not None:
                raise ValueError("GoToPose may not include seconds")
            if node.type == "Wait" and (node.seconds is None or node.seconds <= 0 or node.seconds > 30):
                raise ValueError("Wait node must use seconds in (0, 30]")
            if node.type == "Wait" and node.location is not None:
                raise ValueError("Wait may not include a location")
            if node.type == "Stop" and (node.location is not None or node.seconds is not None):
                raise ValueError("Stop node may not have parameters")
            if node.type == "Stop" and index != len(plan.nodes) - 1:
                raise ValueError("Stop must be the final behavior tree node")
        if plan.nodes[-1].type != "Stop":
            raise ValueError("Behavior tree must end with Stop")

    def _generate_plan(self, goal: str) -> BehaviorTreePlan:
        structured_model = self.model.with_structured_output(BehaviorTreePlan)
        response = structured_model.invoke(
            [
                ("system", BT_GENERATION_PROMPT),
                ("human", "Known locations: " + json.dumps(sorted(self.known_locations)) + "\nUser goal:\n" + goal),
            ]
        )
        plan = response if isinstance(response, BehaviorTreePlan) else BehaviorTreePlan.model_validate(response)
        self._validate(plan)
        return plan

    def _persist(self, plan: BehaviorTreePlan) -> dict[str, str]:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        json_path = self.output_directory / "behavior_tree.json"
        xml_path = self.output_directory / "behavior_tree.xml"
        json_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        xml_path.write_text(
            behavior_tree_to_xml(plan, navigation_retry_count=self.navigation_retry_count),
            encoding="utf-8",
        )
        return {"json_path": str(json_path), "xml_path": str(xml_path)}

    def generate(self, goal: str) -> ToolResult:
        """Generate and persist a BT without sending any robot command."""
        try:
            plan = self._generate_plan(goal)
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"goal": goal},
                error=f"Behavior tree generation failed: {type(exc).__name__}",
                retryable=True,
            )

        try:
            paths = self._persist(plan)
        except OSError as exc:
            return ToolResult(
                status=ToolStatus.FAILED,
                data={"goal": goal},
                error=f"Behavior tree persistence failed: {type(exc).__name__}",
                retryable=True,
            )
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={
                "goal": plan.goal,
                "nodes": [item.model_dump() for item in plan.nodes],
                "navigation_retry_count": self.navigation_retry_count,
                **paths,
            },
        )

    def run(
        self,
        goal: str,
        *,
        navigate: Callable[[str, int], ToolResult],
        stop: Callable[[int], ToolResult],
        wait: Callable[[float, int], ToolResult],
        abort: Callable[[], ToolResult],
        on_node_started: Callable[[int, BehaviorTreeNode], None] | None = None,
    ) -> ToolResult:
        """Generate once, then execute only the validated BT nodes in order."""
        generated = self.generate(goal)
        if generated.status != ToolStatus.SUCCESS:
            return generated
        plan = BehaviorTreePlan.model_validate(
            {"goal": generated.data["goal"], "nodes": generated.data["nodes"]}
        )
        node_results: list[dict[str, Any]] = []
        for index, node in enumerate(plan.nodes, start=1):
            if on_node_started is not None:
                on_node_started(index, node)
            if node.type == "GoToPose":
                attempts: list[dict[str, Any]] = []
                for attempt in range(self.navigation_retry_count + 1):
                    result = navigate(node.location or "", index)
                    attempts.append({"attempt": attempt + 1, "result": result.to_dict()})
                    if result.status == ToolStatus.SUCCESS or not result.retryable:
                        break
            elif node.type == "Wait":
                result = wait(node.seconds or 0.0, index)
                attempts = [{"attempt": 1, "result": result.to_dict()}]
            else:
                result = stop(index)
                attempts = [{"attempt": 1, "result": result.to_dict()}]
            node_results.append(
                {
                    "index": index,
                    "node": node.model_dump(),
                    "result": result.to_dict(),
                    "attempts": attempts,
                }
            )
            if result.status != ToolStatus.SUCCESS:
                abort_result = abort()
                return ToolResult(
                    status=result.status,
                    data={**generated.data, "node_results": node_results, "abort_stop": abort_result.to_dict()},
                    error=result.error or "Behavior tree node failed",
                    retryable=result.retryable,
                )
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data={**generated.data, "node_results": node_results},
        )
