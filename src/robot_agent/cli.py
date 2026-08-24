"""Interactive entry point for the non-simulator planning/tool-calling MVP."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import dotenv

from robot_agent.config import RobotAgentSettings
from robot_agent.harness import RoboticsAgentHarness


def validate_location_file(settings: RobotAgentSettings) -> None:
    if settings.location_file.is_file():
        return
    raise FileNotFoundError(
        f"Robot location file does not exist: {settings.location_file}. "
        "Set ROBOT_AGENT_LOCATION_FILE to a readable TurtleBot location YAML file "
        "or pass --location-file PATH."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeerFlow-inspired ROS2 TurtleBot agent")
    parser.add_argument("--no-trace", action="store_true", help="Disable runtime event printing")
    parser.add_argument("--location-file", type=Path, help="Override TurtleBot location YAML")
    args = parser.parse_args()

    dotenv.load_dotenv(dotenv.find_dotenv())
    root = Path(__file__).resolve().parents[2]
    settings = RobotAgentSettings.from_env(root)
    if args.no_trace:
        settings = replace(settings, trace=False)
    if args.location_file:
        settings = replace(settings, location_file=args.location_file)
    validate_location_file(settings)

    print("ROS2 TurtleBot agent. Type `exit` to quit.")
    while True:
        goal = input("\n> ").strip()
        if goal.lower() in {"exit", "quit"}:
            return
        if not goal:
            continue
        result = RoboticsAgentHarness(settings).invoke(goal)
        print("\nFinal response:")
        print(result["final_response"])
        if result.get("clarification_question"):
            print("\nClarification needed:")
            print(result["clarification_question"])
        print("\nRun summary:")
        robot_state = result["agent_state"]["robot_state"]
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "run_directory": result["run_directory"],
                    "status": result["run_status"],
                    "tool_calls": len(result["tool_history"]),
                    "confirmed_pose": robot_state["pose"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
