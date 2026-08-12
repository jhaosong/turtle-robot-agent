"""Tool definitions shared by registry, skill, and agent prompt."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NavigateToLocationInput(BaseModel):
    location: str = Field(description="Named location from get_known_locations")


class NavigateToPoseInput(BaseModel):
    x: float = Field(description="Target x coordinate in meters in the map frame")
    y: float = Field(description="Target y coordinate in meters in the map frame")
    yaw: float = Field(default=0.0, description="Target heading in radians")
    frame_id: Literal["map"] = Field(
        default="map",
        description="Coordinate frame; direct navigation is restricted to map",
    )


class FindObjectInput(BaseModel):
    color: str | None = Field(default=None, description="Optional object colour, e.g. blue")
    label: str | None = Field(default=None, description="Optional object label, e.g. box")


class InspectForColorInput(BaseModel):
    color: Literal["red", "green", "blue"] = Field(
        description="Color to inspect once using the TurtleBot camera"
    )


class BehaviorTreeSkillInput(BaseModel):
    goal: str = Field(description="The user's robot task expressed as a concise goal")


class WaitInput(BaseModel):
    seconds: float = Field(description="Seconds to wait, within the configured timeout")


class ClarificationInput(BaseModel):
    question: str = Field(min_length=1, max_length=300, description="One concrete question for the user")
    reason: str = Field(min_length=1, max_length=500, description="Why no safe unambiguous action is available")
