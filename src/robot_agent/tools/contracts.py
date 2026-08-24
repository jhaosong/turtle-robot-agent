"""Tool definitions shared by registry, skill, and agent prompt."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class MoveRelativeInput(BaseModel):
    distance_m: float = Field(
        ge=-2.0,
        le=2.0,
        description=(
            "Signed distance in meters along the robot's current heading; "
            "positive moves forward and negative moves backward"
        ),
    )


class FindObjectInput(BaseModel):
    color: str | None = Field(default=None, description="Optional object colour, e.g. blue")
    label: str | None = Field(default=None, description="Optional object label, e.g. box")


class SearchForObjectInput(BaseModel):
    route: list[str] = Field(
        min_length=1,
        max_length=12,
        description="Ordered known-location route to traverse while detecting",
    )
    color: str | None = Field(default=None, description="Optional target color")
    label: str | None = Field(default=None, description="Optional target class label")

    @model_validator(mode="after")
    def require_target(self) -> "SearchForObjectInput":
        if not self.color and not self.label:
            raise ValueError("search_for_object requires a color or label")
        return self


class CircleObjectForInspectionInput(BaseModel):
    label: str = Field(
        min_length=1,
        max_length=100,
        description="Open-vocabulary object label currently visible to the camera",
    )
    color: str | None = Field(default=None, description="Optional color qualifier")
    viewpoint_count: int = Field(
        default=4,
        ge=3,
        le=8,
        description="Number of evenly spaced object-centric camera viewpoints",
    )
    radius_m: float | None = Field(
        default=None,
        ge=0.6,
        le=3.0,
        description="Desired inspection radius; defaults to runtime configuration",
    )


class BehaviorTreeSkillInput(BaseModel):
    goal: str = Field(description="The user's robot task expressed as a concise goal")


class WaitInput(BaseModel):
    seconds: float = Field(description="Seconds to wait, within the configured timeout")


class ClarificationInput(BaseModel):
    question: str = Field(min_length=1, max_length=300, description="One concrete question for the user")
    reason: str = Field(min_length=1, max_length=500, description="Why no safe unambiguous action is available")
