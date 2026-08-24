"""Pure helpers for querying occupancy risk from a Nav2 costmap snapshot."""

from __future__ import annotations

from dataclasses import dataclass
import math


LETHAL_COST = 253
UNKNOWN_COST = 255


@dataclass(frozen=True)
class CostmapSnapshot:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    data: tuple[int, ...]

    def cost_at(self, x: float, y: float) -> int:
        cell = self.world_to_cell(x, y)
        if cell is None:
            return UNKNOWN_COST
        column, row = cell
        return self.data[row * self.width + column]

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        column = math.floor((x - self.origin_x) / self.resolution)
        row = math.floor((y - self.origin_y) / self.resolution)
        if not 0 <= column < self.width or not 0 <= row < self.height:
            return None
        return column, row

    def clearance_at(self, x: float, y: float, max_radius_m: float = 1.0) -> float:
        cell = self.world_to_cell(x, y)
        if cell is None:
            return 0.0
        center_column, center_row = cell
        max_cells = max(1, math.ceil(max_radius_m / self.resolution))
        best = max_radius_m
        for row in range(max(0, center_row - max_cells), min(self.height, center_row + max_cells + 1)):
            for column in range(max(0, center_column - max_cells), min(self.width, center_column + max_cells + 1)):
                cost = self.data[row * self.width + column]
                if cost < LETHAL_COST:
                    continue
                distance = math.hypot(column - center_column, row - center_row) * self.resolution
                best = min(best, distance)
        return best


def path_risk(
    costmap: CostmapSnapshot,
    points: list[tuple[float, float]],
) -> tuple[bool, float, int, float]:
    """Return feasible, normalized risk, maximum cost, and minimum clearance."""
    if not points:
        return False, 1.0, UNKNOWN_COST, 0.0
    costs = [costmap.cost_at(x, y) for x, y in points]
    feasible = all(cost < LETHAL_COST for cost in costs)
    max_cost = max(costs)
    known_costs = [cost for cost in costs if cost != UNKNOWN_COST]
    risk = (
        sum(known_costs) / (len(known_costs) * LETHAL_COST)
        if known_costs
        else 1.0
    )
    clearance = min(costmap.clearance_at(x, y) for x, y in points)
    return feasible, min(1.0, risk), max_cost, clearance
