"""Corner residuals for imported building polygon contours."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


Point = tuple[float, float]


def compute_corner_residuals(
    payload: Any,
    picked_sat: Iterable[Any],
    radius_px: float = 20.0,
    method: str = "acm.tol_1",
    collinear_tolerance_px: float = 1.0,
) -> list[dict[str, Any]]:
    """GUI-friendly wrapper for official_framefield_polygons.json payloads."""
    return match_corner_residuals(
        payload,
        picked_sat,
        radius_px=radius_px,
        method=method,
        collinear_tolerance_px=collinear_tolerance_px,
    )


def match_corner_residuals(
    official_json: Any,
    points: Iterable[Any],
    radius_px: float = 20.0,
    method: str = "acm.tol_1",
    collinear_tolerance_px: float = 1.0,
) -> list[dict[str, Any]]:
    """Match user satellite pixel points to nearby model outline corners."""
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    if collinear_tolerance_px < 0:
        raise ValueError("collinear_tolerance_px must be non-negative")

    data = _load_json(official_json)
    corners = extract_model_corners(data, method, collinear_tolerance_px)
    results: list[dict[str, Any]] = []

    for index, raw_point in enumerate(points):
        point = _xy(raw_point, f"points[{index}]")
        best = None
        best_dist = float("inf")
        for corner in corners:
            dx = corner["corner"][0] - point[0]
            dy = corner["corner"][1] - point[1]
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best = (corner, dx, dy)
                best_dist = dist

        row: dict[str, Any] = {
            "point_index": index,
            "point": point,
            "matched": best is not None and best_dist <= radius_px,
        }
        if row["matched"] and best is not None:
            corner, dx, dy = best
            row.update(
                {
                    "corner": corner["corner"],
                    "dx": dx,
                    "dy": dy,
                    "residual_px": best_dist,
                    "feature": {
                        key: value
                        for key, value in corner.items()
                        if key != "corner"
                    },
                }
            )
        else:
            row.update(
                {
                    "corner": None,
                    "dx": None,
                    "dy": None,
                    "residual_px": None,
                    "feature": None,
                }
            )
        results.append(row)

    return results


def extract_model_corners(
    official_json: Any,
    method: str = "acm.tol_1",
    collinear_tolerance_px: float = 1.0,
) -> list[dict[str, Any]]:
    """Return de-duplicated non-collinear corners from one official method."""
    data = _load_json(official_json)
    corners: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()

    for polygon_index, polygon in enumerate(_method_polygons(data, method)):
        rings = [("exterior", polygon.get("exterior_xy"))]
        rings.extend(
            (f"hole_{hole_index}", hole)
            for hole_index, hole in enumerate(polygon.get("holes_xy") or [])
        )
        for ring_name, raw_ring in rings:
            if raw_ring is None:
                continue
            ring = _clean_ring(raw_ring)
            for corner_index, corner in enumerate(
                _filter_collinear(ring, collinear_tolerance_px)
            ):
                key = (round(corner[0], 6), round(corner[1], 6))
                if key in seen:
                    continue
                seen.add(key)
                corners.append(
                    {
                        "corner": corner,
                        "polygon_index": polygon_index,
                        "ring": ring_name,
                        "corner_index": corner_index,
                    }
                )

    return corners


def _load_json(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        text = Path(value).read_text(encoding="utf-8")
        return json.loads(text)
    return value


def _method_polygons(data: Any, method: str) -> list[dict[str, Any]]:
    matches = [
        node
        for node in _walk(data)
        if isinstance(node, dict) and node.get("method") == method
    ]
    if not matches:
        raise ValueError(f"method {method!r} not found")

    polygons: list[dict[str, Any]] = []
    for match in matches:
        polygons.extend(_polygons_below(match))
    if not polygons:
        raise ValueError(f"method {method!r} has no exterior_xy polygons")
    return polygons


def _polygons_below(node: Any) -> list[dict[str, Any]]:
    return [
        item
        for item in _walk(node)
        if isinstance(item, dict) and "exterior_xy" in item
    ]


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _clean_ring(raw_ring: Any) -> list[Point]:
    if not isinstance(raw_ring, list):
        raise ValueError("polygon ring must be a list of xy points")
    points = [_xy(raw_point, "polygon ring point") for raw_point in raw_ring]

    cleaned: list[Point] = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        raise ValueError("polygon ring must contain at least three points")
    return cleaned


def _filter_collinear(points: list[Point], tolerance: float) -> list[Point]:
    kept: list[Point] = []
    count = len(points)
    for index, point in enumerate(points):
        prev_point = points[(index - 1) % count]
        next_point = points[(index + 1) % count]
        if not _nearly_collinear(prev_point, point, next_point, tolerance):
            kept.append(point)
    if len(kept) < 3:
        return []
    return kept


def _nearly_collinear(a: Point, b: Point, c: Point, tolerance: float) -> bool:
    ab_x = b[0] - a[0]
    ab_y = b[1] - a[1]
    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]
    if ab_x * bc_x + ab_y * bc_y <= 0:
        return False

    ac_x = c[0] - a[0]
    ac_y = c[1] - a[1]
    ac_len = math.hypot(ac_x, ac_y)
    if ac_len == 0:
        return True
    distance = abs((b[0] - a[0]) * ac_y - (b[1] - a[1]) * ac_x) / ac_len
    between = (
        min(a[0], c[0]) - tolerance <= b[0] <= max(a[0], c[0]) + tolerance
        and min(a[1], c[1]) - tolerance <= b[1] <= max(a[1], c[1]) + tolerance
    )
    return distance <= tolerance and between


def _xy(value: Any, name: str) -> Point:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{name} must be an xy pair")
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric x/y") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} must contain finite x/y")
    return (x, y)
