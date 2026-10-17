"""Complete MLSD/OCS flow with the extracted numerical implementations."""

from typing import Any, Callable

import numpy as np

from .mlsd import compute_mlsd
from .ocs import fit_transform_all_points, run_ocs_alignment as _run


def run_ocs_alignment(
    pcd_points: np.ndarray,
    picked_pcd_xy: np.ndarray,
    picked_sat_xy: np.ndarray,
    frame_field_polygons: Any,
    params: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Extract corners, sample uncertainty, and evaluate local/map OCS."""
    return _run(
        pcd_points,
        picked_pcd_xy,
        picked_sat_xy,
        frame_field_polygons,
        params,
        progress=progress,
        should_cancel=should_cancel,
        transform_fn=fit_transform_all_points,
        mlsd_fn=compute_mlsd,
    )
