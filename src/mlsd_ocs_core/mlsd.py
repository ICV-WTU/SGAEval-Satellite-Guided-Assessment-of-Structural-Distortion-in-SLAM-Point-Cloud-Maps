"""Scale-normalized local deformation over a fixed correspondence domain."""

import numpy as np


def compute_mlsd(
    transform_matrix: np.ndarray,
    method: str,
    reference_points: np.ndarray,
    num_samples: int = 1000,
) -> float:
    """Maximum scale-normalized local deformation over the correspondence region."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    transform_matrix = np.asarray(transform_matrix, dtype=float)
    reference_points = np.asarray(reference_points, dtype=float)
    if transform_matrix.ndim != 2 or not np.all(np.isfinite(transform_matrix)):
        raise ValueError("transform_matrix must be a finite matrix")
    if reference_points.ndim != 2 or reference_points.shape[1] < 2 or len(reference_points) == 0:
        raise ValueError("reference_points must contain at least one xy point")
    reference_points = reference_points[:, :2]

    def deviation(jacobian: np.ndarray) -> float:
        singular_values = np.linalg.svd(np.linalg.inv(jacobian), compute_uv=False)
        scale = np.sqrt(abs(np.prod(singular_values)))
        if scale <= 1e-9:
            raise ValueError("Jacobian scale is singular")
        return float(np.max(np.abs(singular_values / scale - 1.0)))

    if method == "affine":
        if transform_matrix.shape[0] < 2 or transform_matrix.shape[1] < 2:
            raise ValueError("affine transform_matrix must have at least a 2x2 linear block")
        return deviation(transform_matrix[:2, :2])
    if method != "homo":
        raise ValueError(f"Unsupported transformation method: {method}")
    if transform_matrix.shape != (3, 3):
        raise ValueError("homography transform_matrix must be 3x3")
    scale = float(np.max(np.abs(transform_matrix)))
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("homography transform_matrix must be non-zero and finite")
    transform_matrix = transform_matrix / scale

    lower = reference_points.min(axis=0)
    upper = reference_points.max(axis=0)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("reference_points must be finite")
    h = transform_matrix
    corners = np.array([
        [lower[0], lower[1]],
        [lower[0], upper[1]],
        [upper[0], lower[1]],
        [upper[0], upper[1]],
    ])
    corner_w = corners @ h[2, :2] + h[2, 2]
    if np.any(np.abs(corner_w) < 1e-9) or np.min(corner_w) < 0 < np.max(corner_w):
        raise ValueError("Homography denominator crosses zero inside fixed Ω")
    count = int(np.sqrt(num_samples))
    x_grid, y_grid = np.meshgrid(
        np.linspace(lower[0], upper[0], count),
        np.linspace(lower[1], upper[1], count),
    )
    max_deviation = 0.0
    valid_count = 0
    for x, y in zip(x_grid.ravel(), y_grid.ravel()):
        w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        if abs(w) < 1e-9:
            raise ValueError("Homography denominator is zero inside fixed Ω")
        u_numerator = h[0, 0] * x + h[0, 1] * y + h[0, 2]
        v_numerator = h[1, 0] * x + h[1, 1] * y + h[1, 2]
        jacobian = np.array([
            [(h[0, 0] * w - u_numerator * h[2, 0]) / w**2,
             (h[0, 1] * w - u_numerator * h[2, 1]) / w**2],
            [(h[1, 0] * w - v_numerator * h[2, 0]) / w**2,
             (h[1, 1] * w - v_numerator * h[2, 1]) / w**2],
        ])
        try:
            max_deviation = max(max_deviation, deviation(jacobian))
        except np.linalg.LinAlgError:
            raise ValueError("Homography Jacobian is singular inside fixed Ω")
        valid_count += 1
    if valid_count == 0:
        raise ValueError("No valid MLSD samples inside fixed Ω")
    return max_deviation
