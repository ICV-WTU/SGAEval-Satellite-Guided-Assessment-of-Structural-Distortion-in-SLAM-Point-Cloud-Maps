"""Existing point-cloud/image loading and display helpers for the single page."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _decimate_rows(arr: np.ndarray, max_points: int | None) -> np.ndarray:
    """Uniformly decimate along axis 0 using a strided read."""
    if not max_points or max_points <= 0 or len(arr) <= max_points:
        return np.asarray(arr)
    step = int(np.ceil(len(arr) / float(max_points)))
    return np.ascontiguousarray(arr[::step])


def load_point_cloud(path: str, max_points: int | None = None) -> np.ndarray:
    """Load a point cloud as an ``(N, 3+)`` array.

    For binary ``.pcd`` the bundled lazy reader is used: it maps the file
    instead of letting open3d build a float64 ``PointCloud`` (48 B/point) and
    then copy it again via ``np.asarray`` (24 B/point). Pass ``max_points`` to
    decimate while reading, which keeps peak memory proportional to the result
    rather than to the file.
    """
    lower = path.lower()
    if lower.endswith(".npy"):
        data = np.load(path, mmap_mode="r" if max_points else None)
        return _decimate_rows(np.asarray(data), max_points)
    if lower.endswith(".npz"):
        data = np.load(path)
        for k in data.files:
            return _decimate_rows(np.asarray(data[k]), max_points)
        raise ValueError("Empty npz file")
    if lower.endswith(".pcd"):
        try:
            from mlsd_ocs_core.gui.pcd_lazy import read_pcd_xyz

            return read_pcd_xyz(path, max_points=max_points)
        except Exception as exc:  # noqa: BLE001 - fall back to open3d below
            # Say something: the open3d path is ~15x slower and needs ~10x the
            # memory, and a silent fallback makes that impossible to notice.
            print(
                f"[load_point_cloud] fast PCD reader unavailable ({exc}); "
                "falling back to open3d (much slower and heavier)."
            )
    if lower.endswith(".pcd") or lower.endswith(".ply"):
        try:
            import open3d as o3d  # type: ignore

            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points)
            del pcd
            return _decimate_rows(points, max_points)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Reading PCD/PLY requires open3d, please install optional dependencies '[geometry,gui]' or use .npy/.npz"
            ) from exc
    raise ValueError("Unsupported point cloud format, please use .npy/.npz or install open3d to read .pcd/.ply")


def load_image(path: str) -> np.ndarray:
    try:
        import cv2
        data = np.fromfile(path, dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    
    try:
        from PIL import Image
        img = Image.open(path)
        return np.array(img.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load image file: {exc}")


def compute_pcd_transform(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    method: str = "affine",
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python library is required, please install: pip install opencv-python")
    
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    
    if method == "affine":
        if len(src) < 3:
            raise ValueError("Affine transformation requires at least 3 point pairs")
        M, inliers = cv2.estimateAffine2D(
            src, dst, ransacReprojThreshold=3.0, confidence=0.99
        )
        return M, inliers
    else:  # homo
        if len(src) < 4:
            raise ValueError("Homography transformation requires at least 4 point pairs")
        H, inliers = cv2.findHomography(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000, confidence=0.999
        )
        return H, inliers


def apply_pcd_transform(
    points_xy: np.ndarray, M_or_H: np.ndarray, method: str = "affine"
) -> np.ndarray:
    xy1 = np.hstack([points_xy, np.ones((points_xy.shape[0], 1))])
    if method == "affine":
        uv = xy1 @ M_or_H.T
        return uv
    else:  # homography
        uvw = xy1 @ M_or_H.T
        w = uvw[:, 2:3]
        w[w == 0] = 1e-9
        uv = uvw[:, :2] / w
        return uv


def color_points_by_height(z: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps
    zmin, zmax = np.min(z), np.max(z)
    if zmax - zmin < 1e-9:
        z_norm = np.zeros_like(z)
    else:
        z_norm = (z - zmin) / (zmax - zmin)
    cmap = colormaps["jet"]
    colors = (cmap(z_norm)[:, :3] * 255).astype(np.uint8)
    return colors
