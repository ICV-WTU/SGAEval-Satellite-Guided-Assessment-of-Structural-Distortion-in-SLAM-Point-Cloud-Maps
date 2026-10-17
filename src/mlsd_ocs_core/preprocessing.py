"""Pure ground fitting and vegetation filtering used by OCS corner extraction.

Extracted from gui/components/canvas.py without its GUI dependencies.
"""

from __future__ import annotations


import numpy as np


GROUND_FIT_METHOD = "deterministic"


GROUND_SEED_PERCENTILES = (1.0, 2.5, 5.0, 10.0, 15.0, 20.0)


GROUND_MIN_INLIER_RATIO = 0.01


GROUND_REFIT_ITERATIONS = 3


GROUND_RANSAC_SEED = 0


TREE_FILTER_RESOLUTION = 0.2


TREE_FILTER_SCALES = (0.6, 1.2)


TREE_FILTER_MIN_NEIGHBORS = 20


TREE_FILTER_MAX_DROP_RATIO = 0.6


HOUGH_MIN_HEIGHT = 0.2


def _plane_normal_pca(points):
    """
    用 PCA（SVD）求一组点的最佳平面法向量。

    返回单位法向量，点数不足或退化时返回 None。
    """

    points = np.asarray(points, dtype=np.float64)

    if len(points) < 3:
        return None

    centered = points - points.mean(axis=0)

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)

    except np.linalg.LinAlgError:
        return None

    normal = vh[-1]

    norm = float(np.linalg.norm(normal))

    if norm < 1e-8:
        return None

    return normal / norm


def _fit_plane_lowest_layer(xyz, distance_threshold, percentile):
    """
    从最低的高度层出发做确定性平面拟合。

    以高度分位数附近的一薄层点作为种子，PCA 求法向，
    再用全部点重新取内点并迭代重拟合。

    全程没有随机采样，因此同一份输入必然得到同一份输出。

    返回：
        (a, b, c, d), inliers

    该分位数下没有可信地面时返回 None。
    """

    xyz = np.asarray(xyz, dtype=np.float64)

    n_points = len(xyz)

    min_inliers = max(10, int(GROUND_MIN_INLIER_RATIO * n_points))

    z_values = xyz[:, 2]

    seed_mask = (
        np.abs(z_values - float(np.percentile(z_values, percentile)))
        <= distance_threshold
    )

    if int(np.count_nonzero(seed_mask)) < 10:
        return None

    normal = _plane_normal_pca(xyz[seed_mask])

    if normal is None:
        return None

    centroid = xyz[seed_mask].mean(axis=0)

    for _ in range(GROUND_REFIT_ITERATIONS):
        offset = -float(normal @ centroid)

        distances = np.abs(xyz @ normal + offset)

        inlier_mask = distances <= distance_threshold

        if int(np.count_nonzero(inlier_mask)) < min_inliers:
            break

        centroid = xyz[inlier_mask].mean(axis=0)

        updated = _plane_normal_pca(xyz[inlier_mask])

        if updated is None:
            break

        normal = updated

    offset = -float(normal @ centroid)

    distances = np.abs(xyz @ normal + offset)

    inliers = np.flatnonzero(distances <= distance_threshold)

    if len(inliers) < min_inliers:
        return None

    return (
        np.array([normal[0], normal[1], normal[2], offset], dtype=np.float64),
        inliers,
    )


def _fit_plane_ransac(
    xyz, distance_threshold, num_iterations, random_seed=GROUND_RANSAC_SEED
):
    """
    Open3D 平面 RANSAC。

    固定随机种子，保证同一份输入得到同一份输出。
    """
    import open3d as o3d

    o3d.utility.random.seed(random_seed)

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=num_iterations
    )

    return (
        np.asarray(plane_model, dtype=np.float64),
        np.asarray(inliers, dtype=np.int64),
    )


def _fit_ground_plane_model(
    xyz,
    distance_threshold=0.05,
    num_iterations=2000,
    method=GROUND_FIT_METHOD,
    random_seed=GROUND_RANSAC_SEED,
):
    """
    拟合地面平面，返回 (a, b, c, d) 与内点索引。

    method:
        "deterministic"  先按高度分位数逐层尝试确定性拟合，
                         全部失败才回退 RANSAC

        "ransac"         直接用 Open3D RANSAC（旧行为）
    """

    if method == "ransac":
        return _fit_plane_ransac(xyz, distance_threshold, num_iterations, random_seed)

    # 逐个分位数试，取内点最多的那个。
    # 只看"第一个够用"会挑到局部低洼面：它的内点明显偏少，
    # 拟合出来的地面只贴合那一小片。
    best = None

    for percentile in GROUND_SEED_PERCENTILES:
        result = _fit_plane_lowest_layer(xyz, distance_threshold, percentile)

        if result is None:
            continue

        if best is None or len(result[1]) > len(best[1]):
            best = result

    if best is not None:
        return best

    return _fit_plane_ransac(xyz, distance_threshold, num_iterations, random_seed)


def _tree_suppression_mask(
    xyz_ground,
    ground_inliers=None,
    min_height=HOUGH_MIN_HEIGHT,
    max_height=None,
    center_xy=None,
    range_size=None,
    resolution=TREE_FILTER_RESOLUTION,
    scales=TREE_FILTER_SCALES,
    min_neighbors=TREE_FILTER_MIN_NEIGHBORS,
):
    """Return a conservative high-confidence tree mask in ground coordinates."""
    import cv2

    xyz_ground = np.asarray(xyz_ground, dtype=np.float64)
    tree_mask = np.zeros(len(xyz_ground), dtype=bool)
    stats = {
        "enabled": False,
        "candidate_points": 0,
        "tree_points": 0,
        "tree_cells": 0,
        "wall_cells": 0,
        "reason": "",
    }
    if len(xyz_ground) == 0:
        stats["reason"] = "empty"
        return tree_mask, stats

    candidate = np.asarray(xyz_ground[:, 2] >= min_height, dtype=bool)
    if max_height is not None:
        candidate &= xyz_ground[:, 2] <= max_height
    if ground_inliers is not None:
        ground_mask = np.zeros(len(xyz_ground), dtype=bool)
        ground_mask[np.asarray(ground_inliers, dtype=np.int64)] = True
        candidate &= ~ground_mask
    if center_xy is not None and range_size is not None:
        center_xy = np.asarray(center_xy, dtype=np.float64)
        range_xy = np.broadcast_to(np.asarray(range_size, dtype=np.float64), (2,))
        half = range_xy / 2.0
        candidate &= (
            (xyz_ground[:, 0] >= center_xy[0] - half[0])
            & (xyz_ground[:, 0] <= center_xy[0] + half[0])
            & (xyz_ground[:, 1] >= center_xy[1] - half[1])
            & (xyz_ground[:, 1] <= center_xy[1] + half[1])
        )

    indices = np.flatnonzero(candidate)
    stats["candidate_points"] = int(len(indices))
    if len(indices) < min_neighbors:
        stats["reason"] = "too_few_candidates"
        return tree_mask, stats

    pts = xyz_ground[indices]
    xy_min = pts[:, :2].min(axis=0)
    ij = np.floor((pts[:, :2] - xy_min) / resolution).astype(np.int64)
    width = int(ij[:, 0].max()) + 1
    height = int(ij[:, 1].max()) + 1
    if width <= 0 or height <= 0 or width * height > 4_000_000:
        stats["reason"] = "grid_too_large"
        return tree_mask, stats

    shape = (height, width)
    rows, cols = ij[:, 1], ij[:, 0]
    count = np.zeros(shape, dtype=np.float64)
    sums = [np.zeros(shape, dtype=np.float64) for _ in range(9)]
    z_min = np.full(shape, np.inf, dtype=np.float64)
    z_max = np.full(shape, -np.inf, dtype=np.float64)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    values = (x, y, z, x * x, y * y, z * z, x * y, x * z, y * z)
    np.add.at(count, (rows, cols), 1.0)
    for target, value in zip(sums, values):
        np.add.at(target, (rows, cols), value)
    np.minimum.at(z_min, (rows, cols), z)
    np.maximum.at(z_max, (rows, cols), z)

    occupied = count > 0
    occ_rows, occ_cols = np.where(occupied)

    def local_shape(radius):
        cells = max(1, int(round(radius / resolution)))
        kernel = (2 * cells + 1, 2 * cells + 1)
        local = [
            cv2.boxFilter(
                arr, -1, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT
            )
            for arr in (count, *sums)
        ]
        n = local[0]
        linearity = np.zeros(shape, dtype=np.float64)
        planarity = np.zeros(shape, dtype=np.float64)
        scattering = np.zeros(shape, dtype=np.float64)
        normal_z = np.ones(shape, dtype=np.float64)
        for r, c in zip(occ_rows, occ_cols):
            if n[r, c] < min_neighbors:
                continue
            mean = (
                np.array(
                    [
                        local[1][r, c],
                        local[2][r, c],
                        local[3][r, c],
                    ]
                )
                / n[r, c]
            )
            second = (
                np.array(
                    [
                        [local[4][r, c], local[7][r, c], local[8][r, c]],
                        [local[7][r, c], local[5][r, c], local[9][r, c]],
                        [local[8][r, c], local[9][r, c], local[6][r, c]],
                    ]
                )
                / n[r, c]
            )
            cov = second - np.outer(mean, mean)
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 0.0)
            l1, l2, l3 = vals[2], vals[1], vals[0]
            if l1 <= 1e-12:
                continue
            linearity[r, c] = (l1 - l2) / l1
            planarity[r, c] = (l2 - l3) / l1
            scattering[r, c] = l3 / l1
            normal_z[r, c] = abs(vecs[2, 0])
        return n, linearity, planarity, scattering, normal_z

    small = local_shape(scales[0])
    large = local_shape(scales[-1])
    radius_cells = max(1, int(round(scales[-1] / resolution)))
    kernel = np.ones((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=np.uint8)
    high = cv2.dilate(
        np.where(occupied, z_max, -1e9),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=-1e9,
    )
    low = -cv2.dilate(
        np.where(occupied, -z_min, -1e9),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=-1e9,
    )
    local_z_span = np.where(occupied, high - low, 0.0)
    wall_like = (large[2] >= 0.45) & (large[4] <= 0.35) & (local_z_span >= 0.8)
    wall_buffer = cv2.dilate(
        wall_like.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    tree_like = (
        (local_z_span >= 1.2)
        & (high >= min_height + 0.8)
        & (large[0] >= min_neighbors)
        & (
            (small[3] >= 0.05)
            | (large[3] >= 0.05)
            | ((large[1] <= 0.70) & (large[2] <= 0.35))
        )
        & ~wall_buffer
    )

    point_tree = tree_like[rows, cols]
    if float(point_tree.sum()) / float(len(indices)) > TREE_FILTER_MAX_DROP_RATIO:
        stats["reason"] = "drop_ratio_guard"
        return tree_mask, stats

    tree_mask[indices[point_tree]] = True
    stats.update(
        {
            "enabled": True,
            "tree_points": int(tree_mask.sum()),
            "tree_cells": int(tree_like.sum()),
            "wall_cells": int(wall_buffer.sum()),
            "reason": "ok",
        }
    )
    return tree_mask, stats
