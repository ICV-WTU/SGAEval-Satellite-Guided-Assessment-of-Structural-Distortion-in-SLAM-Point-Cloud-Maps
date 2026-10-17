from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


UNCALIBRATED_CONFIDENCE = "uncalibrated_geometric_empirical"


def run_ocs_alignment(
    pcd_points: Any,
    picked_pcd_xy: Any,
    picked_sat_xy: Any,
    frame_field_polygons: Any,
    params: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    transform_fn: Callable[[np.ndarray, np.ndarray, str], tuple[np.ndarray | None, Any]] | None = None,
    mlsd_fn: Callable[[np.ndarray, str, np.ndarray], float] | None = None,
) -> dict[str, Any]:
    params = _normalise_params(params or {})
    pcd_manual = as_xy(picked_pcd_xy, "picked_pcd_xy")
    sat_manual = as_xy(picked_sat_xy, "picked_sat_xy")
    if len(pcd_manual) != len(sat_manual):
        raise ValueError("picked_pcd_xy and picked_sat_xy must use the same identities")
    if _cancelled(should_cancel):
        return _json_ready(_empty_result("cancelled", params, len(pcd_manual)))
    _attach_joint_indices(params)

    _emit(progress, "estimating point cloud corners")
    pointcloud = _pointcloud_estimate(pcd_points, pcd_manual, params, progress, should_cancel)
    _emit(progress, "matching satellite corners")
    satellite = _satellite_estimate(sat_manual, frame_field_polygons, params, should_cancel)

    pairs = _pairs(pcd_manual, sat_manual, pointcloud, satellite)
    warnings = _warnings(params, pointcloud, satellite, pairs)
    status = "ok" if not warnings else warnings[0]
    fixed_omega = pcd_manual if params.get("fixed_omega_xy") is None else as_xy(
        params["fixed_omega_xy"], "fixed_omega_xy"
    )

    raw = _joint_ocs_from_samples(
        pcd_manual, sat_manual, pointcloud["raw_samples"], satellite["raw_samples"],
        params["tau_p_m"], params["tau_s_px"], params["identity_confirmed"],
    )
    pcd_refined_center = pointcloud["refined_xy"]
    sat_refined_center = satellite["refined_xy"]
    refined = _joint_ocs_from_samples(
        pcd_refined_center, sat_refined_center, pointcloud["refined_samples"], satellite["refined_samples"],
        params["tau_p_m"], params["tau_s_px"], params["identity_confirmed"],
    )
    mlsd = _mlsd_report(
        pcd_manual, sat_manual, pointcloud, satellite, fixed_omega, params,
        transform_fn or fit_transform_all_points, mlsd_fn, should_cancel,
    )
    if mlsd.get("reason") == "cancelled" or mlsd.get("mc", {}).get("reason") == "cancelled":
        status = "cancelled"
        if "cancelled" not in warnings:
            warnings.append("cancelled")

    return _json_ready({
        "status": status,
        "params": _public_params(params),
        "fixed_omega": fixed_omega.tolist(),
        "pairs": pairs,
        "pointcloud": _public_estimate(pointcloud, "m", pcd_points),
        "satellite": _public_estimate(satellite, "px", None),
        "raw": raw,
        "refined": refined,
        "mlsd": mlsd,
        "calibration": {
            "confidence": UNCALIBRATED_CONFIDENCE,
            "probability": "not calibrated without independent labeled calibration data",
            "pointcloud_uncertainty": pointcloud["uncertainty_note"],
            "satellite_uncertainty": satellite["uncertainty_note"],
        },
        "warnings": warnings,
        "errors": [],
    })


def evaluate_ocs(
    pointcloud_manual_xy: Any,
    pointcloud_auto_xy: Any,
    satellite_manual_xy: Any,
    satellite_auto_xy: Any,
    pointcloud_tolerance_m: float,
    satellite_tolerance_px: float | None = None,
    satellite_tolerance_m: float | None = None,
    gsd_m_per_px: float | None = None,
    pointcloud_refined_xy: Any | None = None,
    satellite_refined_xy: Any | None = None,
) -> dict[str, Any]:
    sat_tol = _satellite_tolerance_px(satellite_tolerance_px, satellite_tolerance_m, gsd_m_per_px)
    pcd_manual = as_xy(pointcloud_manual_xy, "pointcloud_manual_xy")
    sat_manual = as_xy(satellite_manual_xy, "satellite_manual_xy")
    pcd_raw = _samples(pointcloud_auto_xy, len(pcd_manual), "pointcloud_auto_xy")
    sat_raw = _samples(satellite_auto_xy, len(sat_manual), "satellite_auto_xy")
    if pointcloud_refined_xy is None:
        pcd_refined = pcd_raw
        pcd_refined_center = pcd_manual
    else:
        pcd_refined = _samples(pointcloud_refined_xy, len(pcd_manual), "pointcloud_refined_xy")
        pcd_refined_center = pcd_refined[0]
    if satellite_refined_xy is None:
        sat_refined = sat_raw
        sat_refined_center = sat_manual
    else:
        sat_refined = _samples(satellite_refined_xy, len(sat_manual), "satellite_refined_xy")
        sat_refined_center = sat_refined[0]
    return {
        "confidence_calibration": UNCALIBRATED_CONFIDENCE,
        "identity_contract": "same physical corner identity by row index; no many-to-one matching",
        "raw": _joint_ocs_from_samples(
            pcd_manual, sat_manual, pcd_raw, sat_raw, pointcloud_tolerance_m, sat_tol, True
        ),
        "refined": _joint_ocs_from_samples(
            pcd_refined_center, sat_refined_center, pcd_refined, sat_refined, pointcloud_tolerance_m, sat_tol, True
        ),
    }


def evaluate_mlsd_perturbations(
    source_xy: Any,
    target_xy: Any,
    method: str,
    transform_fn: Callable[[np.ndarray, np.ndarray, str], tuple[np.ndarray | None, Any]],
    mlsd_fn: Callable[[np.ndarray, str, np.ndarray], float],
    reference_points_xy: Any | None = None,
    perturb_source_samples: Any | None = None,
    perturb_target_samples: Any | None = None,
    epsilon: float = 1e-3,
    refined_source_xy: Any | None = None,
    refined_target_xy: Any | None = None,
) -> dict[str, Any]:
    source = as_xy(source_xy, "source_xy")
    target = as_xy(target_xy, "target_xy")
    reference = source if reference_points_xy is None else as_xy(reference_points_xy, "reference_points_xy")
    refined_source = source if refined_source_xy is None else as_xy(refined_source_xy, "refined_source_xy")
    refined_target = target if refined_target_xy is None else as_xy(refined_target_xy, "refined_target_xy")
    source_samples = _samples_or_base(perturb_source_samples, source, "perturb_source_samples")
    target_samples = _samples_or_base(perturb_target_samples, target, "perturb_target_samples")
    return _mlsd_mc(
        refined_source, refined_target, source_samples, target_samples,
        method, reference, epsilon, transform_fn, mlsd_fn,
    )


def fit_transform_all_points(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    source = as_xy(source_xy, "source_xy")
    target = as_xy(target_xy, "target_xy")
    if len(source) != len(target):
        raise ValueError("source and target must have the same number of points")
    if method == "affine":
        if len(source) < 3:
            raise ValueError("affine transform requires at least 3 point pairs")
        design = np.column_stack([source, np.ones(len(source))])
        _check_design(design, 3, "affine")
        coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
        return coeff.T, np.ones((len(source), 1), dtype=np.uint8)
    if method != "homo":
        raise ValueError(f"unsupported transform method: {method}")
    if len(source) < 4:
        raise ValueError("homography requires at least 4 point pairs")
    src_norm, src_t = _normalize_points_2d(source)
    dst_norm, dst_t = _normalize_points_2d(target)
    rows = []
    for (x, y), (u, v) in zip(src_norm, dst_norm):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=float)
    _check_design(design, 8, "homography")
    _, _, vh = np.linalg.svd(design)
    h_norm = vh[-1].reshape(3, 3)
    h = np.linalg.inv(dst_t) @ h_norm @ src_t
    scale = float(np.max(np.abs(h)))
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("homography solve produced a zero or non-finite matrix")
    return h / scale, np.ones((len(source), 1), dtype=np.uint8)


def match_satellite_corners_by_identity(
    manual_satellite_xy: Any,
    framefield_corners_xy: Any,
    tolerance_px: float,
) -> dict[str, Any]:
    manual = as_xy(manual_satellite_xy, "manual_satellite_xy")
    corners = as_xy(framefield_corners_xy, "framefield_corners_xy")
    used: set[int] = set()
    matched: list[list[float] | None] = []
    rows: list[dict[str, Any]] = []
    for i, point in enumerate(manual):
        distances = np.linalg.norm(corners - point, axis=1) if len(corners) else np.array([])
        available = [int(j) for j in np.argsort(distances) if int(j) not in used and distances[j] <= tolerance_px]
        if not available:
            matched.append(None)
            rows.append({"point_index": i, "matched": False, "reason": "no_one_to_one_match"})
            continue
        if len(available) > 1:
            matched.append(None)
            rows.append({"point_index": i, "matched": False, "reason": "ambiguous_framefield_match"})
            continue
        best = available[0]
        used.add(best)
        matched.append(corners[best].tolist())
        rows.append({
            "point_index": i,
            "matched": True,
            "corner_index": best,
            "corner_xy_px": corners[best].tolist(),
            "residual_px": float(distances[best]),
        })
    return {"matched_xy": matched, "matches": rows}


def extract_framefield_corners(
    official_json: Any,
    method: str = "acm.tol_1",
    collinear_tolerance_px: float = 1.0,
) -> list[dict[str, Any]]:
    corners: list[dict[str, Any]] = []
    for polygon_index, polygon in enumerate(_method_polygons(official_json, method)):
        rings = [("exterior", polygon.get("exterior_xy"))]
        rings.extend((f"hole_{i}", hole) for i, hole in enumerate(polygon.get("holes_xy") or []))
        for ring_name, raw_ring in rings:
            if raw_ring is None:
                continue
            filtered = _filter_collinear(_clean_ring(raw_ring), collinear_tolerance_px)
            seen_in_ring: set[tuple[float, float]] = set()
            for corner_index, corner in enumerate(filtered):
                key = (round(corner[0], 6), round(corner[1], 6))
                if key in seen_in_ring:
                    continue
                seen_in_ring.add(key)
                prev_corner = filtered[(corner_index - 1) % len(filtered)]
                next_corner = filtered[(corner_index + 1) % len(filtered)]
                corners.append({
                    "corner": corner,
                    "polygon_index": polygon_index,
                    "ring": ring_name,
                    "corner_index": corner_index,
                    "prev_corner": prev_corner,
                    "next_corner": next_corner,
                    "edge_in": (corner[0] - prev_corner[0], corner[1] - prev_corner[1]),
                    "edge_out": (next_corner[0] - corner[0], next_corner[1] - corner[1]),
                })
    return corners


def as_xy(points: Any, name: str = "points") -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must be an Nx2+ array")
    arr = arr[:, :2]
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite coordinates")
    return np.ascontiguousarray(arr)


def _pointcloud_estimate(
    pcd_points: Any,
    manual: np.ndarray,
    params: dict[str, Any],
    progress: Callable[[str], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> dict[str, Any]:
    payload = params.get("pointcloud_estimate")
    source = "params.pointcloud_estimate"
    if payload is None:
        payload = {
            "raw_xy": params.get("pointcloud_auto_xy"),
            "refined_xy": params.get("pointcloud_refined_xy"),
            "raw_samples": params.get("pointcloud_raw_samples_xy", params.get("perturb_source_samples")),
            "refined_samples": params.get("pointcloud_refined_samples_xy"),
            "covariances": params.get("pointcloud_covariances"),
        }
        source = "params.pointcloud_*"
    if payload.get("raw_xy") is None:
        try:
            from mlsd_ocs_core.ocs_geometry import estimate_pointcloud_corners

            source = "evo.ocs_geometry.estimate_pointcloud_corners"
            payload = estimate_pointcloud_corners(
                pcd_points, manual, params, progress=progress, should_cancel=should_cancel
            )
        except Exception as exc:
            return _estimate_missing(len(manual), "missing_pointcloud_estimator", str(exc), source)
    return _estimate_from_payload(
        payload, len(manual), "m", source, params["n_mc"], params.get("rng_seed", 0), params.get("_joint_mc_indices")
    )


def _satellite_estimate(
    manual: np.ndarray, frame_field_polygons: Any, params: dict[str, Any],
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    payload = params.get("satellite_estimate")
    matches = None
    source = "params.satellite_estimate"
    if payload is None:
        payload = {
            "raw_xy": params.get("satellite_auto_xy"),
            "refined_xy": params.get("satellite_refined_xy"),
            "raw_samples": params.get("satellite_raw_samples_xy", params.get("perturb_target_samples")),
            "refined_samples": params.get("satellite_refined_samples_xy"),
            "covariances": params.get("satellite_covariances"),
        }
        source = "params.satellite_*"
    if payload["raw_xy"] is None:
        mode = params.get("satellite_landmark_mode", "framefield")
        if mode != "image_lines" and frame_field_polygons is not None:
            from mlsd_ocs_core.ocs_geometry import extract_satellite_corners as extract_satellite

            matches = extract_satellite(
                frame_field_polygons,
                manual,
                method=params.get("framefield_method", "acm.tol_1"),
                tolerance_px=params["satellite_match_radius_px"],
                collinear_tolerance_px=params.get("collinear_tolerance_px", 1.0),
            )
            source = "evo.ocs_geometry.extract_satellite_corners"
            payload["raw_xy"] = matches.get("corner_xy")
            payload["rows"] = matches.get("rows")
        elif mode != "image_lines" and params.get("framefield_corners_xy") is not None:
            source = "framefield_corners_xy"
            matches = match_satellite_corners_by_identity(manual, params["framefield_corners_xy"], params["satellite_match_radius_px"])
            payload["raw_xy"] = matches["matched_xy"]
            payload["rows"] = matches["matches"]
        if mode in ("image_lines", "framefield_then_image_lines"):
            image_path = params.get("satellite_image_path")
            image_reason = None
            image_matches = None
            if image_path is None:
                image_reason = "missing_satellite_image_path"
            else:
                from mlsd_ocs_core.ocs_satellite_landmarks import extract_satellite_line_intersections

                try:
                    image_matches = extract_satellite_line_intersections(
                        image_path, manual, tolerance_px=params["satellite_match_radius_px"],
                        should_cancel=should_cancel,
                    )
                except (OSError, ValueError, ImportError) as exc:
                    image_reason = f"satellite_image_extraction_failed: {type(exc).__name__}: {exc}"
            if image_matches is None:
                if mode == "image_lines" or matches is None:
                    return _estimate_missing(len(manual), "satellite_image_unavailable", image_reason or "unknown",
                                             "image_line_intersection")
                matches = {**matches, "fallback_reason": image_reason}
            else:
                matches = (_merge_satellite_landmarks(matches, image_matches)
                           if mode == "framefield_then_image_lines" and matches is not None
                           else image_matches)
                payload["raw_xy"] = matches["corner_xy"]
                payload["rows"] = matches["rows"]
                source = "image_line_intersection" if mode == "image_lines" else "framefield_then_image_lines"
            payload["valid_rate"] = matches.get("valid_rate") if matches else None
    if payload["raw_xy"] is not None and payload.get("raw_samples") is None and payload.get("covariances") is None:
        floor = float(params.get("satellite_error_floor_px", 1.0))
        payload["covariances"] = np.tile(np.eye(2)[None, :, :] * floor * floor, (len(manual), 1, 1))
    estimate = _estimate_from_payload(
        payload, len(manual), "px", source, params["n_mc"], params.get("rng_seed", 0) + 101, params.get("_joint_mc_indices")
    )
    estimate["matches"] = matches
    if source in ("image_line_intersection", "framefield_then_image_lines"):
        estimate["uncertainty_note"] = (
            "Original-image finite line intersections; physical landmark identity is unverified. "
            "No empirical image re-extraction samples; covariance/error floor is an assumed model."
        )
    if source == "framefield" and estimate["raw_sample_model"] == "deterministic":
        estimate["uncertainty_note"] = "FrameField vertices only; no empirical satellite samples supplied."
    return estimate


def _merge_satellite_landmarks(framefield: dict[str, Any], image: dict[str, Any]) -> dict[str, Any]:
    """Supplement missing FF candidates; preserve existing unique/ambiguous evidence."""
    old_rows = framefield.get("rows", framefield.get("matches", []))
    if len(old_rows) != len(image["rows"]):
        raise ValueError("FrameField/image identity count mismatch")
    rows = []
    for old, new in zip(old_rows, image["rows"]):
        old = dict(old)
        if "valid" not in old and "matched" in old:
            old["valid"] = bool(old["matched"])
            if "corner_xy_px" in old:
                old["corner_xy"] = old["corner_xy_px"]
        use_existing = bool(old.get("valid") or old.get("candidates")
                            or old.get("reason") == "ambiguous_framefield_match")
        rows.append({**(old if use_existing else new),
                     "landmark_source": "framefield" if use_existing else "image_line_intersection"})
    # Reject reuse symmetrically, including evidence owned by an ambiguous point.
    for i, row in enumerate(rows):
        for other in rows[i + 1:]:
            if "image_line_intersection" not in (row["landmark_source"], other["landmark_source"]):
                continue
            first = row.get("candidates") or ([row] if row.get("valid") else [])
            second = other.get("candidates") or ([other] if other.get("valid") else [])
            if any(np.any(np.linalg.norm(_satellite_evidence_xy(a)[:, None, :]
                                         - _satellite_evidence_xy(b)[None, :, :], axis=2) <= 2.0)
                   for a in first for b in second):
                row["valid"] = other["valid"] = False
                row["reason"] = other["reason"] = "satellite_feature_reused_across_points"
    points = [row["corner_xy"] if row.get("valid") else [math.nan, math.nan] for row in rows]
    count = sum(bool(row.get("valid")) for row in rows)
    cancelled = image.get("status") == "cancelled"
    return {"rows": rows, "corner_xy": points, "source": "framefield_then_image_lines",
            "valid": bool(rows and count == len(rows) and not cancelled),
            "status": "cancelled" if cancelled else "ok" if rows and count == len(rows) else "partial",
            "valid_rate": count / len(rows) if rows else None,
            "identity_confirmed": False, "coordinate_frame": "original_image_pixels_origin_top_left",
            "fallback_policy": "only missing candidates; existing ambiguity preserved",
            "image_evidence": image}


def _estimate_from_payload(
    payload: dict[str, Any],
    count: int,
    unit: str,
    source: str,
    n_mc: int,
    rng_seed: int,
    shared_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    raw_xy = _xy_or_none(payload.get("raw_xy"), count)
    refined_xy = _xy_or_none(payload.get("refined_xy"), count)
    if refined_xy is None:
        refined_xy = raw_xy
    covariances = _covariances_or_none(payload.get("covariances"), count)
    raw_samples, raw_model = _samples_from_payload(
        payload.get("raw_samples"), raw_xy, covariances, n_mc, rng_seed, shared_indices
    )
    original_samples = None
    sample_indices = None
    if payload.get("raw_samples") is not None:
        original_samples = _samples(payload["raw_samples"], count, "raw_samples")
        sample_indices = _resample_indices(len(original_samples), n_mc, rng_seed, shared_indices)
    if payload.get("refined_samples") is None and raw_xy is not None and refined_xy is not None and len(raw_samples):
        refined_samples = raw_samples + (refined_xy - raw_xy).reshape(1, count, 2)
        refined_model = raw_model
    else:
        refined_samples, refined_model = _samples_from_payload(
            payload.get("refined_samples"), refined_xy, covariances, n_mc, rng_seed, shared_indices
        )
    status = "ok" if raw_xy is not None and len(raw_samples) else "missing_auto_corners"
    return {
        "status": status,
        "reason": None if status == "ok" else status,
        "raw_xy": raw_xy,
        "refined_xy": refined_xy,
        "raw_samples": raw_samples,
        "original_raw_samples": original_samples,
        "raw_sample_indices": sample_indices,
        "raw_sampling_n_mc": n_mc,
        "raw_sampling_seed": rng_seed,
        "refined_samples": refined_samples,
        "raw_sample_model": raw_model,
        "refined_sample_model": refined_model,
        "invalid_count": _invalid_sample_count(raw_samples) + _invalid_sample_count(refined_samples),
        "covariances": covariances,
        "valid_rate": payload.get("valid_rate"),
        "reasons": payload.get("reasons") or {},
        "correlation_model": payload.get("correlation_model", "joint_empirical" if shared_indices is not None else "assumed_independent"),
        "source": source,
        "matches": None,
        "rows": payload.get("rows") or [],
        "uncertainty_note": (
            "empirical samples supplied" if raw_model == "empirical"
            else "assumed covariance/error-floor model" if raw_model == "assumed_gaussian_covariance"
            else "deterministic point estimate; no empirical uncertainty samples supplied"
        ),
        "unit": unit,
    }


def _estimate_missing(count: int, status: str, reason: str, source: str) -> dict[str, Any]:
    empty = np.empty((0, count, 2), dtype=float)
    return {
        "status": status,
        "reason": reason,
        "raw_xy": None,
        "refined_xy": None,
        "raw_samples": empty,
        "refined_samples": empty,
        "raw_sample_model": "missing",
        "refined_sample_model": "missing",
        "invalid_count": 0,
        "covariances": None,
        "valid_rate": None,
        "reasons": {},
        "correlation_model": "none",
        "source": source,
        "matches": None,
        "rows": [],
        "uncertainty_note": reason,
        "unit": None,
    }


def _joint_ocs_from_samples(
    pcd_center: np.ndarray | None,
    sat_center: np.ndarray | None,
    pcd_samples: np.ndarray,
    sat_samples: np.ndarray,
    tau_p: float,
    tau_s: float,
    identity_confirmed: bool,
) -> dict[str, Any]:
    attempts = max(len(pcd_samples), len(sat_samples))
    if pcd_center is None or sat_center is None:
        pair_count = 0
        if pcd_samples.ndim == 3 and pcd_samples.shape[1]:
            pair_count = int(pcd_samples.shape[1])
        elif sat_samples.ndim == 3 and sat_samples.shape[1]:
            pair_count = int(sat_samples.shape[1])
        return _score_none("missing_centers", attempts, pair_count)
    if not identity_confirmed:
        return _score_none("identity_unconfirmed", attempts, len(pcd_center))
    if attempts == 0:
        return _score_none("missing_samples", 0, len(pcd_center))
    success_count = 0
    invalid_count = 0
    pair_success = np.zeros(len(pcd_center), dtype=int)
    pair_valid = np.zeros(len(pcd_center), dtype=int)
    for i in range(attempts):
        pcd = pcd_samples[i % len(pcd_samples)] if len(pcd_samples) else None
        sat = sat_samples[i % len(sat_samples)] if len(sat_samples) else None
        if pcd is None or sat is None:
            invalid_count += len(pcd_center)
            continue
        if pcd.ndim != 2 or sat.ndim != 2 or pcd.shape != pcd_center.shape or sat.shape != sat_center.shape:
            invalid_count += len(pcd_center)
            continue
        valid = np.all(np.isfinite(pcd), axis=1) & np.all(np.isfinite(sat), axis=1)
        invalid_count += int(len(valid) - np.count_nonzero(valid))
        both = (
            (np.linalg.norm(pcd - pcd_center, axis=1) <= tau_p)
            & (np.linalg.norm(sat - sat_center, axis=1) <= tau_s)
            & valid
        )
        pair_success += both.astype(int)
        pair_valid += valid.astype(int)
        success_count += int(np.count_nonzero(both))
    denominator = attempts * len(pcd_center)
    return {
        "ocs": float(success_count / denominator) if denominator else None,
        "valid_rate": float(np.sum(pair_valid) / denominator) if denominator else None,
        "valid_count": int(np.sum(pair_valid)),
        "conditional_success_rate": float(success_count / np.sum(pair_valid)) if np.sum(pair_valid) else None,
        "attempts": int(attempts),
        "success_count": int(success_count),
        "invalid_count": int(invalid_count),
        "per_pair": [
            {
                "identity": int(i),
                "success_rate": float(pair_success[i] / attempts),
                "valid_rate": float(pair_valid[i] / attempts),
                "conditional_success_rate": float(pair_success[i] / pair_valid[i]) if pair_valid[i] else None,
            }
            for i in range(len(pcd_center))
        ],
        "reason": None,
    }


def _mlsd_report(
    pcd_manual: np.ndarray,
    sat_manual: np.ndarray,
    pointcloud: dict[str, Any],
    satellite: dict[str, Any],
    fixed_omega: np.ndarray,
    params: dict[str, Any],
    transform_fn: Callable[[np.ndarray, np.ndarray, str], tuple[np.ndarray | None, Any]],
    mlsd_fn: Callable[[np.ndarray, str, np.ndarray], float] | None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": params["method"],
        "raw": None,
        "raw_all": None,
        "raw_common": None,
        "refined": None,
        "availability": None,
        "mc": _score_none("not_run", 0, len(pcd_manual)),
        "ocs_map": None,
        "ocs_map_reason": None,
        "candidate_ocs": None,
        "epsilon": params["epsilon_mlsd"],
        "correlation_model": _combined_correlation(pointcloud, satellite),
    }
    if mlsd_fn is None:
        out["reason"] = "mlsd_fn_required"
        return out
    raw_all = _single_mlsd(pcd_manual, sat_manual, params["method"], fixed_omega, transform_fn, mlsd_fn)
    out["raw"] = raw_all
    out["raw_all"] = raw_all
    availability = _availability_report(pcd_manual, sat_manual, pointcloud, satellite, params, transform_fn, mlsd_fn, fixed_omega)
    out["availability"] = availability
    common_ids = availability["identities"]
    if common_ids:
        common = np.asarray(common_ids, dtype=int)
        out["raw_common"] = _single_mlsd(
            pcd_manual[common], sat_manual[common], params["method"], fixed_omega, transform_fn, mlsd_fn
        )
    else:
        out["raw_common"] = {"value": None, "reason": "no_common_available_identities"}
    if not params["identity_confirmed"]:
        out["reason"] = "identity_unconfirmed"
        out["refined"] = _candidate_refined_none("identity_unconfirmed", availability)
        out["ocs_map_reason"] = "identity_unconfirmed"
        out["candidate_ocs"] = _candidate_ocs_none("identity_unconfirmed", availability)
        return out
    refined = _candidate_refined_report(availability, params, fixed_omega, transform_fn, mlsd_fn, should_cancel)
    out["refined"] = refined
    out["candidate_ocs"] = _candidate_ocs_report(
        refined, availability, pointcloud, satellite, params, fixed_omega, transform_fn, mlsd_fn, should_cancel
    )
    if out["candidate_ocs"].get("cancelled"):
        out["ocs_map_reason"] = "cancelled"
        out["reason"] = "cancelled"
        out["mc"] = _score_none("cancelled", 0, len(common_ids))
        out["mc"]["planned_attempts"] = max(len(pointcloud["raw_samples"]), len(satellite["raw_samples"]))
        out["mc"]["cancelled"] = True
        return out
    if refined.get("status") == "unique" and len(pointcloud["raw_samples"]) and len(satellite["raw_samples"]) and common_ids:
        common = np.asarray(common_ids, dtype=int)
        refined_source = np.asarray([refined["unique_source_xy_m"][i] for i in range(len(common))], dtype=float)
        refined_target = np.asarray([refined["unique_target_xy_px"][i] for i in range(len(common))], dtype=float)
        out["mc"] = _mlsd_mc(
            refined_source,
            refined_target,
            pointcloud["raw_samples"][:, common, :],
            satellite["raw_samples"][:, common, :],
            params["method"],
            fixed_omega,
            params["epsilon_mlsd"],
            transform_fn,
            mlsd_fn,
            should_cancel,
        )
        out["ocs_map"] = out["mc"]["ocs_map"]
        out["ocs_map_reason"] = out["mc"].get("reason")
    elif refined.get("status") == "ambiguous":
        out["ocs_map_reason"] = "refined_candidate_baseline_not_unique"
    else:
        out["ocs_map_reason"] = refined.get("reason") or availability.get("reason")
    return out


def _mlsd_mc(
    refined_source: np.ndarray,
    refined_target: np.ndarray,
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    method: str,
    fixed_omega: np.ndarray,
    epsilon: float,
    transform_fn: Callable[[np.ndarray, np.ndarray, str], tuple[np.ndarray | None, Any]],
    mlsd_fn: Callable[[np.ndarray, str, np.ndarray], float],
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    planned_attempts = max(len(source_samples), len(target_samples))
    attempts = 0
    cancelled = False
    refined = _single_mlsd(refined_source, refined_target, method, fixed_omega, transform_fn, mlsd_fn)
    refined_value = refined.get("value")
    values: list[float] = []
    within = 0
    invalid = 0
    for i in range(planned_attempts):
        if _cancelled(should_cancel):
            cancelled = True
            break
        attempts += 1
        try:
            source = source_samples[i % len(source_samples)]
            target = target_samples[i % len(target_samples)]
            if _sample_invalid(source) or _sample_invalid(target):
                raise ValueError("invalid sample")
            sample = _single_mlsd(source, target, method, fixed_omega, transform_fn, mlsd_fn)
            value = sample.get("value")
            if value is None or not math.isfinite(value):
                raise ValueError(sample.get("reason", "invalid MLSD"))
            values.append(float(value))
            if refined_value is not None and abs(float(value) - float(refined_value)) <= epsilon:
                within += 1
        except Exception:
            invalid += 1
    interval = [float(x) for x in np.percentile(np.asarray(values), [2.5, 97.5])] if values else None
    reason = None
    if cancelled:
        reason = "cancelled"
    elif refined_value is None:
        reason = "not_estimable"
    elif attempts and not values:
        reason = "all_samples_invalid"
    return {
        "refined_mlsd": refined_value,
        "planned_attempts": int(planned_attempts),
        "cancelled": cancelled,
        "mlsd_samples": values,
        "mlsd_interval_95": interval,
        "valid_count": len(values),
        "attempts": int(attempts),
        "failure_count": int(invalid),
        "valid_rate": float((attempts - invalid) / attempts) if attempts else None,
        "success_count": int(within),
        "conditional_stability_rate": float(within / len(values)) if values and refined_value is not None and not cancelled else None,
        "ocs_map": None if attempts == 0 or cancelled else float(within / attempts),
        "epsilon": float(epsilon),
        "reason": reason,
    }


def _single_mlsd(
    source: np.ndarray,
    target: np.ndarray,
    method: str,
    fixed_omega: np.ndarray,
    transform_fn: Callable[[np.ndarray, np.ndarray, str], tuple[np.ndarray | None, Any]],
    mlsd_fn: Callable[[np.ndarray, str, np.ndarray], float],
) -> dict[str, Any]:
    try:
        _validate_fit_points(source, target)
        matrix, inliers = transform_fn(source, target, method)
        if matrix is None:
            raise ValueError("transform estimation failed")
        value = float(mlsd_fn(matrix, method, fixed_omega))
        if not math.isfinite(value):
            raise ValueError("non-finite MLSD")
        return {
            "value": value,
            "matrix": np.asarray(matrix, dtype=float).tolist(),
            "inliers": None if inliers is None else np.asarray(inliers).astype(int).ravel().tolist(),
        }
    except Exception as exc:
        return {"value": None, "reason": str(exc)}


def _candidate_refined_none(reason: str, availability: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "insufficient",
        "value": None,
        "reason": reason,
        "candidate_range": None,
        "range_type": "conditional_candidate_sensitivity_not_ci",
        "candidate_results": [],
        "scenario_count": int(availability.get("scenario_count", 0)),
        "evaluated_count": 0,
        "truncated": False,
    }


def _availability_report(
    pcd_manual: np.ndarray,
    sat_manual: np.ndarray,
    pointcloud: dict[str, Any],
    satellite: dict[str, Any],
    params: dict[str, Any],
    transform_fn: Callable[[np.ndarray, np.ndarray, str], tuple[np.ndarray | None, Any]],
    mlsd_fn: Callable[[np.ndarray, str, np.ndarray], float],
    fixed_omega: np.ndarray,
) -> dict[str, Any]:
    pcd_options = [_pointcloud_candidate_options(i, pcd_manual[i], pointcloud) for i in range(len(pcd_manual))]
    sat_options = [_satellite_candidate_options(i, satellite) for i in range(len(sat_manual))]
    excluded: list[dict[str, Any]] = []
    identities: list[int] = []
    for i, (pc_opts, sat_opts) in enumerate(zip(pcd_options, sat_options)):
        reasons = []
        if not pc_opts:
            reasons.append("missing_pointcloud_candidate")
        if not sat_opts:
            reasons.append("missing_satellite_candidate")
        if reasons:
            excluded.append({"identity": i, "reasons": reasons})
        else:
            identities.append(i)
    required = 4 if params["method"] == "homo" else 3
    full_area = _convex_hull_area(pcd_manual)
    common_manual = pcd_manual[np.asarray(identities, dtype=int)] if identities else np.empty((0, 2), dtype=float)
    common_area = _convex_hull_area(common_manual) if len(common_manual) >= 3 else None
    coverage = None if full_area is None or full_area <= 0.0 or common_area is None else float(common_area / full_area)
    reason = None
    raw_common_status = _single_mlsd(
        pcd_manual[np.asarray(identities, dtype=int)] if identities else np.empty((0, 2)),
        sat_manual[np.asarray(identities, dtype=int)] if identities else np.empty((0, 2)),
        params["method"],
        fixed_omega,
        transform_fn,
        mlsd_fn,
    ) if len(identities) >= required else {"value": None, "reason": f"{params['method']} requires at least {required} available identities"}
    if full_area is None or full_area <= 0.0:
        reason = "manual_identity_hull_degenerate"
    elif len(identities) < required:
        reason = "insufficient_available_identities"
    elif common_area is None or common_area <= 0.0:
        reason = "available_identity_hull_degenerate"
    elif raw_common_status.get("value") is None:
        reason = raw_common_status.get("reason", "raw_common_fit_invalid")
    elif coverage is not None and coverage < params["availability_coverage_threshold"]:
        reason = "available_identity_coverage_below_threshold"
    scenario_count = 1
    for i in identities:
        scenario_count *= max(1, len(pcd_options[i]) * len(sat_options[i]))
    return {
        "status": "ok" if reason is None else "insufficient",
        "reason": reason,
        "identities": identities,
        "excluded": excluded,
        "required_min_identities": required,
        "coverage": coverage,
        "total_identity_count": len(pcd_manual),
        "available_identity_count": len(identities),
        "point_count_coverage": float(len(identities) / len(pcd_manual)) if len(pcd_manual) else None,
        "coverage_threshold": float(params["availability_coverage_threshold"]),
        "coverage_threshold_type": "heuristic_manual_convex_hull_area_ratio",
        "manual_hull_area": full_area,
        "available_hull_area": common_area,
        "scenario_count": int(scenario_count if identities else 0),
        "pointcloud_candidate_counts": [len(item) for item in pcd_options],
        "satellite_candidate_counts": [len(item) for item in sat_options],
        "pointcloud_options": pcd_options,
        "satellite_options": sat_options,
    }


def _candidate_refined_report(
    availability: dict[str, Any],
    params: dict[str, Any],
    fixed_omega: np.ndarray,
    transform_fn: Callable[[np.ndarray, np.ndarray, str], tuple[np.ndarray | None, Any]],
    mlsd_fn: Callable[[np.ndarray, str, np.ndarray], float],
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    if availability.get("status") != "ok":
        return _candidate_refined_none(availability.get("reason", "availability_not_estimable"), availability)
    identities = list(availability["identities"])
    cap = max(0, int(params["candidate_combo_cap"]))
    per_identity_options = []
    for identity in identities:
        per_identity_options.append(list(itertools.product(
            availability["pointcloud_options"][identity],
            availability["satellite_options"][identity],
        )))
    total = int(availability.get("scenario_count", 0))
    candidate_results: list[dict[str, Any]] = []
    values: list[float] = []
    cancelled = False
    for scenario_index, combo in enumerate(itertools.islice(itertools.product(*per_identity_options), cap)):
        if _cancelled(should_cancel):
            cancelled = True
            break
        source = np.asarray([pc["xy"] for pc, _sat in combo], dtype=float)
        target = np.asarray([sat["xy"] for _pc, sat in combo], dtype=float)
        collision = _pointcloud_candidate_collision([pc for pc, _sat in combo])
        if collision is None:
            collision = _satellite_feature_collision([sat for _pc, sat in combo])
        if collision is not None:
            sample = {"value": None, "reason": collision}
        else:
            sample = _single_mlsd(source, target, params["method"], fixed_omega, transform_fn, mlsd_fn)
        value = sample.get("value")
        status = "ok" if value is not None and math.isfinite(float(value)) else "invalid"
        if status == "ok":
            values.append(float(value))
        candidate_results.append({
            "scenario_index": int(scenario_index),
            "status": status,
            "value": None if value is None else float(value),
            "reason": sample.get("reason"),
            "candidates": [
                {
                    "identity": int(identity),
                    "pointcloud_candidate_index": int(pc["candidate_index"]),
                    "satellite_candidate_index": int(sat["candidate_index"]),
                    "pcd_xy_m": pc["xy"].tolist(),
                    "sat_xy_px": sat["xy"].tolist(),
                    "pointcloud_source": pc["source"],
                    "satellite_source": sat["source"],
                    "satellite_feature_key": sat.get("feature_key"),
                }
                for identity, (pc, sat) in zip(identities, combo)
            ],
        })
    truncated = total > cap
    aggregation = None
    if not values:
        status = "insufficient"
        value = None
        reason = "all_candidate_scenarios_invalid"
        candidate_range = None
    elif total == 1 and not truncated and len(values) == 1:
        status = "unique"
        value = values[0]
        reason = None
        candidate_range = [values[0], values[0]]
        aggregation = "single_candidate"
    else:
        status = "ambiguous"
        value = float(np.mean(np.asarray(values, dtype=float)))
        reason = "candidate_baseline_not_unique"
        candidate_range = [float(min(values)), float(max(values))]
        aggregation = "arithmetic_mean_of_valid_candidates"
    out = {
        "status": status,
        "value": value,
        "reason": reason,
        "candidate_range": candidate_range,
        "range_type": "conditional_candidate_sensitivity_not_ci",
        "aggregation": aggregation,
        "aggregation_note": None if aggregation != "arithmetic_mean_of_valid_candidates" else "mean is computed over valid evaluated candidate scenarios only; it is not a probability-weighted estimate",
        "candidate_results": candidate_results,
        "scenario_count": total,
        "evaluated_count": len(candidate_results),
        "valid_count": len(values),
        "truncated": bool(truncated),
        "truncated_reason": None if not truncated else "candidate_combo_cap_reached_partial_range",
    }
    if cancelled:
        out.update({"status": "cancelled", "reason": "cancelled", "value": None,
                    "candidate_range": None, "cancelled": True, "partial": True,
                    "unprocessed_count": total - len(candidate_results)})
    if status == "unique" and not cancelled:
        first = candidate_results[0]["candidates"]
        out["unique_source_xy_m"] = [row["pcd_xy_m"] for row in first]
        out["unique_target_xy_px"] = [row["sat_xy_px"] for row in first]
    return out


def _pointcloud_candidate_options(index: int, manual_xy: np.ndarray, estimate: dict[str, Any]) -> list[dict[str, Any]]:
    refined = estimate.get("refined_xy")
    raw = estimate.get("raw_xy")
    row = _diag_row(estimate, index) or {}
    if refined is not None and index < len(refined) and np.all(np.isfinite(refined[index])):
        return [{"candidate_index": 0, "xy": np.asarray(refined[index], dtype=float), "source": "nominal_refined"}]
    if raw is not None and index < len(raw) and np.all(np.isfinite(raw[index])):
        return [{"candidate_index": 0, "xy": np.asarray(raw[index], dtype=float), "source": "nominal_raw_fallback"}]
    if row.get("reason") != "ambiguous_double_wall":
        return []
    metadata = row.get("metadata") or {}
    candidates = row.get("candidates") or []
    if metadata.get("candidate_frame") != "ground_xy":
        return []
    rotation = metadata.get("ground_rotation_input_to_ground")
    center = metadata.get("ground_center")
    crop_width = metadata.get("crop_width_m")
    if rotation is None or center is None or crop_width is None:
        return []
    rot = np.asarray(rotation, dtype=float)
    cen = np.asarray(center, dtype=float)
    try:
        width = np.broadcast_to(np.asarray(crop_width, dtype=float), (2,))
    except Exception:
        return []
    if rot.shape != (3, 3) or cen.shape[0] < 3 or not np.all(np.isfinite(rot)) or not np.all(np.isfinite(cen[:3])):
        return []
    if not np.all(np.isfinite(width)) or np.any(width <= 0.0):
        return []
    manual_radius = max(1.5, 0.18 * float(np.min(width / 2.0)))
    out: list[dict[str, Any]] = []
    seen_xy: set[tuple[int, int]] = set()
    for cand_idx, candidate in enumerate(candidates):
        distance = float(candidate.get("distance_to_manual", math.nan))
        if not math.isfinite(distance) or distance < 0.0 or distance > manual_radius:
            continue
        if "corner_xy" not in candidate:
            continue
        ground_xy = np.asarray(candidate["corner_xy"], dtype=float)[:2]
        if not np.all(np.isfinite(ground_xy)):
            continue
        input_xy = (np.r_[ground_xy, 0.0] @ rot.T + cen)[:2]
        if np.all(np.isfinite(input_xy)):
            key = _xy_key(input_xy)
            if key in seen_xy:
                continue
            seen_xy.add(key)
            out.append({
                "candidate_index": cand_idx,
                "xy": input_xy.astype(float),
                "source": "ambiguous_double_wall_candidate",
                "ground_xy": ground_xy.tolist(),
                "distance_to_manual": distance,
            })
    return out


def _candidate_ocs_none(reason: str, availability: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "unavailable", "reason": reason,
        "scenario_count": int(availability.get("scenario_count", 0)),
        "evaluated_count": 0, "unavailable_count": 0,
        "candidate_results": [], "ocs_map_range": None,
        "valid_rate_range": None, "conditional_stability_rate_range": None,
        "truncated": False, "probability_weights": None,
        "range_type": "conditional_candidate_sensitivity_not_ci",
        "interpretation": "stability about each candidate's own center; does not establish accuracy",
    }


def _candidate_samples(
    estimate: dict[str, Any], candidates: list[dict[str, Any]], side: str,
    round_indices: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str | None, str]:
    """Substitute measured branch rounds before resampling, never shift another branch."""
    identity_key = f"{side}_candidate_index"
    source_key = f"{side}_source"
    xy_key = "pcd_xy_m" if side == "pointcloud" else "sat_xy_px"
    ambiguous = [row for row in candidates if row[source_key] not in ("nominal_refined", "nominal_raw_fallback")]
    identities = np.asarray([row["identity"] for row in candidates], dtype=int)
    if not ambiguous:
        original = estimate.get("original_raw_samples")
        if round_indices is not None and original is not None:
            if len(round_indices) and np.max(round_indices) >= len(original):
                return None, f"{side}_joint_bootstrap_round_count_mismatch", "missing"
            return original[round_indices][:, identities, :], None, estimate["raw_sample_model"]
        samples = estimate["raw_samples"]
        if not len(samples):
            return None, f"{side}_samples_missing", "missing"
        return samples[:, identities, :], None, estimate["raw_sample_model"]
    original = estimate.get("original_raw_samples")
    indices = estimate.get("raw_sample_indices")
    combined = None
    if original is not None and len(original):
        combined = original[:, identities, :].copy()
    branch_initialized = False
    for item in ambiguous:
        identity = item["identity"]
        row = _diag_row(estimate, identity) or {}
        matches = [branch for branch in row.get("candidate_bootstrap", [])
                   if branch.get("candidate_index") == item[identity_key]]
        if len(matches) != 1:
            return None, f"{side}_candidate_bootstrap_missing_or_duplicate", "missing"
        branch = matches[0]
        try:
            samples = np.asarray(branch["samples"], dtype=float)
            center = np.asarray(branch["center_xy"], dtype=float)
            expected_center = np.asarray(item[xy_key], dtype=float)
            # With no unique nominal identity, geometry retains one legacy NaN
            # placeholder. Complete branch rounds supply the actual evidence.
            if combined is None or (
                len(combined) == 1 and not branch_initialized and len(ambiguous) == len(candidates)
                and np.all(np.isnan(combined)) and len(samples) > 1
            ):
                if len(ambiguous) != len(candidates):
                    return None, f"{side}_joint_bootstrap_rounds_missing", "missing"
                combined = np.full((len(samples), len(candidates), 2), np.nan)
                indices = _resample_indices(
                    len(samples), estimate["raw_sampling_n_mc"], estimate["raw_sampling_seed"], None
                )
            if samples.shape != (len(combined), 2) or branch.get("attempts") != len(combined):
                return None, f"{side}_candidate_bootstrap_round_count_mismatch", "missing"
            if center.shape != (2,) or not np.allclose(center, expected_center, rtol=0.0, atol=1e-6):
                return None, f"{side}_candidate_bootstrap_center_mismatch", "missing"
            valid = int(np.count_nonzero(np.all(np.isfinite(samples), axis=1)))
            if branch.get("valid_count") != valid:
                return None, f"{side}_candidate_bootstrap_valid_count_mismatch", "missing"
        except (KeyError, TypeError, ValueError):
            return None, f"{side}_candidate_bootstrap_invalid", "missing"
        column = int(np.flatnonzero(identities == identity)[0])
        combined[:, column, :] = samples
        branch_initialized = True
    if round_indices is not None:
        if len(round_indices) and np.max(round_indices) >= len(combined):
            return None, f"{side}_joint_bootstrap_round_count_mismatch", "missing"
        indices = round_indices
    return np.ascontiguousarray(combined[indices]), None, "empirical_candidate_bootstrap"


def _candidate_ocs_report(
    refined: dict[str, Any], availability: dict[str, Any],
    pointcloud: dict[str, Any], satellite: dict[str, Any], params: dict[str, Any],
    fixed_omega: np.ndarray,
    transform_fn: Callable[[np.ndarray, np.ndarray, str], tuple[np.ndarray | None, Any]],
    mlsd_fn: Callable[[np.ndarray, str, np.ndarray], float],
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    out = _candidate_ocs_none(refined.get("reason") or "not_estimable", availability)
    out["truncated"] = bool(refined.get("truncated"))
    out["truncated_reason"] = refined.get("truncated_reason")
    if refined.get("cancelled"):
        out.update({"status": "cancelled", "reason": "cancelled", "cancelled": True,
                    "partial": True, "unprocessed_count": out["scenario_count"]})
        return out
    results = []
    cancelled = False
    for scenario in refined.get("candidate_results", []):
        if _cancelled(should_cancel):
            cancelled = True
            break
        candidates = scenario["candidates"]
        result = {
            "scenario_index": scenario["scenario_index"], "candidates": candidates,
            "status": "unavailable", "reason": scenario.get("reason"), "mc": None,
        }
        if scenario["status"] == "ok":
            joint_indices = None
            joint_reason = None
            if _combined_correlation(pointcloud, satellite) == "joint_empirical":
                source_count = _candidate_round_count(pointcloud, candidates, "pointcloud")
                target_count = _candidate_round_count(satellite, candidates, "satellite")
                if source_count != target_count:
                    joint_reason = "joint_bootstrap_round_count_mismatch"
                elif source_count:
                    joint_indices = _resample_indices(
                        source_count, pointcloud["raw_sampling_n_mc"], pointcloud["raw_sampling_seed"], None
                    )
            source_samples, source_reason, source_model = _candidate_samples(
                pointcloud, candidates, "pointcloud", joint_indices
            )
            target_samples, target_reason, target_model = _candidate_samples(
                satellite, candidates, "satellite", joint_indices
            )
            result["sample_models"] = {"pointcloud": source_model, "satellite": target_model}
            result["correlation_model"] = _combined_correlation(pointcloud, satellite)
            result["reason"] = joint_reason or source_reason or target_reason
            if result["reason"] is None:
                source = np.asarray([row["pcd_xy_m"] for row in candidates], dtype=float)
                target = np.asarray([row["sat_xy_px"] for row in candidates], dtype=float)
                result["mc"] = _mlsd_mc(
                    source, target, source_samples, target_samples, params["method"],
                    fixed_omega, params["epsilon_mlsd"], transform_fn, mlsd_fn, should_cancel,
                )
                result["status"] = "evaluated"
                if result["mc"].get("cancelled"):
                    result["status"] = "cancelled"
                    result["reason"] = "cancelled"
                    cancelled = True
        results.append(result)
        if cancelled:
            break
    out["candidate_results"] = results
    out["evaluated_count"] = sum(row["status"] == "evaluated" for row in results)
    out["unavailable_count"] = sum(row["status"] == "unavailable" for row in results)
    out["unprocessed_count"] = out["scenario_count"] - out["evaluated_count"] - out["unavailable_count"]
    out["cancelled"] = cancelled
    out["partial"] = bool(cancelled or out["truncated"] or out["unavailable_count"])
    for key in ("ocs_map", "valid_rate", "conditional_stability_rate"):
        values = [row["mc"][key] for row in results if row["mc"] is not None and row["mc"][key] is not None]
        out[f"{key}_range"] = [min(values), max(values)] if values else None
    if cancelled:
        out["status"] = "cancelled"
        out["reason"] = "cancelled"
        for key in ("ocs_map", "valid_rate", "conditional_stability_rate"):
            out[f"{key}_range"] = None
    elif out["evaluated_count"]:
        out["status"] = "partial" if out["truncated"] or out["unavailable_count"] else "evaluated"
        out["reason"] = "partial_candidate_coverage" if out["status"] == "partial" else None
    elif results:
        out["reason"] = "all_candidate_uncertainty_unavailable"
    return out


def _candidate_round_count(estimate: dict[str, Any], candidates: list[dict[str, Any]], side: str) -> int:
    """Count real rounds, excluding a fully ambiguous legacy NaN placeholder."""
    original = estimate.get("original_raw_samples")
    if original is not None and len(original):
        if len(original) != 1 or np.any(np.isfinite(original)):
            return len(original)
    for item in candidates:
        row = _diag_row(estimate, item["identity"]) or {}
        for branch in row.get("candidate_bootstrap", []):
            if branch.get("candidate_index") == item[f"{side}_candidate_index"]:
                try:
                    return int(branch.get("attempts", 0))
                except (TypeError, ValueError, OverflowError):
                    return 0
    return 0


def _satellite_candidate_options(index: int, estimate: dict[str, Any]) -> list[dict[str, Any]]:
    refined = estimate.get("refined_xy")
    raw = estimate.get("raw_xy")
    row = _diag_row(estimate, index) or {}
    if refined is not None and index < len(refined) and np.all(np.isfinite(refined[index])):
        return [{
            "candidate_index": 0,
            "xy": np.asarray(refined[index], dtype=float),
            "source": "nominal_refined",
            "feature_key": _feature_key(row.get("feature")),
            "feature": row.get("feature"),
        }]
    if raw is not None and index < len(raw) and np.all(np.isfinite(raw[index])):
        return [{
            "candidate_index": 0,
            "xy": np.asarray(raw[index], dtype=float),
            "source": "nominal_raw_fallback",
            "feature_key": _feature_key(row.get("feature")),
            "feature": row.get("feature"),
        }]
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cand_idx, candidate in enumerate(row.get("candidates") or []):
        xy = candidate.get("corner_xy", candidate.get("xy"))
        if xy is None:
            continue
        arr = np.asarray(xy, dtype=float)[:2]
        if not np.all(np.isfinite(arr)):
            continue
        feature_key = _feature_key(candidate.get("feature"))
        dedup_key = (tuple(feature_key) if feature_key is not None else None, _xy_key(arr))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append({
            "candidate_index": int(candidate.get("candidate_index", cand_idx)),
            "xy": arr.astype(float),
            "source": row.get("reason") or "satellite_candidate",
            "feature_key": feature_key,
            "feature": candidate.get("feature"),
        })
    return out


def _satellite_evidence_xy(record: dict[str, Any]) -> np.ndarray:
    """Preserve measured cluster members when rejecting cross-identity reuse."""
    observations = [record]
    feature = record.get("feature")
    if isinstance(feature, dict) and feature.get("source") == "image_line_intersection":
        observations += feature.get("members") or []
    points = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        try:
            xy = np.asarray(observation.get("corner_xy", observation.get("xy")), dtype=float)
        except (TypeError, ValueError):
            continue
        if xy.shape == (2,) and np.isfinite(xy).all():
            points.append(xy)
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _feature_key(feature: Any) -> list[Any] | None:
    if not isinstance(feature, dict):
        return None
    keys = ["polygon_index", "ring", "corner_index"]
    if all(key in feature for key in keys):
        return [feature["polygon_index"], feature["ring"], feature["corner_index"]]
    if feature.get("source") == "image_line_intersection":
        members = feature.get("members") or []
        representative = feature.get("representative_member", 0)
        if isinstance(representative, int) and 0 <= representative < len(members):
            point = members[representative].get("corner_xy")
            if point is not None and len(point) == 2 and np.isfinite(point).all():
                return ["image_line_intersection", float(point[0]), float(point[1])]
        return ["image_line_intersection"]
    return None


def _satellite_feature_collision(options: list[dict[str, Any]]) -> str | None:
    seen_feature: set[tuple[Any, ...]] = set()
    seen_xy: set[tuple[int, int]] = set()
    for index, option in enumerate(options):
        key = option.get("feature_key")
        is_image = bool(key and key[0] == "image_line_intersection")
        if key is not None and not is_image:
            tup = tuple(key)
            if tup in seen_feature:
                return "satellite_candidate_feature_reused_across_identities"
            seen_feature.add(tup)
        for previous in options[:index]:
            previous_key = previous.get("feature_key")
            if is_image or (previous_key and previous_key[0] == "image_line_intersection"):
                if np.any(np.linalg.norm(_satellite_evidence_xy(option)[:, None, :]
                                         - _satellite_evidence_xy(previous)[None, :, :], axis=2) <= 2.0):
                    return "satellite_image_landmark_reused_across_identities"
        xy_key = _xy_key(option.get("xy"))
        if xy_key in seen_xy:
            return "satellite_candidate_xy_reused_across_identities"
        seen_xy.add(xy_key)
    return None


def _pointcloud_candidate_collision(options: list[dict[str, Any]]) -> str | None:
    seen_xy: set[tuple[int, int]] = set()
    for option in options:
        key = _xy_key(option["xy"])
        if key in seen_xy:
            return "pointcloud_candidate_xy_reused_across_identities"
        seen_xy.add(key)
    return None


def _xy_key(value: Any, scale: float = 1e6) -> tuple[int, int]:
    arr = np.asarray(value, dtype=float)[:2]
    return (int(round(float(arr[0]) * scale)), int(round(float(arr[1]) * scale)))


def _convex_hull_area(points: np.ndarray) -> float | None:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 3 or not np.all(np.isfinite(pts[:, :2])):
        return None
    pts = pts[:, :2]
    center = pts.mean(axis=0)
    scale = float(np.max(np.abs(pts - center)))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    norm = (pts - center) / scale
    unique = sorted({(float(x), float(y)) for x, y in norm})
    if len(unique) < 3:
        return None
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None
    arr = np.asarray(hull, dtype=float)
    area_norm = 0.5 * abs(float(np.dot(arr[:, 0], np.roll(arr[:, 1], -1)) - np.dot(arr[:, 1], np.roll(arr[:, 0], -1))))
    area = area_norm * scale * scale
    if area <= 1e-12 or not math.isfinite(area):
        return None
    return float(area)


def _validate_fit_points(source: np.ndarray, target: np.ndarray) -> None:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape or source.shape[1] < 2:
        raise ValueError("fit points must be matching Nx2 arrays")
    bad = np.flatnonzero(
        ~np.all(np.isfinite(source[:, :2]), axis=1)
        | ~np.all(np.isfinite(target[:, :2]), axis=1)
    )
    if len(bad):
        raise ValueError(f"missing/non-finite corner identities: {bad.tolist()}")


def _pairs(
    pcd_manual: np.ndarray,
    sat_manual: np.ndarray,
    pointcloud: dict[str, Any],
    satellite: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(pcd_manual)):
        p_raw = None if pointcloud["raw_xy"] is None else pointcloud["raw_xy"][i]
        p_ref = None if pointcloud["refined_xy"] is None else pointcloud["refined_xy"][i]
        s_raw = None if satellite["raw_xy"] is None else satellite["raw_xy"][i]
        s_ref = None if satellite["refined_xy"] is None else satellite["refined_xy"][i]
        ok = all(item is not None and np.all(np.isfinite(item)) for item in (p_raw, s_raw))
        rows.append({
            "identity": i,
            "status": "ok" if ok else "missing_auto_corner",
            "manual": {"pcd_xy_m": pcd_manual[i].tolist(), "sat_xy_px": sat_manual[i].tolist()},
            "raw": {"pcd_xy_m": _row_or_none(p_raw), "sat_xy_px": _row_or_none(s_raw)},
            "refined": {"pcd_xy_m": _row_or_none(p_ref), "sat_xy_px": _row_or_none(s_ref)},
            "offsets": {
                "raw_pcd_m": _offset_or_none(pcd_manual[i], p_raw),
                "raw_sat_px": _offset_or_none(sat_manual[i], s_raw),
                "refined_pcd_m": _offset_or_none(pcd_manual[i], p_ref),
                "refined_sat_px": _offset_or_none(sat_manual[i], s_ref),
            },
            "samples": {
                "pcd_raw": _sample_count_for_pair(pointcloud["raw_samples"], i),
                "pcd_refined": _sample_count_for_pair(pointcloud["refined_samples"], i),
                "sat_raw": _sample_count_for_pair(satellite["raw_samples"], i),
                "sat_refined": _sample_count_for_pair(satellite["refined_samples"], i),
            },
            "reasons": [] if ok else ["missing_auto_corner"],
            "diagnostics": {
                "pointcloud": _diag_row(pointcloud, i),
                "satellite": _diag_row(satellite, i),
            },
        })
    return rows


def _diag_row(estimate: dict[str, Any], index: int) -> dict[str, Any] | None:
    rows = estimate.get("rows") or []
    if index < len(rows):
        return rows[index]
    return None


def _normalise_params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    out["method"] = str(out.get("method", "affine"))
    out["identity_confirmed"] = bool(out.get("identity_confirmed", False))
    out["tau_p_m"] = float(out.get("tau_p_m", out.get("pointcloud_tolerance_m", 0.5)))
    out["tau_s_px"] = float(out.get("tau_s_px", out.get("satellite_tolerance_px", 2.0)))
    out["epsilon_mlsd"] = float(out.get("epsilon_mlsd", out.get("ocs_map_epsilon", 0.05)))
    out["satellite_match_radius_px"] = float(out.get("satellite_match_radius_px", 20.0))
    out["n_mc"] = int(out.get("n_mc", 200))
    out["n_bootstrap"] = int(out.get("n_bootstrap", 40))
    out["rng_seed"] = int(out.get("rng_seed", out.get("seed", 0)))
    out["seed"] = out["rng_seed"]
    out["crop_width_m"] = float(out.get("crop_width_m", out.get("crop_width", 40.0)))
    out["voxel_m"] = float(out.get("voxel_m", out.get("voxel", 0.2)))
    out["satellite_error_floor_px"] = float(out.get("satellite_error_floor_px", 1.0))
    out["framefield_method"] = out.get("framefield_method", "acm.tol_1")
    out["candidate_combo_cap"] = int(out.get("candidate_combo_cap", 256))
    out["candidate_bootstrap"] = bool(out.get("candidate_bootstrap", False))
    out["candidate_association"] = str(out.get("candidate_association", "radius_unique"))
    if out["candidate_association"] not in ("radius_unique", "wall_geometry"):
        raise ValueError("candidate_association must be radius_unique or wall_geometry")
    out["candidate_angle_tolerance_deg"] = float(out.get("candidate_angle_tolerance_deg", 8.0))
    out["candidate_cost_margin"] = float(out.get("candidate_cost_margin", 0.15))
    out["satellite_landmark_mode"] = str(out.get("satellite_landmark_mode", "framefield"))
    if out["satellite_landmark_mode"] not in ("framefield", "image_lines", "framefield_then_image_lines"):
        raise ValueError("Unknown satellite_landmark_mode")
    radius = out.get("candidate_match_radius_m")
    out["candidate_match_radius_m"] = None if radius is None else float(radius)
    out["availability_coverage_threshold"] = float(out.get("availability_coverage_threshold", 0.5))
    if not math.isfinite(out["availability_coverage_threshold"]):
        raise ValueError("availability_coverage_threshold must be finite")
    out["availability_coverage_threshold"] = min(1.0, max(0.0, out["availability_coverage_threshold"]))
    return out


def _attach_joint_indices(params: dict[str, Any]) -> None:
    source = params.get("pointcloud_raw_samples_xy", params.get("perturb_source_samples"))
    target = params.get("satellite_raw_samples_xy", params.get("perturb_target_samples"))
    if source is None or target is None:
        return
    source_count = len(source)
    target_count = len(target)
    if source_count == 0 or target_count == 0:
        return
    rng = np.random.default_rng(params.get("rng_seed", 0))
    params["_joint_mc_indices"] = rng.integers(0, min(source_count, target_count), params["n_mc"])


def _public_params(params: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "method", "identity_confirmed", "tau_p_m", "tau_s_px", "epsilon_mlsd",
        "satellite_match_radius_px", "n_mc", "n_bootstrap", "gsd_m_per_px",
        "rng_seed", "seed", "crop_width_m", "voxel_m", "satellite_error_floor_px", "framefield_method",
        "candidate_combo_cap", "availability_coverage_threshold",
        "candidate_bootstrap", "candidate_match_radius_m",
        "candidate_association", "candidate_angle_tolerance_deg", "candidate_cost_margin",
        "corner_selection",
        "satellite_landmark_mode", "satellite_image_path",
    ]
    return {key: params.get(key) for key in keys if key in params}


def _warnings(params: dict[str, Any], pointcloud: dict[str, Any], satellite: dict[str, Any], pairs: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if not params["identity_confirmed"]:
        warnings.append("identity_unconfirmed")
    if pointcloud["status"] != "ok":
        warnings.append(pointcloud["status"])
    if satellite["status"] != "ok":
        warnings.append(satellite["status"])
    if any(pair["status"] != "ok" for pair in pairs):
        warnings.append("missing_pair_auto_corner")
    return list(dict.fromkeys(warnings))


def _public_estimate(estimate: dict[str, Any], unit: str, pcd_points: Any) -> dict[str, Any]:
    out = {
        "status": estimate["status"],
        "reason": estimate["reason"],
        f"raw_xy_{unit}": _array_or_none(estimate["raw_xy"]),
        f"refined_xy_{unit}": _array_or_none(estimate["refined_xy"]),
        "sample_count": int(len(estimate["raw_samples"])),
        "invalid_sample_count": int(estimate["invalid_count"]),
        "correlation_model": estimate["correlation_model"],
        "source": estimate["source"],
        "sample_model": estimate["raw_sample_model"],
        "covariances": _array_or_none(estimate.get("covariances")),
        "valid_rate": estimate.get("valid_rate"),
        "reasons": estimate.get("reasons") or {},
        "matches": estimate["matches"],
        "rows": estimate.get("rows") or [],
    }
    if pcd_points is not None:
        out["point_count"] = int(len(np.asarray(pcd_points)))
    return out


def _score_none(reason: str, attempts: int, pair_count: int) -> dict[str, Any]:
    return {
        "ocs": None,
        "valid_rate": None,
        "valid_count": 0,
        "conditional_success_rate": None,
        "conditional_stability_rate": None,
        "attempts": int(attempts),
        "success_count": 0,
        "invalid_count": int(attempts),
        "per_pair": [{"identity": i, "success_rate": None, "valid_rate": None, "conditional_success_rate": None} for i in range(pair_count)],
        "reason": reason,
    }


def _empty_result(reason: str, params: dict[str, Any], pair_count: int) -> dict[str, Any]:
    return {
        "status": reason,
        "params": _public_params(params),
        "fixed_omega": [],
        "pairs": [],
        "pointcloud": {},
        "satellite": {},
        "raw": _score_none(reason, 0, pair_count),
        "refined": _score_none(reason, 0, pair_count),
        "mlsd": {},
        "calibration": {"confidence": UNCALIBRATED_CONFIDENCE},
        "warnings": [reason],
        "errors": [],
    }


def _samples_from_payload(
    value: Any,
    point: np.ndarray | None,
    covariances: np.ndarray | None,
    n_mc: int,
    rng_seed: int,
    shared_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if value is not None:
        samples = _samples(value, len(point) if point is not None else None, "samples")
        return _resample(samples, n_mc, rng_seed, shared_indices), "empirical"
    if point is None:
        return np.empty((0, 0, 2), dtype=float), "missing"
    if covariances is not None:
        rng = np.random.default_rng(rng_seed)
        samples = np.empty((n_mc, len(point), 2), dtype=float)
        for k in range(n_mc):
            for i, (mean, cov) in enumerate(zip(point, covariances)):
                samples[k, i] = rng.multivariate_normal(mean, cov)
        return samples, "assumed_gaussian_covariance"
    return point.reshape(1, len(point), 2), "deterministic"


def _resample(samples: np.ndarray, n_mc: int, rng_seed: int, shared_indices: np.ndarray | None = None) -> np.ndarray:
    if len(samples) == 0 or len(samples) == n_mc:
        return samples
    if shared_indices is not None:
        return np.ascontiguousarray(samples[shared_indices % len(samples)])
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(samples), n_mc)
    return np.ascontiguousarray(samples[indices])


def _resample_indices(count: int, n_mc: int, rng_seed: int, shared_indices: np.ndarray | None) -> np.ndarray:
    """Mirror legacy resampling exactly, retaining whole-round correspondence."""
    if count == 0 or count == n_mc:
        return np.arange(count, dtype=int)
    if shared_indices is not None:
        return shared_indices % count
    return np.random.default_rng(rng_seed).integers(0, count, n_mc)


def _samples(value: Any, count: int | None, name: str) -> np.ndarray:
    if isinstance(value, (list, tuple)) and value and any(np.asarray(item).ndim == 2 for item in value):
        sample_count = len(value)
        inferred = count
        if inferred is None:
            inferred = max(np.asarray(item).shape[0] for item in value)
        out = np.full((sample_count, inferred, 2), np.nan, dtype=float)
        for i, item in enumerate(value):
            arr_i = np.asarray(item, dtype=float)
            if arr_i.ndim == 2 and arr_i.shape[1] >= 2 and arr_i.shape[0] == inferred:
                out[i] = arr_i[:, :2]
        return out
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError(f"{name} must be KxNx2 or Nx2")
    arr = arr[:, :, :2]
    if count is not None and arr.shape[1] != count:
        raise ValueError(f"{name} must contain {count} identities")
    return np.ascontiguousarray(arr)


def _samples_or_base(value: Any | None, base: np.ndarray, name: str) -> np.ndarray:
    if value is None:
        return base.reshape(1, len(base), 2)
    return _samples(value, len(base), name)


def _xy_or_none(value: Any, count: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, list) and any(item is None for item in value):
        out = np.full((count, 2), np.nan, dtype=float)
        for i, item in enumerate(value):
            if item is not None:
                out[i] = np.asarray(item, dtype=float)[:2]
        return out
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("auto xy values must be an Nx2+ array")
    arr = np.ascontiguousarray(arr[:, :2])
    if len(arr) != count:
        raise ValueError(f"expected {count} identities")
    return arr


def _covariances_or_none(value: Any, count: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (count, 2, 2):
        raise ValueError("covariances must be an Nx2x2 array")
    return arr


def _check_design(design: np.ndarray, rank: int, name: str) -> None:
    actual_rank = int(np.linalg.matrix_rank(design))
    if actual_rank < rank:
        raise ValueError(f"{name} design is rank deficient ({actual_rank} < {rank})")
    condition = float(np.linalg.cond(design))
    if not math.isfinite(condition) or condition > 1e12:
        raise ValueError(f"{name} design is ill-conditioned ({condition:.3g})")


def _normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    shifted = points - center
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    if mean_distance <= 1e-12 or not math.isfinite(mean_distance):
        raise ValueError("homography points are degenerate")
    scale = math.sqrt(2.0) / mean_distance
    transform = np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0],
    ])
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = homogeneous @ transform.T
    return normalized[:, :2], transform


def _combined_correlation(pointcloud: dict[str, Any], satellite: dict[str, Any]) -> str:
    pc = pointcloud.get("correlation_model")
    sat = satellite.get("correlation_model")
    if pc == sat == "joint_empirical":
        return "joint_empirical"
    return "assumed_independent"


def _sample_invalid(sample: np.ndarray) -> bool:
    return bool(sample.ndim != 2 or sample.shape[1] != 2 or not np.all(np.isfinite(sample)))


def _invalid_sample_count(samples: np.ndarray) -> int:
    return int(sum(_sample_invalid(sample) for sample in samples))


def _array_or_none(value: np.ndarray | None) -> list[Any] | None:
    return None if value is None else value.tolist()


def _row_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return arr[:2].tolist()


def _offset_or_none(manual: np.ndarray, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    delta = arr[:2] - manual[:2]
    return {"xy": delta.tolist(), "norm": float(np.linalg.norm(delta))}


def _sample_count_for_pair(samples: np.ndarray, index: int) -> int:
    if len(samples) == 0:
        return 0
    return int(np.count_nonzero(np.all(np.isfinite(samples[:, index, :]), axis=1)))


def _satellite_tolerance_px(
    satellite_tolerance_px: float | None,
    satellite_tolerance_m: float | None,
    gsd_m_per_px: float | None,
) -> float:
    if satellite_tolerance_px is not None:
        return float(satellite_tolerance_px)
    if satellite_tolerance_m is None or gsd_m_per_px is None:
        raise ValueError("provide satellite_tolerance_px or satellite_tolerance_m with gsd_m_per_px")
    if gsd_m_per_px <= 0:
        raise ValueError("gsd_m_per_px must be positive")
    return float(satellite_tolerance_m / gsd_m_per_px)


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _cancelled(should_cancel: Callable[[], bool] | None) -> bool:
    return bool(should_cancel is not None and should_cancel())


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _method_polygons(data: Any, method: str) -> list[dict[str, Any]]:
    if isinstance(data, (str, Path)):
        data = json.loads(Path(data).read_text(encoding="utf-8"))
    matches = [node for node in _walk(data) if isinstance(node, dict) and node.get("method") == method]
    polygons: list[dict[str, Any]] = []
    for match in matches:
        polygons.extend(item for item in _walk(match) if isinstance(item, dict) and "exterior_xy" in item)
    return polygons


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _clean_ring(raw_ring: Any) -> list[tuple[float, float]]:
    points = [_xy(raw_point) for raw_point in raw_ring]
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else []


def _filter_collinear(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    count = len(points)
    for index, point in enumerate(points):
        prev_point = points[(index - 1) % count]
        next_point = points[(index + 1) % count]
        if not _nearly_collinear(prev_point, point, next_point, tolerance):
            kept.append(point)
    return kept if len(kept) >= 3 else []


def _nearly_collinear(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], tolerance: float) -> bool:
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


def _xy(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("xy value must be a pair")
    x = float(value[0])
    y = float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("xy value must be finite")
    return (x, y)
