"""Dialog components."""

from __future__ import annotations

import os
import json
import csv
from typing import Optional
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from mlsd_ocs_core.gui.helpers import (
    load_point_cloud,
    load_image,
    compute_pcd_transform,
    apply_pcd_transform,
    color_points_by_height,
)
from mlsd_ocs_core import compute_mlsd, run_ocs_alignment
from mlsd_ocs_core.ocs import extract_framefield_corners
from mlsd_ocs_core.gui.residuals import compute_corner_residuals, extract_model_corners


class OcsWorker(QThread):
    result_ready = pyqtSignal(object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        pcd_points: np.ndarray,
        picked_pcd_xy: np.ndarray,
        picked_sat_xy: np.ndarray,
        frame_field_polygons: Optional[dict],
        params: dict,
        generation: int,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self.pcd_points = pcd_points
        self.picked_pcd_xy = picked_pcd_xy
        self.picked_sat_xy = picked_sat_xy
        self.frame_field_polygons = frame_field_polygons
        self.params = params
        self.generation = generation

    def stop(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        try:
            result = run_ocs_alignment(
                self.pcd_points,
                self.picked_pcd_xy,
                self.picked_sat_xy,
                self.frame_field_polygons,
                self.params,
                progress=self.progress.emit,
                should_cancel=self.isInterruptionRequested,
            )
            result["_generation"] = self.generation
            if not self.isInterruptionRequested():
                self.result_ready.emit(result)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.error.emit(str(exc))


class PcdAlignDialog(QDialog):
    """Dialog for aligning point clouds with satellite imagery."""
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MLSD / OCS evaluation")
        self.resize(1400, 800)
        
        # Data
        self.pcd_points: Optional[np.ndarray] = None
        self.sat_image: Optional[np.ndarray] = None
        self.building_outline: Optional[np.ndarray] = None
        self.building_polygons: Optional[dict] = None
        self.corner_residuals: list[dict] = []
        self._ocs_worker: Optional[OcsWorker] = None
        self._ocs_generation = 0
        self._close_pending = False
        self.ocs_result: Optional[dict] = None
        self.picked_sat: list[list[float]] = []  # [(u, v), ...]
        self.picked_pcd: list[list[float]] = []  # [(x, y), ...]
        self.transform_matrix: Optional[np.ndarray] = None
        self.method = "affine"  # "affine" or "homo"
        self.click_state = "sat"  # "sat" or "pcd"
        self.mlsd_value: Optional[float] = None
        # Store point cloud display data (for redrawing)
        self.pcd_xy_display: Optional[np.ndarray] = None
        self.pcd_colors_display: Optional[np.ndarray] = None
        self.pcd_yaxis_inverted = False  # Whether y-axis should be inverted (default: not inverted)
        # Cached cropped overlay image
        self.overlay_cropped: Optional[np.ndarray] = None
        self.overlay_crop_bounds: Optional[tuple] = None  # (u_min, v_min, u_max, v_max)
        # Cached deformation visualization data
        self.deformation_vis_data: Optional[dict] = None  # {uv, dev, H, W, max_dev}
        
        # Layout
        main_layout = QVBoxLayout()
        
        # Control bar
        ctrl_layout = QHBoxLayout()
        self.btn_load_sat = QPushButton("Load satellite image")
        self.btn_load_sat.clicked.connect(self._load_satellite)
        self.btn_load_sat.setObjectName("primary_btn")
        self.btn_load_contours = QPushButton("Load contours (JSON)")
        self.btn_load_contours.clicked.connect(self._load_building_polygons)
        self.btn_load_pcd = QPushButton("Load point cloud")
        self.btn_load_pcd.clicked.connect(self._load_pointcloud)
        self.btn_load_pcd.setObjectName("primary_btn")
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Affine transform (3+ pairs)", "Homography (4+ pairs)"])
        self.combo_method.currentIndexChanged.connect(self._change_method)
        self.btn_compute = QPushButton("Compute MLSD")
        self.btn_compute.clicked.connect(self._compute_transform)
        self.btn_compute.setObjectName("primary_btn")
        self.btn_compute_ocs = QPushButton("Compute OCS")
        self.btn_compute_ocs.clicked.connect(self._start_ocs)
        self.btn_compute_ocs.setObjectName("primary_btn")
        self.btn_clear = QPushButton("Clear point pairs")
        self.btn_clear.clicked.connect(self._clear_pairs)
        self.btn_save_points = QPushButton("Save point pairs")
        self.btn_save_points.clicked.connect(self._save_points)
        self.btn_load_points = QPushButton("Load point pairs")
        self.btn_load_points.clicked.connect(self._load_points)
        self.btn_save = QPushButton("Save results")
        self.btn_save.clicked.connect(self._save_result)
        self.label_status = QLabel("Status: please load satellite image and point cloud first")
        self.label_status.setWordWrap(True)
        self.label_mlsd = QLabel("MLSD: -")
        self.label_outline = QLabel("Building contours: -")
        self.label_outline.setWordWrap(True)
        self.label_residuals = QLabel("Corner residuals: -")
        self.label_residuals.setWordWrap(True)
        self.corner_radius = QSpinBox()
        self.corner_radius.setRange(1, 500)
        self.corner_radius.setValue(20)
        self.corner_radius.setSuffix(" px")
        self.corner_radius.valueChanged.connect(self._update_corner_residuals)
        self.tau_p_m = QDoubleSpinBox()
        self.tau_p_m.setRange(0.01, 100.0)
        self.tau_p_m.setValue(0.5)
        self.tau_p_m.setDecimals(2)
        self.tau_p_m.setSuffix(" m")
        self.tau_s_px = QDoubleSpinBox()
        self.tau_s_px.setRange(0.1, 1000.0)
        self.tau_s_px.setValue(2.0)
        self.tau_s_px.setDecimals(1)
        self.tau_s_px.setSuffix(" px")
        self.epsilon_mlsd = QDoubleSpinBox()
        self.epsilon_mlsd.setRange(0.001, 10.0)
        self.epsilon_mlsd.setValue(0.05)
        self.epsilon_mlsd.setDecimals(3)
        self.n_bootstrap = QSpinBox()
        self.n_bootstrap.setRange(1, 10000)
        self.n_bootstrap.setValue(40)
        self.n_mc = QSpinBox()
        self.n_mc.setRange(1, 100000)
        self.n_mc.setValue(200)
        self.gsd_m_per_px = QDoubleSpinBox()
        self.gsd_m_per_px.setRange(0.0, 1000.0)
        self.gsd_m_per_px.setValue(0.0)
        self.gsd_m_per_px.setDecimals(4)
        self.gsd_m_per_px.setSpecialValueText("GSD off")
        self.gsd_m_per_px.setSuffix(" m/px")
        self.identity_confirmed = QCheckBox("Identity confirmed")
        self.identity_confirmed.setChecked(False)
        for widget in (
            self.tau_p_m,
            self.tau_s_px,
            self.epsilon_mlsd,
            self.n_bootstrap,
            self.n_mc,
            self.gsd_m_per_px,
        ):
            widget.valueChanged.connect(self._invalidate_ocs_result)
        self.identity_confirmed.toggled.connect(self._invalidate_ocs_result)
        self.label_ocs = QLabel("OCS: -")
        self.label_ocs.setWordWrap(True)
        self.ocs_details = QTextEdit()
        self.ocs_details.setReadOnly(True)
        self.ocs_details.setMaximumHeight(110)
        self.ocs_details.setPlainText("OCS details: -")
        
        ctrl_layout.addWidget(self.btn_load_sat)
        ctrl_layout.addWidget(self.btn_load_pcd)
        ctrl_layout.addWidget(QLabel("Transformation method:"))
        ctrl_layout.addWidget(self.combo_method)
        ctrl_layout.addWidget(self.btn_compute)
        ctrl_layout.addWidget(self.btn_compute_ocs)
        ctrl_layout.addStretch()
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.btn_load_contours)
        file_layout.addWidget(self.btn_clear)
        file_layout.addWidget(self.btn_save_points)
        file_layout.addWidget(self.btn_load_points)
        file_layout.addWidget(self.btn_save)
        file_layout.addStretch()
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(self.label_mlsd)
        metric_layout.addWidget(QLabel("Corner search:"))
        metric_layout.addWidget(self.corner_radius)
        metric_layout.addStretch()
        ocs_param_layout = QGridLayout()
        ocs_param_layout.addWidget(QLabel("tauP:"), 0, 0)
        ocs_param_layout.addWidget(self.tau_p_m, 0, 1)
        ocs_param_layout.addWidget(QLabel("tauS:"), 0, 2)
        ocs_param_layout.addWidget(self.tau_s_px, 0, 3)
        ocs_param_layout.addWidget(QLabel("epsilon:"), 0, 4)
        ocs_param_layout.addWidget(self.epsilon_mlsd, 0, 5)
        ocs_param_layout.addWidget(QLabel("bootstrap:"), 1, 0)
        ocs_param_layout.addWidget(self.n_bootstrap, 1, 1)
        ocs_param_layout.addWidget(QLabel("MC:"), 1, 2)
        ocs_param_layout.addWidget(self.n_mc, 1, 3)
        ocs_param_layout.addWidget(QLabel("GSD:"), 2, 0)
        ocs_param_layout.addWidget(self.gsd_m_per_px, 2, 1)
        ocs_param_layout.addWidget(self.identity_confirmed, 2, 2, 1, 2)
        ocs_param_layout.setColumnStretch(6, 1)
        
        # Canvas area
        canvas_layout = QHBoxLayout()
        
        # Satellite image canvas and toolbar
        sat_widget = QWidget()
        sat_widget.setMinimumWidth(0)
        sat_widget_layout = QVBoxLayout()
        self.sat_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.sat_canvas.setMinimumWidth(0)
        self.sat_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.sat_ax = self.sat_canvas.figure.add_subplot(111)
        self.sat_ax.set_title("Satellite image (click here first)")
        self.sat_ax.set_axis_off()
        self.sat_canvas.mpl_connect("button_press_event", self._on_sat_click)
        self.sat_toolbar = NavigationToolbar(self.sat_canvas, self)
        self.sat_toolbar.setMaximumHeight(30)
        self.sat_toolbar.setMinimumWidth(0)
        self.sat_toolbar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        sat_widget_layout.addWidget(self.sat_toolbar)
        sat_widget_layout.addWidget(self.sat_canvas)
        sat_widget.setLayout(sat_widget_layout)
        
        # Top-view point cloud canvas and toolbar
        pcd_widget = QWidget()
        pcd_widget.setMinimumWidth(0)
        pcd_widget_layout = QVBoxLayout()
        self.pcd_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.pcd_canvas.setMinimumWidth(0)
        self.pcd_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.pcd_ax = self.pcd_canvas.figure.add_subplot(111)
        self.pcd_ax.set_title("Top view of point cloud (click here next)")
        # self.pcd_ax.set_aspect("equal")
        # Don't invert y-axis, keep normal point cloud top-view direction
        self.pcd_canvas.mpl_connect("button_press_event", self._on_pcd_click)
        self.pcd_toolbar = NavigationToolbar(self.pcd_canvas, self)
        self.pcd_toolbar.setMaximumHeight(30)
        self.pcd_toolbar.setMinimumWidth(0)
        self.pcd_toolbar.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        pcd_widget_layout.addWidget(self.pcd_toolbar)
        pcd_widget_layout.addWidget(self.pcd_canvas)
        pcd_widget.setLayout(pcd_widget_layout)
        
        # Overlay result canvas and title
        overlay_widget = QWidget()
        overlay_widget.setMinimumWidth(0)
        overlay_widget_layout = QVBoxLayout()
        overlay_title = QLabel("Overlay result")
        overlay_title.setMinimumWidth(0)
        overlay_title.setFixedHeight(30)
        overlay_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.overlay_canvas = FigureCanvas(Figure(figsize=(6, 6)))
        self.overlay_canvas.setMinimumWidth(0)
        self.overlay_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.overlay_ax = self.overlay_canvas.figure.add_subplot(111)
        # Don't set title in figure to avoid affecting resolution
        self.overlay_ax.set_axis_off()
        overlay_widget_layout.addWidget(overlay_title)
        overlay_widget_layout.addWidget(self.overlay_canvas)
        overlay_widget.setLayout(overlay_widget_layout)
        
        canvas_layout.addWidget(sat_widget)
        canvas_layout.addWidget(pcd_widget)
        canvas_layout.addWidget(overlay_widget)
        
        main_layout.addLayout(ctrl_layout)
        main_layout.addLayout(file_layout)
        main_layout.addWidget(self.label_status)
        main_layout.addLayout(metric_layout)
        main_layout.addLayout(ocs_param_layout)
        main_layout.addWidget(self.label_outline)
        main_layout.addWidget(self.label_residuals)
        main_layout.addWidget(self.label_ocs)
        main_layout.addWidget(self.ocs_details)
        main_layout.addLayout(canvas_layout)
        self.setLayout(main_layout)
        self.setMinimumSize(1100, 600)
    
    def _load_satellite(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select satellite image",
            "",
            "Image files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if path:
            try:
                self.sat_image = load_image(path)
                self.building_outline = None
                self.building_polygons = None
                self._invalidate_ocs_result()
                self._clear_pairs()
                self.label_outline.setText("Building contours: load contours JSON for OCS")
                self.label_outline.setToolTip("")
                self._update_corner_residuals()
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to load satellite image:\n{exc}")

    def _load_building_polygons(self) -> None:
        if self.sat_image is None:
            QMessageBox.warning(self, "Warning", "Please load satellite image first")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select building contours JSON", "", "JSON files (*.json);;All files (*.*)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8-sig") as stream:
                polygons = json.load(stream)
            residual_corners = extract_model_corners(polygons)
            corners = extract_framefield_corners(polygons)
            if not residual_corners or not corners:
                raise ValueError("Contours JSON must contain valid non-collinear corners")
            outline = np.zeros(self.sat_image.shape[:2], dtype=np.uint8)
            for corner in corners:
                start = tuple(round(value) for value in corner["corner"])
                end = tuple(round(value) for value in corner["next_corner"])
                cv2.line(outline, start, end, 255, 1)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load building contours:\n{exc}")
            return
        self.building_polygons = polygons
        self.building_outline = outline
        self._invalidate_ocs_result()
        self.label_outline.setText("Building contours: ready (red)")
        self.label_outline.setToolTip(path)
        self._update_corner_residuals()

    def _update_corner_residuals(self) -> None:
        self.corner_residuals = []
        if self.transform_matrix is None or not self.picked_sat:
            self.label_residuals.setText("Corner residuals: -")
        elif self.building_polygons is None:
            self.label_residuals.setText("Corner residuals: load contours JSON")
        else:
            try:
                self.corner_residuals = compute_corner_residuals(
                    self.building_polygons,
                    self.picked_sat[:len(self.picked_pcd)],
                    radius_px=self.corner_radius.value(),
                )
                matched = [row for row in self.corner_residuals if row["matched"]]
                details = "; ".join(
                    f"#{row['point_index'] + 1}: "
                    + (f"{row['residual_px']:.1f} px (dx={row['dx']:+.1f}, dy={row['dy']:+.1f})"
                       if row["matched"] else "no corner")
                    for row in self.corner_residuals
                )
                summary = f"Corner residuals: {len(matched)}/{len(self.corner_residuals)} matched"
                if matched:
                    distances = [row["residual_px"] for row in matched]
                    summary += f", mean {np.mean(distances):.1f} px, max {max(distances):.1f} px"
                self.label_residuals.setText(f"{summary}\n{details}")
            except Exception as exc:
                self.label_residuals.setText(f"Corner residuals: failed ({exc})")
        self._redraw_satellite()

    def closeEvent(self, event) -> None:
        running = False
        if self._ocs_worker is not None and self._ocs_worker.isRunning():
            self._ocs_worker.stop()
            running = True
        if running:
            self._close_pending = True
            event.ignore()
            return
        super().closeEvent(event)

    def _invalidate_ocs_result(self, *args) -> None:
        self._ocs_generation += 1
        if self._ocs_worker is not None and self._ocs_worker.isRunning():
            self._ocs_worker.stop()
        self.ocs_result = None
        if hasattr(self, "label_ocs"):
            self.label_ocs.setText("OCS: stale or not computed")
        if hasattr(self, "ocs_details"):
            self.ocs_details.setPlainText("OCS details: -")
        if hasattr(self, "sat_canvas"):
            self._redraw_satellite()
        if hasattr(self, "pcd_canvas"):
            self._redraw_pointcloud()

    def _ocs_params(self) -> dict:
        gsd = self.gsd_m_per_px.value()
        return {
            "tau_p_m": self.tau_p_m.value(),
            "tau_s_px": self.tau_s_px.value(),
            "epsilon_mlsd": self.epsilon_mlsd.value(),
            "n_bootstrap": self.n_bootstrap.value(),
            "n_mc": self.n_mc.value(),
            "gsd_m_per_px": gsd if gsd > 0 else None,
            "identity_confirmed": self.identity_confirmed.isChecked(),
            "method": self.method,
        }

    def _start_ocs(self) -> None:
        if self._ocs_worker is not None and self._ocs_worker.isRunning():
            QMessageBox.warning(self, "Info", "OCS is already running")
            return
        if self.pcd_points is None:
            QMessageBox.warning(self, "Warning", "Please load point cloud first")
            return
        if self.building_polygons is None:
            QMessageBox.warning(self, "Warning", "Please load building contours JSON first")
            return
        if len(self.picked_sat) != len(self.picked_pcd):
            QMessageBox.warning(self, "Warning", "Please finish the current point pair first")
            return
        if len(self.picked_sat) < 3:
            QMessageBox.warning(self, "Warning", "At least 3 point pairs are required")
            return
        if self.method == "homo" and len(self.picked_sat) < 4:
            QMessageBox.warning(self, "Warning", "Homography requires at least 4 point pairs")
            return

        self._invalidate_ocs_result()
        self.btn_compute_ocs.setEnabled(False)
        self.btn_compute_ocs.setText("OCS running...")
        self.label_ocs.setText("OCS: running")
        self.ocs_details.setPlainText("OCS details: starting...")

        worker = OcsWorker(
            self.pcd_points,
            np.asarray(self.picked_pcd, dtype=np.float64),
            np.asarray(self.picked_sat, dtype=np.float64),
            self.building_polygons,
            self._ocs_params(),
            self._ocs_generation,
            self,
        )
        self._ocs_worker = worker
        worker.progress.connect(self._on_ocs_progress)
        worker.result_ready.connect(self._on_ocs_result)
        worker.error.connect(self._on_ocs_error)
        worker.finished.connect(self._on_ocs_finished)
        worker.start()

    def _on_ocs_progress(self, message: str) -> None:
        current = self.ocs_details.toPlainText()
        self.ocs_details.setPlainText((current + "\n" + message).strip())

    def _on_ocs_result(self, result: dict) -> None:
        if result.get("_generation") != self._ocs_generation:
            return
        self.ocs_result = result
        self._show_ocs_summary(result)
        self._redraw_satellite()
        self._redraw_pointcloud()

    def _on_ocs_error(self, message: str) -> None:
        self.ocs_result = None
        self.label_ocs.setText(f"OCS: failed ({message})")
        self.ocs_details.setPlainText(message)

    def _on_ocs_finished(self) -> None:
        self.btn_compute_ocs.setEnabled(True)
        self.btn_compute_ocs.setText("Compute OCS")
        if self._ocs_worker is not None:
            self._ocs_worker.deleteLater()
            self._ocs_worker = None
        self._finish_pending_close()

    def _finish_pending_close(self) -> None:
        ocs_running = self._ocs_worker is not None and self._ocs_worker.isRunning()
        if self._close_pending and not ocs_running:
            self._close_pending = False
            self.close()

    def _show_ocs_summary(self, result: dict) -> None:
        raw_ocs = self._nested(result, ("raw", "ocs"))
        refined_ocs = self._nested(result, ("refined", "ocs"))
        raw_all = self._mlsd_item(result, "raw_all")
        raw_common = self._mlsd_item(result, "raw_common")
        refined_item = self._mlsd_item(result, "refined")
        raw_mlsd = self._mlsd_value(raw_all)
        raw_common_mlsd = self._mlsd_value(raw_common)
        refined_mlsd = self._mlsd_value(refined_item)
        scenarios = self._candidate_scenarios(result)
        refined_range = self._candidate_range(result)
        status = result.get("status", "-")
        warnings = result.get("warnings") or []
        valid_rate = self._nested(result, ("refined", "valid_rate"))
        ocs_map = self._nested(result, ("mlsd", "ocs_map"))
        refined_text = self._refined_mlsd_text(refined_mlsd, refined_range, refined_item)
        mlsd_state = self._mlsd_availability(refined_item)

        self.label_ocs.setText(
            "OCS: "
            f"status={status}, raw={self._fmt(raw_ocs)}, refined={self._fmt(refined_ocs)}, "
            f"MLSD={mlsd_state}, raw_all={self._fmt(raw_mlsd)}, raw_S={self._fmt(raw_common_mlsd)}, refined={refined_text}, "
            f"valid={self._fmt(valid_rate)}"
        )

        identity = "confirmed" if self.identity_confirmed.isChecked() else "unconfirmed"
        coverage = self._coverage_summary(result)
        lines = [
            "Confidence: uncalibrated geometry confidence / empirical support",
            f"Identity check: {identity}; roof/wall-foot same-corner identity is not assumed",
            f"Warnings: {', '.join(warnings) if warnings else '-'}",
            f"Local OCS status: raw={self._fmt(raw_ocs)}, refined={self._fmt(refined_ocs)}, status={status}",
            f"MLSD availability: {mlsd_state}",
            f"MLSD raw all points: {self._mlsd_text(raw_all)}",
            f"MLSD raw same subset S: {self._mlsd_text(raw_common)}",
            f"MLSD refined same subset S: {refined_text}",
            f"Candidate conditional range: {self._fmt_range(refined_range)}; not a 95% CI{self._range_scope_note(refined_item)}",
            f"Same subset S: {coverage['included']}",
            f"Coverage: {coverage['coverage']}",
            f"Excluded: {coverage['excluded']}",
            f"Candidate scenarios: {coverage['scenarios']}",
            f"OCS_map: {self._fmt(ocs_map)}",
            f"Raw valid: {self._fmt(self._nested(result, ('raw', 'valid_rate')))}, "
            f"refined valid: {self._fmt(self._nested(result, ('refined', 'valid_rate')))}, "
            f"map valid: {self._fmt(self._nested(result, ('mlsd', 'mc', 'valid_rate')))}",
        ]
        lines.extend(self._scenario_lines(scenarios))
        lines.extend(self._ocs_pair_lines(result))
        self.ocs_details.setPlainText("\n".join(lines))

    def _mlsd_item(self, result: dict, name: str):
        mlsd = result.get("mlsd") or {}
        return mlsd.get(name)

    def _mlsd_value(self, item):
        if isinstance(item, dict):
            for key in ("value", "mlsd", "refined_mlsd", "raw_mlsd"):
                if item.get(key) is not None:
                    return item[key]
            return None
        return item

    def _mlsd_text(self, item) -> str:
        value = self._mlsd_value(item)
        if value is not None:
            return self._fmt(value)
        reason = item.get("reason") if isinstance(item, dict) else None
        return f"not evaluable ({reason or '-'})"

    def _refined_mlsd_text(self, value, value_range, item) -> str:
        status = item.get("status") if isinstance(item, dict) else None
        if status == "ambiguous" and value_range is not None:
            return f"candidate mean {self._fmt(value)}, range {self._fmt_range(value_range)}"
        if value is not None:
            return self._fmt(value)
        reason = item.get("reason") if isinstance(item, dict) else None
        return f"not evaluable ({reason or '-'})"

    def _mlsd_availability(self, item) -> str:
        status = item.get("status") if isinstance(item, dict) else None
        if status == "ambiguous":
            return "candidate conditional range"
        if status == "unique" or self._mlsd_value(item) is not None:
            return "unique value"
        return "not evaluable"

    def _candidate_scenarios(self, result: dict) -> list[dict]:
        value = self._nested(result, ("mlsd", "refined", "candidate_results"))
        return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []

    def _candidate_range(self, result: dict):
        value = self._nested(result, ("mlsd", "refined", "candidate_range"))
        return [value[0], value[1]] if isinstance(value, (list, tuple)) and len(value) >= 2 else None

    def _range_scope_note(self, refined_item) -> str:
        if not isinstance(refined_item, dict) or not refined_item.get("truncated"):
            return ""
        reason = refined_item.get("truncated_reason") or "candidate_combo_cap_reached_partial_range"
        return f"; partial range, cap truncated ({reason})"

    def _coverage_summary(self, result: dict) -> dict[str, str]:
        mlsd = result.get("mlsd") or {}
        availability = mlsd.get("availability") or {}
        refined = mlsd.get("refined") or {}
        included = availability.get("identities")
        excluded = availability.get("excluded") or []
        total = len(result.get("pairs") or [])
        if total is None:
            total = len(result.get("pairs") or [])
        included_count = len(included) if isinstance(included, (list, tuple)) else None
        area = availability.get("coverage")
        coverage_text = f"{included_count if included_count is not None else '-'}/{total}"
        if area is not None:
            coverage_text += f", hull_area_ratio={self._fmt(area)}"
        excluded_text = ", ".join(self._exclusion_text(row) for row in excluded) if isinstance(excluded, list) else "-"
        return {
            "included": self._cell(included) if included is not None else "-",
            "coverage": coverage_text,
            "excluded": excluded_text or "-",
            "scenarios": (
                f"total={refined.get('scenario_count', '-')}, "
                f"evaluated={refined.get('evaluated_count', '-')}, "
                f"valid={refined.get('valid_count', '-')}, "
                f"truncated={bool(refined.get('truncated'))}"
            ),
        }

    def _exclusion_text(self, row) -> str:
        if isinstance(row, dict):
            idx = self._first_present(row, "id", "identity", "index", "point_index")
            reason = self._first_present(row, "reason", "reasons", "status")
            return f"#{idx}:{reason}"
        return str(row)

    def _scenario_lines(self, scenarios: list[dict]) -> list[str]:
        lines = []
        for idx, row in enumerate(scenarios[:6], start=1):
            identities = [item.get("identity") for item in row.get("candidates", []) if isinstance(item, dict)]
            lines.append(
                f"Scenario {idx}: status={row.get('status', '-')}, "
                f"refined_S={self._fmt(row.get('value'))}, S={self._cell(identities)}"
            )
        if len(scenarios) > 6:
            lines.append(f"Scenario rows shown: 6/{len(scenarios)}")
        return lines

    def _fmt_range(self, value) -> str:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return "-"
        return f"[{self._fmt(value[0])}, {self._fmt(value[1])}]"

    def _ocs_pair_lines(self, result: dict) -> list[str]:
        lines: list[str] = []
        pair_rows = result.get("pairs") or []
        gsd = self._ocs_params().get("gsd_m_per_px")
        for index, pair in enumerate(pair_rows):
            offsets = pair.get("offsets", {})
            raw_score = self._score_row(result, "raw", index)
            refined_score = self._score_row(result, "refined", index)
            diagnostics = pair.get("diagnostics", {})
            pcd_diag = diagnostics.get("pointcloud") or {}
            sat_diag = diagnostics.get("satellite") or {}
            pcd_offset = self._nested(offsets, ("refined_pcd_m", "xy")) or self._nested(offsets, ("raw_pcd_m", "xy"))
            sat_offset = self._nested(offsets, ("refined_sat_px", "xy")) or self._nested(offsets, ("raw_sat_px", "xy"))
            line = (
                f"#{index + 1}: pcd offset {self._fmt_xy(pcd_offset)} m, "
                f"sat offset {self._fmt_xy(sat_offset)} px"
            )
            if gsd is not None:
                norm = None
                if isinstance(sat_offset, (list, tuple)) and len(sat_offset) >= 2:
                    norm = float(np.linalg.norm(np.asarray(sat_offset, dtype=float)) * gsd)
                line += f" / {self._fmt(norm)} m"
            line += (
                f", status={pair.get('status', '-')}, "
                f"raw={self._fmt(raw_score.get('success_rate'))}/{self._fmt(raw_score.get('valid_rate'))}, "
                f"refined={self._fmt(refined_score.get('success_rate'))}/{self._fmt(refined_score.get('valid_rate'))}, "
                f"samples={pair.get('samples', {})}, "
                f"reason={pcd_diag.get('reason') or sat_diag.get('reason') or ','.join(pair.get('reasons') or []) or '-'}"
            )
            lines.append(line)
        return lines

    def _score_row(self, result: dict, branch: str, index: int) -> dict:
        rows = self._nested(result, (branch, "per_pair")) or []
        if index < len(rows):
            return rows[index]
        return {}

    def _ocs_auto_xy(self, row: dict, frame: str) -> Optional[list[float]]:
        keys = (
            ("refined_xy_m", "auto_xy_m", "raw_xy_m")
            if frame == "pcd"
            else ("refined_xy_px", "auto_xy_px", "raw_xy_px", "corner")
        )
        value = self._first_present(row, *keys)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
        return None

    def _nested(self, data: dict, path: tuple[str, ...]):
        value = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    def _first_present(self, data: dict, *keys: str):
        for key in keys:
            if isinstance(data, dict) and data.get(key) is not None:
                return data[key]
        return None

    def _fmt(self, value) -> str:
        if value is None:
            return "-"
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_xy(self, value) -> str:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return "-"
        return f"({self._fmt(value[0])}, {self._fmt(value[1])})"

    def _draw_candidates(self, ax, row: dict, frame: str, row_index: int) -> None:
        for cand_index, candidate in enumerate(self._row_candidates(row, frame), start=1):
            xy = self._candidate_xy(candidate, frame)
            if xy is None:
                continue
            ax.scatter(
                xy[0], xy[1], s=72, facecolors="none", edgecolors="orange",
                marker="o", linewidths=1.5, zorder=12
            )
            ax.text(
                xy[0], xy[1], f"{row_index + 1}.{cand_index}",
                color="orange", fontsize=8, zorder=13
            )

    def _row_candidates(self, row: dict, frame: str) -> list[dict]:
        identity = row.get("identity")
        if identity is None:
            return []
        key = "pointcloud_options" if frame == "pointcloud" else "satellite_options"
        options = self._nested(self.ocs_result or {}, ("mlsd", "availability", key)) or []
        if not isinstance(options, list) or int(identity) >= len(options):
            return []
        rows = options[int(identity)]
        return [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []

    def _candidate_xy(self, candidate: dict, frame: str) -> Optional[list[float]]:
        if frame == "satellite":
            keys = ("sat_xy_px", "xy")
        else:
            keys = ("pcd_xy_m", "xy")
        value = self._first_present(candidate, *keys)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
        return None
    
    def _redraw_satellite(self) -> None:
        """Redraw satellite image and picked points."""
        # Save current view limits
        if self.sat_image is not None and len(self.sat_ax.get_images()) > 0:
            xlim = self.sat_ax.get_xlim()
            ylim = self.sat_ax.get_ylim()
        else:
            xlim = ylim = None
        
        self.sat_ax.clear()
        if self.sat_image is not None:
            if self.building_outline is None:
                self.sat_ax.imshow(self.sat_image)
            else:
                display = self.sat_image.copy()
                visible_outline = cv2.dilate(self.building_outline, np.ones((5, 5), np.uint8)) > 0
                display[visible_outline] = (255, 0, 0)
                self.sat_ax.imshow(display)
            # Restore view limits
            if xlim is not None and ylim is not None:
                self.sat_ax.set_xlim(xlim)
                self.sat_ax.set_ylim(ylim)
        self.sat_ax.set_title("Satellite image (click here first)")
        self.sat_ax.set_axis_off()
        # Redraw picked points
        if self.picked_sat:
            sat_pts = np.array(self.picked_sat)
            self.sat_ax.scatter(sat_pts[:, 0], sat_pts[:, 1], s=100, c="yellow", 
                              marker="x", linewidths=2, zorder=10)
        if self.ocs_result is not None:
            for index, row in enumerate(self.ocs_result.get("pairs", [])):
                point = self._nested(row, ("refined", "sat_xy_px")) or self._nested(row, ("raw", "sat_xy_px"))
                manual = self._nested(row, ("manual", "sat_xy_px"))
                self._draw_candidates(self.sat_ax, row, "satellite", index)
                if point is None:
                    continue
                self.sat_ax.scatter(point[0], point[1], s=60, c="lime",
                                    marker="+", linewidths=2, zorder=13)
                if isinstance(manual, (list, tuple)) and len(manual) >= 2:
                    self.sat_ax.plot([manual[0], point[0]], [manual[1], point[1]],
                                     color="lime", linewidth=1.2, zorder=12)
        for row in self.corner_residuals:
            if row["matched"]:
                point, corner = row["point"], row["corner"]
                self.sat_ax.plot([point[0], corner[0]], [point[1], corner[1]],
                                 color="cyan", linewidth=1.5, zorder=11)
                self.sat_ax.scatter(corner[0], corner[1], s=55, c="cyan",
                                    marker="o", edgecolors="black", zorder=12)
        self.sat_canvas.draw()
    
    def _load_pointcloud(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select point cloud file", "", 
            "Point cloud files (*.pcd *.ply *.npy *.npz);;All files (*.*)"
        )
        if path:
            try:
                points = load_point_cloud(path)
                if points.shape[1] > 3:
                    points = points[:, :3]
                self.pcd_points = points
                self._invalidate_ocs_result()
                
                # Extract XY and Z
                xy = points[:, :2]
                z = points[:, 2]
                colors = color_points_by_height(z)
                
                # Limit displayed points
                max_pts = 200000
                if len(xy) > max_pts:
                    idx = np.random.choice(len(xy), max_pts, replace=False)
                    xy = xy[idx]
                    colors = colors[idx]
                
                # Save display data
                self.pcd_xy_display = xy
                self.pcd_colors_display = colors
                
                self._redraw_pointcloud(reset_view=True)  # Reset view when loading new point cloud
                self._update_status()
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to load point cloud:\n{exc}")
    
    def _redraw_pointcloud(self, reset_view: bool = False) -> None:
        """Redraw point cloud and markers
        
        Args:
            reset_view: If True, reset view range to fit new data; if False, keep current view range
        """
        # Save current view range (only when not resetting)
        if not reset_view:
            if len(self.pcd_ax.collections) > 0 or len(self.pcd_ax.patches) > 0:
                xlim = self.pcd_ax.get_xlim()
                ylim = self.pcd_ax.get_ylim()
            else:
                xlim = ylim = None
        else:
            xlim = ylim = None
        
        self.pcd_ax.clear()
        if self.pcd_xy_display is not None and self.pcd_colors_display is not None:
            self.pcd_ax.scatter(self.pcd_xy_display[:, 0], self.pcd_xy_display[:, 1], 
                              s=1, c=self.pcd_colors_display / 255.0, marker=".")
        self.pcd_ax.set_aspect("equal")
        self.pcd_ax.set_title("Point cloud top-view (click here again)")
        # Restore y-axis inversion state (using flag to ensure consistency)
        if self.pcd_yaxis_inverted:
            self.pcd_ax.invert_yaxis()
        # Restore view range (must be after setting aspect and invert)
        # If reset_view is True, let matplotlib automatically adjust view range
        if not reset_view and xlim is not None and ylim is not None:
            self.pcd_ax.set_xlim(xlim)
            self.pcd_ax.set_ylim(ylim)
        # Redraw markers
        if self.picked_pcd:
            pcd_pts = np.array(self.picked_pcd)
            self.pcd_ax.scatter(pcd_pts[:, 0], pcd_pts[:, 1], s=100, c="yellow",
                              marker="x", linewidths=2, zorder=10)
        if self.ocs_result is not None:
            for index, row in enumerate(self.ocs_result.get("pairs", [])):
                point = self._nested(row, ("refined", "pcd_xy_m")) or self._nested(row, ("raw", "pcd_xy_m"))
                manual = self._nested(row, ("manual", "pcd_xy_m"))
                self._draw_candidates(self.pcd_ax, row, "pointcloud", index)
                if point is None:
                    continue
                self.pcd_ax.scatter(point[0], point[1], s=60, c="lime",
                                    marker="+", linewidths=2, zorder=11)
                if isinstance(manual, (list, tuple)) and len(manual) >= 2:
                    self.pcd_ax.plot([manual[0], point[0]], [manual[1], point[1]],
                                     color="lime", linewidth=1.2, zorder=10)
        self.pcd_canvas.draw()
    
    def _change_method(self, index: int) -> None:
        self.method = "homo" if index == 1 else "affine"
        self._invalidate_ocs_result()
        self._update_status()
    
    def _on_sat_click(self, event) -> None:
        if event.inaxes != self.sat_ax or self.sat_image is None:
            return
        # Check if toolbar is in zoom/pan mode (to avoid accidental clicks)
        if self.sat_toolbar.mode != '':
            return
        # Only handle left-click
        if event.button != 1:
            return
        if self.click_state == "sat":
            u, v = event.xdata, event.ydata
            if u is None or v is None:  # Skip if coordinates are invalid
                return
            self.picked_sat.append([u, v])
            self.click_state = "pcd"
            self._update_markers()
    
    def _on_pcd_click(self, event) -> None:
        if event.inaxes != self.pcd_ax or self.pcd_points is None:
            return
        # Check if toolbar is in zoom/pan mode (to avoid accidental clicks)
        if self.pcd_toolbar.mode != '':
            return
        # Only handle left-click
        if event.button != 1:
            return
        if self.click_state == "pcd":
            x, y = event.xdata, event.ydata
            if x is None or y is None:  # Skip if coordinates are invalid
                return
            self.picked_pcd.append([x, y])
            self.click_state = "sat"
            self._update_markers()
    
    def _update_markers(self) -> None:
        self._invalidate_ocs_result()
        self.corner_residuals = []
        self.label_residuals.setText("Corner residuals: recalculate MLSD to refresh")
        # Redraw satellite image and markers
        self._redraw_satellite()
        # Redraw point cloud and markers
        self._redraw_pointcloud()
        self._update_status()
    
    def _update_status(self) -> None:
        status = f"Status: {len(self.picked_sat)} point pairs selected | Method: {self.method}"
        if self.sat_image is None:
            status += " | Please load satellite image"
        if self.pcd_points is None:
            status += " | Please load point cloud"
        if self.click_state == "pcd":
            status += " | Please click corresponding points on the point cloud plot"
        else:
            status += " | Please click on the satellite image"
        self.label_status.setText(status)
    
    def _clear_pairs(self) -> None:
        self.picked_sat.clear()
        self.picked_pcd.clear()
        self.click_state = "sat"
        self.transform_matrix = None
        self.mlsd_value = None
        self._invalidate_ocs_result()
        self.label_mlsd.setText("MLSD: -")
        self.corner_residuals = []
        self.label_residuals.setText("Corner residuals: -")
        # Reset clipping related variables
        self.overlay_cropped = None
        self.overlay_crop_bounds = None
        # Redraw
        self._redraw_satellite()
        self._redraw_pointcloud()
        self.overlay_ax.clear()
        # Don't set title in figure, title is displayed in GUI
        self.overlay_ax.set_axis_off()
        self.overlay_canvas.draw()
        self._update_status()
    
    def _save_points(self) -> None:
        """Save point pairs to JSON file"""
        if len(self.picked_sat) == 0:
            QMessageBox.warning(self, "Warning", "No point pairs to save")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save point pairs", "", 
            "JSON files (*.json);;All files (*.*)"
        )
        if not path:
            return
        
        try:
            data = {
                "picked_sat": self.picked_sat,
                "picked_pcd": self.picked_pcd,
                "method": self.method,
                "num_pairs": len(self.picked_sat)
            }
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(
                self, "Save successful", 
                f"Point pairs saved to:\n{path}\n\nTotal {len(self.picked_sat)} point pairs"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save point pairs:\n{exc}")
    
    def _load_points(self) -> None:
        """Load point pairs from JSON file"""
        if self.sat_image is None or self.pcd_points is None:
            QMessageBox.warning(
                self, "Warning", 
                "Please load satellite image and point cloud first, then import point pairs"
            )
            return
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Import point pairs", "", 
            "JSON files (*.json);;All files (*.*)"
        )
        if not path:
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validate data format
            if not isinstance(data, dict):
                raise ValueError("JSON file format error: root object must be a dictionary")
            
            if "picked_sat" not in data or "picked_pcd" not in data:
                raise ValueError("JSON file format error: missing required fields")
            
            picked_sat = data["picked_sat"]
            picked_pcd = data["picked_pcd"]
            
            if not isinstance(picked_sat, list) or not isinstance(picked_pcd, list):
                raise ValueError("JSON file format error: point pairs must be a list")
            
            if len(picked_sat) != len(picked_pcd):
                raise ValueError("JSON file format error: satellite image points and point cloud points do not match")
            
            # Validate each point pair format
            for i, (sat_pt, pcd_pt) in enumerate(zip(picked_sat, picked_pcd)):
                if not isinstance(sat_pt, list) or not isinstance(pcd_pt, list):
                    raise ValueError(f"Point pair {i+1} format error: must be a list")
                if len(sat_pt) != 2 or len(pcd_pt) != 2:
                    raise ValueError(f"Point pair {i+1} format error: must contain 2 coordinate values")
            
            # Import point pairs
            self.picked_sat = picked_sat
            self.picked_pcd = picked_pcd
            
            # Import method (if exists)
            if "method" in data:
                method = data["method"]
                if method == "homo":
                    self.combo_method.setCurrentIndex(1)
                else:
                    self.combo_method.setCurrentIndex(0)
                self.method = method
            
            # Reset transformation matrix (needs to be recalculated)
            self.transform_matrix = None
            self.mlsd_value = None
            self.label_mlsd.setText("MLSD: -")
            self.overlay_cropped = None
            self.overlay_crop_bounds = None
            
            # Update display
            self.click_state = "sat"  # 重置点击状态
            self._update_markers()
            
            QMessageBox.information(
                self, "Import successful", 
                f"Successfully imported {len(picked_sat)} point pairs\n\n"
                f"Method: {self.method}\n"
                "Continue selecting point pairs, then click 'Compute MLSD' when ready."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to import point pairs:\n{exc}")
    
    def _compute_transform(self) -> None:
        if len(self.picked_sat) != len(self.picked_pcd):
            QMessageBox.warning(self, "Warning", "Please select a matching point on the point cloud plot")
            return
        if len(self.picked_sat) < 3:
            QMessageBox.warning(self, "Warning", "At least 3 point pairs are required to compute transformation")
            return
        if self.method == "homo" and len(self.picked_sat) < 4:
            QMessageBox.warning(self, "Warning", "Homography transformation requires at least 4 point pairs")
            return
        
        try:
            src_pts = np.array(self.picked_pcd, dtype=np.float32)
            dst_pts = np.array(self.picked_sat, dtype=np.float32)
            
            self.transform_matrix, inliers = compute_pcd_transform(
                src_pts, dst_pts, self.method
            )
            
            if self.transform_matrix is None:
                QMessageBox.warning(self, "Error", "Transformation calculation failed, please try selecting more dispersed points")
                return
            
            num_inliers = int(inliers.sum()) if inliers is not None else len(src_pts)
            
            # Calculate MLSD metric
            try:
                self.mlsd_value = compute_mlsd(
                    self.transform_matrix, self.method, src_pts, num_samples=1000
                )
                self.label_mlsd.setText(f"MLSD: {self.mlsd_value:.6f}")
            except Exception as exc:
                self.mlsd_value = None
                self.label_mlsd.setText(f"MLSD: calculation failed ({exc})")

            self._update_corner_residuals()
            
            QMessageBox.information(
                self, "Success", 
                f"Transformation calculation successful!\nMethod: {self.method}\nInliers: {num_inliers}/{len(src_pts)}\n"
                + (f"MLSD: {self.mlsd_value:.6f}" if self.mlsd_value is not None else "MLSD: calculation failed")
            )
            
            # Generate overlay image
            self._generate_overlay()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to compute transformation:\n{exc}")
    
    def _generate_overlay(self) -> None:
        if self.transform_matrix is None or self.pcd_points is None or self.sat_image is None:
            return
        
        try:
            xy = self.pcd_points[:, :2]
            z = self.pcd_points[:, 2]
            colors = color_points_by_height(z)
            
            # Apply transformation
            uv = apply_pcd_transform(xy, self.transform_matrix, self.method).astype(int)

            # Calculate local deformation for each point (same as MLSD: remove overall scale, only preserve anisotropy)
            deformation_values = self._compute_point_deformation(
                xy, self.transform_matrix, self.method
            )
            
            # Create overlay image
            overlay = self.sat_image.copy()
            H, W = overlay.shape[:2]
            
            # Draw point cloud (limit point count to improve performance)
            step = max(1, int(len(uv) / 500000))
            alpha = 0.4
            valid_uv = []
            valid_dev = []
            for (u, v), c, dev in zip(uv[::step], colors[::step], deformation_values[::step]):
                if 0 <= u < W and 0 <= v < H:
                    overlay[v, u] = (1 - alpha) * overlay[v, u] + alpha * c
                    valid_uv.append([u, v])
                    valid_dev.append(dev)
            
            # Calculate bounding box of point cloud coverage area (for clipping)
            if len(valid_uv) > 0:
                valid_uv = np.array(valid_uv)
                valid_dev = np.array(valid_dev)
                # Add margin (5% of image size)
                padding = int(min(H, W) * 0.05)
                u_min = max(0, int(valid_uv[:, 0].min()) - padding)
                u_max = min(W, int(valid_uv[:, 0].max()) + padding)
                v_min = max(0, int(valid_uv[:, 1].min()) - padding)
                v_max = min(H, int(valid_uv[:, 1].max()) + padding)
                
                # Crop image
                overlay_cropped = overlay[v_min:v_max, u_min:u_max]
                # Save cropped boundaries
                self.overlay_crop_bounds = (u_min, v_min, u_max, v_max)
            else:
                # If no valid points, don't crop
                overlay_cropped = overlay
                self.overlay_crop_bounds = None
            
            # Save cropped image for subsequent saving
            self.overlay_cropped = overlay_cropped
            # Save deformation visualization data (for generating deformation color map)
            if len(valid_uv) > 0 and len(valid_dev) > 0:
                self.deformation_vis_data = {
                    "uv": valid_uv,
                    "dev": valid_dev,
                    "H": H,
                    "W": W,
                    "max_dev": float(np.nanmax(valid_dev)) if np.any(np.isfinite(valid_dev)) else 0.0,
                }
            else:
                self.deformation_vis_data = None
            
            # Display overlay result
            self.overlay_ax.clear()
            self.overlay_ax.imshow(overlay_cropped)
            # Don't set title in figure, title is displayed in GUI
            self.overlay_ax.set_axis_off()
            self.overlay_canvas.draw()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to generate overlay image:\n{exc}")
    
    def _compute_point_deformation(
        self,
        points_xy: np.ndarray,
        transform_matrix: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """
        Calculate local deformation for each point (same as MLSD: remove overall scale, only preserve anisotropy)
        Return deviation array corresponding to points_xy, if cannot be calculated then return nan.
        """
        n = len(points_xy)
        if n == 0:
            return np.array([])

        dev = np.full(n, np.nan, dtype=float)

        if method == "affine":
            J = transform_matrix[:2, :2]
            try:
                J_inv = np.linalg.inv(J)
                sigma = np.linalg.svd(J_inv, compute_uv=False)
                sigma1, sigma2 = sigma[0], sigma[1]
                scale = np.sqrt(abs(sigma1 * sigma2))
                if scale > 1e-9:
                    sigma1_norm = sigma1 / scale
                    sigma2_norm = sigma2 / scale
                    deviation = max(abs(sigma1_norm - 1.0), abs(sigma2_norm - 1.0))
                else:
                    deviation = 1e6
                dev[:] = deviation
            except np.linalg.LinAlgError:
                dev[:] = np.nan
        else:  # homography
            H = transform_matrix
            h11, h12, h13 = H[0, 0], H[0, 1], H[0, 2]
            h21, h22, h23 = H[1, 0], H[1, 1], H[1, 2]
            h31, h32 = H[2, 0], H[2, 1]

            for idx, (x, y) in enumerate(points_xy):
                p_homogeneous = np.array([x, y, 1.0])
                p_transformed = H @ p_homogeneous
                w = p_transformed[2]
                if abs(w) < 1e-9:
                    continue

                numerator_u = h11 * x + h12 * y + h13
                numerator_v = h21 * x + h22 * y + h23

                du_dx = (h11 * w - numerator_u * h31) / (w * w)
                du_dy = (h12 * w - numerator_u * h32) / (w * w)
                dv_dx = (h21 * w - numerator_v * h31) / (w * w)
                dv_dy = (h22 * w - numerator_v * h32) / (w * w)

                J = np.array([[du_dx, du_dy], [dv_dx, dv_dy]])
                try:
                    J_inv = np.linalg.inv(J)
                    sigma = np.linalg.svd(J_inv, compute_uv=False)
                    sigma1, sigma2 = sigma[0], sigma[1]
                    scale = np.sqrt(abs(sigma1 * sigma2))
                    if scale > 1e-9:
                        sigma1_norm = sigma1 / scale
                        sigma2_norm = sigma2 / scale
                        deviation = max(abs(sigma1_norm - 1.0), abs(sigma2_norm - 1.0))
                    else:
                        deviation = 1e6
                    dev[idx] = deviation
                except np.linalg.LinAlgError:
                    continue

        return dev

    def _save_result(self) -> None:
        if self.transform_matrix is None:
            QMessageBox.warning(self, "Warning", "Please compute transformation and generate overlay result first")
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, "Select save directory", "")
        if not save_dir:
            return
        
        try:
            # Save overlay image (300dpi, high resolution and no white edges)
            overlay_path = os.path.join(save_dir, "overlay_result.png")
            self.overlay_canvas.figure.savefig(
                overlay_path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            
            # Save transformation matrix
            matrix_path = os.path.join(save_dir, f"transform_{self.method}.npy")
            np.save(matrix_path, self.transform_matrix)
            
            # Save MLSD metric
            mlsd_path = os.path.join(save_dir, "mlsd_metric.txt")
            with open(mlsd_path, "w", encoding="utf-8") as f:
                if self.mlsd_value is not None:
                    f.write(f"MLSD (Map Local Scale Deviation): {self.mlsd_value:.6f}\n")
                else:
                    f.write("MLSD (Map Local Scale Deviation): calculation failed\n")
                f.write(f"Transformation method: {self.method}\n")
                f.write("Definition: maximum scale-normalized local anisotropy over Ω.\n")
                f.write("Formula: max_p max(|σ1/sqrt(σ1σ2)-1|, |σ2/sqrt(σ1σ2)-1|), where σ are singular values of J_p^-1.\n")
            
            save_msg = f"Result saved to:\n{save_dir}\n\n- overlay_result.png\n- transform_{self.method}.npy\n- mlsd_metric.txt"
            if self.mlsd_value is not None:
                save_msg += f"\n\nMLSD: {self.mlsd_value:.6f}"

            # Save deformation color map (if exists)
            if self.deformation_vis_data is not None:
                try:
                    from matplotlib import pyplot as plt
                    from matplotlib import colors as mcolors

                    data = self.deformation_vis_data
                    uv = data["uv"]
                    dev = data["dev"]
                    max_dev = data.get("max_dev", float(np.nanmax(dev)))

                    # Filter valid values
                    mask = np.isfinite(dev)
                    uv_valid = uv[mask]
                    dev_valid = dev[mask]

                    if len(dev_valid) > 0:
                        # Adapt image scale based on aspect ratio of point cloud in image, avoid color bar being too high
                        u_span = float(uv_valid[:, 0].max() - uv_valid[:, 0].min())
                        v_span = float(uv_valid[:, 1].max() - uv_valid[:, 1].min())
                        if u_span <= 0 or v_span <= 0:
                            fig_w, fig_h = 8.0, 8.0
                        else:
                            aspect = v_span / u_span
                            base = 6.0
                            fig_w = base
                            fig_h = base * aspect
                            # Limit height to avoid extreme long bar
                            fig_h = max(4.0, min(10.0, fig_h))

                        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                        norm = mcolors.Normalize(vmin=0, vmax=max_dev if max_dev > 0 else 1.0)
                        sc = ax.scatter(uv_valid[:, 0], uv_valid[:, 1], s=1, c=dev_valid,
                                        cmap="plasma", norm=norm, alpha=0.7)
                        ax.set_aspect("equal")
                        ax.set_xlabel("u (px)")
                        ax.set_ylabel("v (px)")
                        ax.grid(True, alpha=0.3)
                        title_max = f"{max_dev:.6f}" if np.isfinite(max_dev) else "nan"
                        ax.set_title(
                            f"Deformation heatmap (method: {self.method})\n"
                            f"max deviation: {title_max}",
                            fontsize=11,
                        )
                        # Make color bar height follow current coordinate axis, not the entire rectangular canvas
                        cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
                        cbar.set_label("Local deformation (anisotropy)", fontsize=10)
                        deform_path = os.path.join(save_dir, "overlay_deformation.png")
                        plt.savefig(deform_path, dpi=300, bbox_inches="tight", pad_inches=0)
                        plt.close(fig)
                        save_msg += "\n- overlay_deformation.png"
                except Exception as exc:
                    save_msg += f"\n- overlay_deformation.png (Failed to save: {exc})"

            if self.ocs_result is not None:
                try:
                    self._save_ocs_exports(save_dir)
                    save_msg += "\n- ocs_result.json\n- ocs_points.csv\n- ocs_candidate_scenarios.csv"
                except Exception as exc:
                    save_msg += f"\n- OCS export failed: {exc}"
            
            QMessageBox.information(self, "Save successful", save_msg)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Save failed:\n{exc}")

    def _save_ocs_exports(self, save_dir: str) -> None:
        if self.ocs_result is None:
            return
        payload = {
            "calibration": {
                "label": "uncalibrated_geometry_confidence",
                "text": "未校准几何置信/经验支持",
            },
            "coordinate_frames": {
                "pcd": "pcd_original_xy_m",
                "satellite": "satellite_pixel_xy",
            },
            "params": self._ocs_params(),
            "result": self.ocs_result,
        }
        json_path = os.path.join(save_dir, "ocs_result.json")
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(self._jsonable(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)

        csv_path = os.path.join(save_dir, "ocs_points.csv")
        pair_rows = self.ocs_result.get("pairs") or []
        fields = [
            "index",
            "pcd_manual_xy_m",
            "pcd_raw_xy_m",
            "pcd_refined_xy_m",
            "pcd_raw_offset_m",
            "pcd_refined_offset_m",
            "pcd_status",
            "sat_manual_xy_px",
            "sat_raw_xy_px",
            "sat_refined_xy_px",
            "sat_raw_offset_px",
            "sat_refined_offset_px",
            "sat_status",
            "raw_success_rate",
            "raw_valid_rate",
            "refined_success_rate",
            "refined_valid_rate",
            "pcd_raw_samples",
            "pcd_refined_samples",
            "sat_raw_samples",
            "sat_refined_samples",
            "reason",
            "pcd_coordinate_frame",
            "sat_coordinate_frame",
            "uncalibrated_label",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, pair in enumerate(pair_rows):
                offsets = pair.get("offsets", {})
                samples = pair.get("samples", {})
                raw_score = self._score_row(self.ocs_result, "raw", index)
                refined_score = self._score_row(self.ocs_result, "refined", index)
                diagnostics = pair.get("diagnostics", {})
                pcd_diag = diagnostics.get("pointcloud") or {}
                sat_diag = diagnostics.get("satellite") or {}
                writer.writerow(
                    {
                        "index": index,
                        "pcd_manual_xy_m": self._cell(self._nested(pair, ("manual", "pcd_xy_m"))),
                        "pcd_raw_xy_m": self._cell(self._nested(pair, ("raw", "pcd_xy_m"))),
                        "pcd_refined_xy_m": self._cell(self._nested(pair, ("refined", "pcd_xy_m"))),
                        "pcd_raw_offset_m": self._cell(self._nested(offsets, ("raw_pcd_m", "xy"))),
                        "pcd_refined_offset_m": self._cell(self._nested(offsets, ("refined_pcd_m", "xy"))),
                        "pcd_status": pair.get("status", ""),
                        "sat_manual_xy_px": self._cell(self._nested(pair, ("manual", "sat_xy_px"))),
                        "sat_raw_xy_px": self._cell(self._nested(pair, ("raw", "sat_xy_px"))),
                        "sat_refined_xy_px": self._cell(self._nested(pair, ("refined", "sat_xy_px"))),
                        "sat_raw_offset_px": self._cell(self._nested(offsets, ("raw_sat_px", "xy"))),
                        "sat_refined_offset_px": self._cell(self._nested(offsets, ("refined_sat_px", "xy"))),
                        "sat_status": pair.get("status", ""),
                        "raw_success_rate": raw_score.get("success_rate", ""),
                        "raw_valid_rate": raw_score.get("valid_rate", ""),
                        "refined_success_rate": refined_score.get("success_rate", ""),
                        "refined_valid_rate": refined_score.get("valid_rate", ""),
                        "pcd_raw_samples": samples.get("pcd_raw", ""),
                        "pcd_refined_samples": samples.get("pcd_refined", ""),
                        "sat_raw_samples": samples.get("sat_raw", ""),
                        "sat_refined_samples": samples.get("sat_refined", ""),
                        "reason": pcd_diag.get("reason") or sat_diag.get("reason") or ",".join(pair.get("reasons") or []),
                        "pcd_coordinate_frame": "pcd_original_xy_m",
                        "sat_coordinate_frame": "satellite_pixel_xy",
                        "uncalibrated_label": "未校准几何置信/经验支持",
                    }
                )

        scenario_path = os.path.join(save_dir, "ocs_candidate_scenarios.csv")
        scenario_fields = [
            "scenario_index",
            "included_ids",
            "excluded",
            "raw_same_subset_mlsd",
            "refined_same_subset_mlsd",
            "refined_mean_mlsd",
            "refined_range_min",
            "refined_range_max",
            "range_type",
            "aggregation",
            "truncated",
            "truncated_reason",
            "valid",
            "reason",
        ]
        scenarios = self._candidate_scenarios(self.ocs_result)
        refined_range = self._candidate_range(self.ocs_result)
        raw_common = self._mlsd_value(self._mlsd_item(self.ocs_result, "raw_common"))
        refined = self._mlsd_item(self.ocs_result, "refined") or {}
        refined_mean = self._mlsd_value(refined)
        with open(scenario_path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=scenario_fields)
            writer.writeheader()
            if scenarios:
                for row in scenarios:
                    writer.writerow(
                        {
                            "scenario_index": row.get("scenario_index", ""),
                            "included_ids": self._cell([item.get("identity") for item in row.get("candidates", []) if isinstance(item, dict)]),
                            "excluded": self._coverage_summary(self.ocs_result)["excluded"],
                            "raw_same_subset_mlsd": raw_common,
                            "refined_same_subset_mlsd": row.get("value"),
                            "refined_mean_mlsd": refined_mean,
                            "refined_range_min": "",
                            "refined_range_max": "",
                            "range_type": refined.get("range_type", ""),
                            "aggregation": refined.get("aggregation", ""),
                            "truncated": refined.get("truncated", ""),
                            "truncated_reason": refined.get("truncated_reason", ""),
                            "valid": row.get("status") == "ok",
                            "reason": row.get("reason", ""),
                        }
                    )
            else:
                writer.writerow(
                    {
                        "scenario_index": "",
                        "included_ids": self._coverage_summary(self.ocs_result)["included"],
                        "excluded": self._coverage_summary(self.ocs_result)["excluded"],
                        "raw_same_subset_mlsd": raw_common,
                        "refined_same_subset_mlsd": "",
                        "refined_mean_mlsd": refined_mean,
                        "refined_range_min": refined_range[0] if refined_range is not None else "",
                        "refined_range_max": refined_range[1] if refined_range is not None else "",
                        "range_type": refined.get("range_type", ""),
                        "aggregation": refined.get("aggregation", ""),
                        "truncated": refined.get("truncated", ""),
                        "truncated_reason": refined.get("truncated_reason", ""),
                        "valid": refined_range is not None,
                        "reason": "",
                    }
                )

    def _cell(self, value) -> str:
        if value is None:
            return ""
        return json.dumps(self._jsonable(value), ensure_ascii=False, allow_nan=False)

    def _jsonable(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return self._jsonable(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        return value

