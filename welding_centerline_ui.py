# -*- coding: utf-8 -*-
"""
Welding Centerline UI (PySide6) — with configurable depth_margin & pixel_size_mm

新增：
- 控件可编辑 depth_margin(mm) 与 pixel_size_mm(mm/px)，覆盖 ui_pipeline 的默认配置。
- 新增“表面掩码提取方法”选择（最近Z层 / RANSAC平面 / 倾斜校正网格），并提供 RANSAC 高级参数。
"""
import os
import sys
import time
import traceback
import subprocess
from typing import Optional, Tuple

import numpy as np
import cv2

from PySide6.QtCore import Qt, QRect, QPoint, QSize, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QPushButton, QVBoxLayout, QGroupBox, QFormLayout,
    QGridLayout, QStatusBar, QSplitter, QSpinBox, QScrollArea, QDoubleSpinBox, QComboBox, QHBoxLayout
)

# ---- Percipio SDK ----
try:
    import pcammls
    from pcammls import *
except Exception as e:
    pcammls = None
    print("[WARN] Percipio SDK not available:", e)

# ---- detect 导入（不改动原文件） ----
try:
    import detect_simplified as detect
except Exception:
    try:
        import detect as detect
    except Exception as e:
        detect = None
        print("[WARN] detect module import failed:", e)

# ---- UI 工具 ----
def np_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    if img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())
    if img.ndim == 3 and img.shape[2] == 4:
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimg.copy())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return np_to_qpixmap(gray)

class ScaledImageLabel(QLabel):
    """
    等比缩放显示图像的 QLabel。
    - set_image(np.ndarray 或 QPixmap)
    - 自动在 resizeEvent 中重算缩放，保持窗口不被原始大图撑大
    """
    def __init__(self, parent=None, min_h: int = 260):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;")
        self._orig_pm: Optional[QPixmap] = None
        self._scaled_pm: Optional[QPixmap] = None
        self.setMinimumHeight(min_h)
    def set_image(self, img):
        if isinstance(img, np.ndarray):
            self._orig_pm = np_to_qpixmap(img)
        elif isinstance(img, QPixmap):
            self._orig_pm = img
        else:
            self._orig_pm = None
        self._rescale_and_set()
    def resizeEvent(self, e):
        super().resizeEvent(e); self._rescale_and_set()
    def _rescale_and_set(self):
        if self._orig_pm is None or self.width() <= 2 or self.height() <= 2:
            self.setPixmap(QPixmap()); return
        self._scaled_pm = self._orig_pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self._scaled_pm)

class ROIOverlayLabel(ScaledImageLabel):
    """
    支持鼠标拖拽 ROI 的等比缩放图像控件。
    关键点：
    - 始终对齐 set_image / resizeEvent 的缩放结果，记录实际显示矩形(_display_rect)
    - 将控件坐标映射回原图像像素坐标时，考虑(居中)偏移与缩放比
    """
    roiChanged = Signal(QRect)
    def __init__(self, parent=None, min_h: int = 320):
        super().__init__(parent, min_h=min_h)
        self.setMouseTracking(True)
        self._dragging = False
        self._start = QPoint()
        self._rect = QRect()
        self._display_rect = QRect()
        self._orig_size = QSize()
    def set_image(self, img):
        if isinstance(img, np.ndarray):
            pm = np_to_qpixmap(img); self._orig_size = QSize(pm.width(), pm.height()); self._orig_pm = pm
        elif isinstance(img, QPixmap):
            self._orig_size = QSize(img.width(), img.height()); self._orig_pm = img
        else:
            self._orig_pm = None; self._orig_size = QSize()
        self._rescale_and_set()
    def _rescale_and_set(self):
        if self._orig_pm is None or self.width() <= 2 or self.height() <= 2:
            self.setPixmap(QPixmap()); self._display_rect = QRect(); return
        scaled = self._orig_pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_pm = scaled
        xoff = (self.width() - scaled.width()) // 2
        yoff = (self.height() - scaled.height()) // 2
        self._display_rect = QRect(xoff, yoff, scaled.width(), scaled.height())
        self.setPixmap(scaled); self.update()
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._rect.isNull():
            painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine); painter.setPen(pen)
            roi_draw = self._rect & self._display_rect
            painter.drawRect(roi_draw if not roi_draw.isNull() else self._rect)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True; self._start = event.position().toPoint()
            self._rect = QRect(self._start, self._start); self.update()
    def mouseMoveEvent(self, event):
        if self._dragging:
            end = event.position().toPoint()
            self._rect = QRect(self._start, end).normalized(); self.update(); self.roiChanged.emit(self._rect)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False; end = event.position().toPoint()
            self._rect = QRect(self._start, end).normalized(); self.update(); self.roiChanged.emit(self._rect)
    def get_roi_on_image(self, _ignored_shape=None) -> Optional[Tuple[int, int, int, int]]:
        if self._orig_size.isEmpty() or self._rect.isNull() or self._display_rect.isNull():
            return None
        roi = self._rect & self._display_rect
        if roi.isNull(): return None
        sx = self._orig_size.width() / max(1, self._display_rect.width())
        sy = self._orig_size.height() / max(1, self._display_rect.height())
        def map_pt(p: QPoint):
            x = (p.x() - self._display_rect.left()) * sx
            y = (p.y() - self._display_rect.top()) * sy
            return int(np.clip(x, 0, self._orig_size.width()-1)), int(np.clip(y, 0, self._orig_size.height()-1))
        x1, y1 = map_pt(roi.topLeft()); x2, y2 = map_pt(roi.bottomRight())
        if x2 <= x1 or y2 <= y1: return None
        return (x1, y1, x2, y2)

# ---- 取流线程 ----
class PercipioCaptureThread(QThread):
    frameReady = Signal(object, object, object)
    status = Signal(str)
    startedOk = Signal(bool)
    def __init__(self, parent=None):
        super().__init__(parent); self._running = False
    def run(self):
        if pcammls is None:
            self.status.emit("Percipio SDK 不可用，无法连接相机。"); self.startedOk.emit(False); return
        try:
            cl = pcammls.PercipioSDK()
            dev_list = cl.ListDevice()
            if not dev_list:
                self.status.emit("未找到相机设备。"); self.startedOk.emit(False); return
            handle = cl.Open(dev_list[0].id)
            if not cl.isValidHandle(handle):
                self.status.emit("打开设备失败。"); self.startedOk.emit(False); return
            err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
            if err:
                self.status.emit(f"启用数据流失败: {err}"); self.startedOk.emit(False); return
            cl.DeviceLoadDefaultParameters(handle)
            scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
            depth_calib_data = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
            cl.DeviceStreamOn(handle); self._running = True; self.startedOk.emit(True)
            img_render = pcammls.image_data(); img_color = pcammls.image_data()
            pointcloud_data_arr = pcammls.pointcloud_data_list()
            last_emit = time.time()
            while self._running:
                image_list = cl.DeviceStreamRead(handle, 2000)
                if not image_list: continue
                depth_frame = None; color_frame = None
                for frame in image_list:
                    if frame.streamID == PERCIPIO_STREAM_DEPTH: depth_frame = frame
                    elif frame.streamID == PERCIPIO_STREAM_COLOR: color_frame = frame
                if depth_frame is None or color_frame is None: continue
                cl.DeviceStreamDepthRender(depth_frame, img_render)
                depth_bgr = img_render.as_nparray().copy()
                cl.DeviceStreamImageDecode(color_frame, img_color)
                color_bgr = img_color.as_nparray().copy()
                cl.DeviceStreamMapDepthImageToPoint3D(depth_frame, depth_calib_data, scale_unit, pointcloud_data_arr)
                p3d = pointcloud_data_arr.as_nparray().copy()
                now = time.time()
                if now - last_emit > 0.03:
                    self.frameReady.emit(depth_bgr, color_bgr, p3d); last_emit = now
            cl.DeviceStreamOff(handle); cl.Close(handle)
        except Exception as e:
            self.status.emit("采集线程异常: " + str(e)); traceback.print_exc(); self.startedOk.emit(False)
    def stop(self): self._running = False

# ---- 管线 ----
from ui_pipeline import Pipeline, ProcessingResult

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("融海微-搅拌摩擦焊在线检测|GUI")
        self.resize(1280, 800)
        self.depth_last = None; self.color_last = None; self.p3d_last = None
        self.pipeline = Pipeline()
        self._build_ui(); self._connect_signals(); self._refresh_asset_labels()

    def _build_ui(self):
        menubar = self.menuBar()
        m_dev = menubar.addMenu("设备"); m_tools = menubar.addMenu("工具/脚本"); m_help = menubar.addMenu("帮助")
        self.act_connect = QAction("连接相机", self); m_dev.addAction(self.act_connect)
        self.act_disconnect = QAction("断开相机", self); m_dev.addAction(self.act_disconnect)
        self.act_roi = QAction("交互式 ROI（cal_roi.py）", self); m_tools.addAction(self.act_roi)
        self.act_draw = QAction("理论中轴线绘制/提取（draw_centerline.py）", self); m_tools.addAction(self.act_draw)
        self.act_tilt = QAction("平面倾斜校正（my_calibrate.py）", self); m_tools.addAction(self.act_tilt)
        self.act_handeye = QAction("手眼标定（calibrate_hand_eye.py）", self); m_tools.addAction(self.act_handeye)
        self.act_detect = QAction("原检测流程（detect_simplified.py）", self); m_tools.addAction(self.act_detect)
        self.act_readme = QAction("打开 README", self); m_help.addAction(self.act_readme)

        # 左侧：预览/ROI
        left_box = QGroupBox("预览 / ROI")
        self.lbl_depth = ROIOverlayLabel(min_h=360)
        self.lbl_color = ScaledImageLabel(min_h=260)
        lv = QVBoxLayout(left_box)
        lv.addWidget(QLabel("深度渲染")); lv.addWidget(self.lbl_depth, 4)
        lv.addWidget(QLabel("彩色图"));   lv.addWidget(self.lbl_color, 3)

        # 右侧：结果（滚动）
        right_content = QWidget(); grid = QGridLayout(right_content)
        right_content.setStyleSheet("background:#0b0b0b;")
        self.lbl_mask = ScaledImageLabel(min_h=240)
        self.lbl_skel_overlay = ScaledImageLabel(min_h=240)
        self.lbl_compare = ScaledImageLabel(min_h=340)
        grid.addWidget(QLabel("表面掩码"), 0, 0); grid.addWidget(self.lbl_mask, 1, 0)
        grid.addWidget(QLabel("骨架叠加（深度图）"), 0, 1); grid.addWidget(self.lbl_skel_overlay, 1, 1)
        grid.addWidget(QLabel("中轴线比较（含偏移箭头）"), 2, 0, 1, 2); grid.addWidget(self.lbl_compare, 3, 0, 1, 2)
        scroll = QScrollArea(); scroll.setWidget(right_content); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: #111; }")

        # 控制 / 参数 / 状态
        ctrl_box = QGroupBox("控制 / 参数 / 状态")
        self.btn_process = QPushButton("处理选中 ROI")
        self.btn_export = QPushButton("导出 CSV")
        self.btn_reload_assets = QPushButton("重新加载参数(.npy)")
        self.ed_num_keypoints = QSpinBox(); self.ed_num_keypoints.setRange(4, 200); self.ed_num_keypoints.setValue(12)

        # 通用参数
        self.ed_depth_margin = QDoubleSpinBox()
        self.ed_depth_margin.setRange(0.1, 50.0); self.ed_depth_margin.setSingleStep(0.1)
        self.ed_depth_margin.setDecimals(2); self.ed_depth_margin.setValue(3.5)
        self.ed_depth_margin.setSuffix(" mm")
        self.ed_depth_margin.setToolTip("离表面最近层的深度容差 (单位: mm)")

        self.ed_pixel_size_mm = QDoubleSpinBox()
        self.ed_pixel_size_mm.setRange(0.01, 10.0); self.ed_pixel_size_mm.setSingleStep(0.05)
        self.ed_pixel_size_mm.setDecimals(3); self.ed_pixel_size_mm.setValue(0.5)
        self.ed_pixel_size_mm.setSuffix(" mm/px")
        self.ed_pixel_size_mm.setToolTip("栅格像素对应的物理尺寸，仅 create_corrected_mask 时生效")

        # 表面方法选择
        self.cmb_surface_method = QComboBox()
        self.cmb_surface_method.addItems([
            "最近Z层(快速)",               # nearest
            "RANSAC平面(粗糙场景)",        # ransac
            "倾斜校正网格(create_corrected_mask)"  # corrected
        ])
        self.cmb_surface_method.setCurrentIndex(0)  # 默认最近Z层

        # RANSAC 高级参数
        ran_box = QGroupBox("RANSAC 高级参数（粗糙场景）")
        self.ed_ransac_iters = QSpinBox(); self.ed_ransac_iters.setRange(50, 5000); self.ed_ransac_iters.setValue(400)
        self.ed_ransac_dist = QDoubleSpinBox(); self.ed_ransac_dist.setRange(0.05, 10.0); self.ed_ransac_dist.setDecimals(3); self.ed_ransac_dist.setValue(0.8); self.ed_ransac_dist.setSuffix(" mm")
        self.ed_ransac_front = QDoubleSpinBox(); self.ed_ransac_front.setRange(1.0, 100.0); self.ed_ransac_front.setDecimals(1); self.ed_ransac_front.setValue(20.0); self.ed_ransac_front.setSuffix(" %")
        self.ed_ransac_subs = QSpinBox(); self.ed_ransac_subs.setRange(1000, 2000000); self.ed_ransac_subs.setValue(50000)
        self.ed_ransac_seed = QSpinBox(); self.ed_ransac_seed.setRange(-1, 10**9); self.ed_ransac_seed.setValue(-1)
        rf = QFormLayout(ran_box)
        rf.addRow("迭代次数", self.ed_ransac_iters)
        rf.addRow("距离阈值", self.ed_ransac_dist)
        rf.addRow("前景百分位", self.ed_ransac_front)
        rf.addRow("下采样上限", self.ed_ransac_subs)
        rf.addRow("随机种子(-1随机)", self.ed_ransac_seed)

        # 资产状态
        asset_box = QGroupBox("自动加载参数状态")
        self.lab_theory = QLabel("theoretical_centerline.npy: 未加载")
        self.lab_tilt = QLabel("tilt_correction_matrix.npy: 未加载")
        self.lab_handeye = QLabel("hand_eye_transform.npy: 未加载")

        # 偏移输出
        out_box = QGroupBox("偏移输出"); self.lab_px = QLabel("平均像素偏移: -"); self.lab_mm = QLabel("平均物理偏移: -")

        # 布局
        form = QFormLayout(ctrl_box)
        form.addRow(self.btn_process, self.btn_export)
        form.addRow("表面方法", self.cmb_surface_method)
        form.addRow("Depth Margin", self.ed_depth_margin)
        form.addRow("Pixel Size (mm/px)", self.ed_pixel_size_mm)
        form.addRow("偏移关键点数量", self.ed_num_keypoints)
        form.addRow(ran_box)
        form.addRow(self.btn_reload_assets)

        va = QVBoxLayout(asset_box); va.addWidget(self.lab_theory); va.addWidget(self.lab_tilt); va.addWidget(self.lab_handeye)
        vo = QVBoxLayout(out_box); vo.addWidget(self.lab_px); vo.addWidget(self.lab_mm)

        self.log = QLabel("Ready."); self.log.setStyleSheet("color:#aaa")

        left_panel = QWidget(); left_v = QVBoxLayout(left_panel)
        left_v.addWidget(left_box); left_v.addWidget(ctrl_box); left_v.addWidget(asset_box); left_v.addWidget(out_box); left_v.addWidget(self.log)

        splitter = QSplitter(); splitter.addWidget(left_panel); splitter.addWidget(scroll); splitter.setSizes([620, 660])
        self.setCentralWidget(splitter); self.setStatusBar(QStatusBar())

        # 初始：根据方法选择显示/隐藏 RANSAC 框
        ran_box.setVisible(self.cmb_surface_method.currentIndex() == 1)
        self._ran_box = ran_box  # 持有引用，便于切换可见性

    def _connect_signals(self):
        self.act_connect.triggered.connect(self.start_camera)
        self.act_disconnect.triggered.connect(self.stop_camera)
        self.act_roi.triggered.connect(lambda: self._run_script("cal_roi.py"))
        self.act_draw.triggered.connect(lambda: self._run_script("draw_centerline.py"))
        self.act_tilt.triggered.connect(lambda: self._run_script("my_calibrate.py"))
        self.act_handeye.triggered.connect(lambda: self._run_script("calibrate_hand_eye.py"))
        self.act_detect.triggered.connect(lambda: self._run_script("detect_simplified.py"))
        self.act_readme.triggered.connect(self._open_readme)
        self.btn_process.clicked.connect(self.on_process_clicked)
        self.btn_export.clicked.connect(self.on_export_clicked)
        self.btn_reload_assets.clicked.connect(self._reload_assets)

        # 当参数改变时，同步到 Pipeline 的默认配置（即使不点处理也先存起来）
        self.ed_depth_margin.valueChanged.connect(lambda v: self.pipeline.set_config(depth_margin=v))
        self.ed_pixel_size_mm.valueChanged.connect(lambda v: self.pipeline.set_config(pixel_size_mm=v))

        self.cmb_surface_method.currentIndexChanged.connect(self._on_surface_method_changed)

        # RANSAC 实时同步默认配置
        self.ed_ransac_iters.valueChanged.connect(lambda v: self.pipeline.set_config(ransac_iters=v))
        self.ed_ransac_dist.valueChanged.connect(lambda v: self.pipeline.set_config(ransac_dist_thresh=v))
        self.ed_ransac_front.valueChanged.connect(lambda v: self.pipeline.set_config(ransac_front_percentile=v))
        self.ed_ransac_subs.valueChanged.connect(lambda v: self.pipeline.set_config(ransac_subsample=v))
        self.ed_ransac_seed.valueChanged.connect(lambda v: self.pipeline.set_config(ransac_seed=v))

        # 初始化一次默认方法
        self.pipeline.set_config(surface_method=self._current_method_key())

    def _current_method_key(self) -> str:
        idx = self.cmb_surface_method.currentIndex()
        return ("nearest", "ransac", "corrected")[idx]

    def _on_surface_method_changed(self, idx: int):
        # 切换 RANSAC 参数框可见性
        self._ran_box.setVisible(idx == 1)
        # 同步配置
        self.pipeline.set_config(surface_method=self._current_method_key())

    # ---- 设备 ----
    def start_camera(self):
        if hasattr(self, "cap_thread") and self.cap_thread.isRunning():
            self._log("相机已连接。"); return
        self.cap_thread = PercipioCaptureThread(self)
        self.cap_thread.frameReady.connect(self._on_frame)
        self.cap_thread.status.connect(self._log)
        self.cap_thread.startedOk.connect(lambda ok: self.statusBar().showMessage("已连接" if ok else "连接失败"))
        self.cap_thread.start()

    def stop_camera(self):
        if hasattr(self, "cap_thread"): self.cap_thread.stop()
        self.statusBar().showMessage("已断开")

    def _on_frame(self, depth_bgr, color_bgr, p3d):
        self.depth_last = depth_bgr; self.color_last = color_bgr; self.p3d_last = p3d
        self.lbl_depth.set_image(depth_bgr); self.lbl_color.set_image(color_bgr)

    # ---- 处理 ----
    def _get_current_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self.lbl_depth.get_roi_on_image()

    def on_process_clicked(self):
        if self.depth_last is None or self.p3d_last is None:
            QMessageBox.warning(self, "提示", "还没有相机帧。请先连接设备。"); return
        roi = self._get_current_roi()
        if roi is None:
            QMessageBox.warning(self, "提示", "ROI 为空或越界，请在左侧“深度渲染”区域框选可见区域内的 ROI。"); return

        dm = float(self.ed_depth_margin.value())
        pmm = float(self.ed_pixel_size_mm.value())
        method_key = self._current_method_key()

        # RANSAC 参数（仅当选择 RANSAC 时会真正影响）
        iters = int(self.ed_ransac_iters.value())
        dist_th = float(self.ed_ransac_dist.value())
        front_p = float(self.ed_ransac_front.value())
        subs = int(self.ed_ransac_subs.value())
        seed = int(self.ed_ransac_seed.value())

        try:
            res: ProcessingResult = self.pipeline.process_frame(
                p3d_nparray=self.p3d_last,
                depth_render_bgr=self.depth_last,
                roi_xyxy=roi,
                num_key_points=self.ed_num_keypoints.value(),
                depth_margin=dm,
                pixel_size_mm=pmm,
                surface_method=method_key,
                ransac_iters=iters,
                ransac_dist_thresh=dist_th,
                ransac_front_percentile=front_p,
                ransac_subsample=subs,
                ransac_seed=seed
            )
        except Exception as e:
            traceback.print_exc(); QMessageBox.critical(self, "处理失败", str(e)); return

        # 可视化
        self.lbl_mask.set_image(res.surface_mask if res.surface_mask is not None else None)
        self.lbl_skel_overlay.set_image(res.skeleton_overlay if res.skeleton_overlay is not None else None)
        self.lbl_compare.set_image(res.comparison_with_vectors if res.comparison_with_vectors is not None else None)

        # 偏移输出
        if res.avg_pixel_offset is not None:
            dx, dy = res.avg_pixel_offset; self.lab_px.setText(f"平均像素偏移: ({dx:.2f}, {dy:.2f}) px")
        else:
            self.lab_px.setText("平均像素偏移: -")
        if res.avg_mm_offset is not None:
            mx, my = res.avg_mm_offset; self.lab_mm.setText(f"平均物理偏移: ({mx:.3f}, {my:.3f}) mm")
        else:
            self.lab_mm.setText("平均物理偏移: -")

        self._log(res.msg or "处理完成")

    def on_export_clicked(self):
        if not self.pipeline.has_result():
            QMessageBox.information(self, "提示", "暂无可导出的检测结果。请先点击“处理选中 ROI”。"); return
        default = os.path.join(os.getcwd(), time.strftime("centerline_offsets_%Y%m%d_%H%M%S.csv"))
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", default, "CSV (*.csv)")
        if not path: return
        try:
            self.pipeline.export_csv(path)
            QMessageBox.information(self, "导出成功", f"CSV 已导出：\n{path}")
        except Exception as e:
            traceback.print_exc(); QMessageBox.critical(self, "导出失败", str(e))

    # ---- 资产 ----
    def _reload_assets(self):
        self.pipeline.load_assets(); self._refresh_asset_labels(); self._log("参数重载完成。")

    def _refresh_asset_labels(self):
        self.lab_theory.setText(f"theoretical_centerline.npy: {'已加载' if self.pipeline.theory_ok else '未找到'}")
        self.lab_tilt.setText(f"tilt_correction_matrix.npy: {'已加载' if self.pipeline.tilt_ok else '未找到'}")
        self.lab_handeye.setText(f"hand_eye_transform.npy: {'已加载' if self.pipeline.handeye_ok else '未找到'}")

    # ---- 其他 ----
    def _run_script(self, filename: str):
        script_path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "找不到脚本", f"{filename} 不存在。"); return
        try:
            subprocess.Popen([sys.executable, script_path]); self._log(f"已启动脚本：{filename}")
        except Exception:
            QMessageBox.critical(self, "启动失败", f"无法启动：{filename}")

    def _open_readme(self):
        path = os.path.join(os.getcwd(), "README.md")
        if os.path.exists(path):
            if sys.platform.startswith("win"): os.startfile(path)  # type: ignore
            elif sys.platform == "darwin": subprocess.call(["open", path])
            else: subprocess.call(["xdg-open", path])
        else:
            QMessageBox.information(self, "README 缺失", "项目目录下未找到 README.md")

    def _log(self, text: str):
        self.log.setText(text); print(text)

def main():
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()
