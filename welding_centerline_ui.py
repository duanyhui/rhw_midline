# -*- coding: utf-8 -*-
"""
Welding Centerline UI (PySide6)

交付特性：
- 统一 GUI，将现有脚本（ROI、手眼标定、倾斜校正、理论中轴线绘制/提取、检测）整合到一个工程化 UI。
- 不更动既有脚本文件的注释与函数；仅通过 import/subprocess 调用。
- 全链路：ROI → 表面掩码 → 骨架（实际中轴线） → 与理论中轴线比较 → 偏移向量与物理坐标（若有） → 导出 CSV。
- 所有关键可视化在 UI 内通过 QLabel(QPixmap) 呈现；若原函数带有弹窗，允许保留。
- 一键运行：python welding_centerline_ui.py
"""
import os
import sys
import time
import traceback
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cv2

from PySide6.QtCore import Qt, QRect, QPoint, QSize, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QGridLayout, QStatusBar, QTabWidget, QSplitter, QLineEdit, QSpinBox
)

# ---- Percipio SDK ----
try:
    import pcammls
    from pcammls import *
except Exception as e:
    pcammls = None  # 允许在无 SDK 环境下加载 UI
    print("[WARN] Percipio SDK not available in this environment:", e)

# ---- 现有算法函数（只 import，不改动脚本源文件） ----
# detect_simplified.py 作为核心算法模块（用户提供）
try:
    import detect_simplified as detect
except Exception:
    # 也允许文件名为 detect.py 的情况
    try:
        import detect as detect
    except Exception as e:
        detect = None
        print("[WARN] detect module import failed:", e)

# 其他原脚本的菜单直启（保持原注释/交互）：
# - cal_roi.py
# - draw_centerline.py
# - my_calibrate.py
# - calibrate_hand_eye.py

# ---- UI 工具函数 ----
def np_to_qpixmap(img: np.ndarray) -> QPixmap:
    """将 BGR/灰度 numpy 图像转换为 QPixmap。自动处理通道和步长。"""
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    elif img.ndim == 3:
        if img.shape[2] == 3:  # BGR -> RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())
        elif img.shape[2] == 4:
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg.copy())
    # 兜底：转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return np_to_qpixmap(gray)

class ROIOverlayLabel(QLabel):
    """支持鼠标拖拽选择 ROI 的 QLabel。内部维护一个选框矩形。"""
    roiChanged = Signal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._dragging = False
        self._start = QPoint()
        self._rect = QRect()
        self._last_frame_size = QSize()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._rect.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._start = event.position().toPoint()
            self._rect = QRect(self._start, self._start)
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            end = event.position().toPoint()
            self._rect = QRect(self._start, end).normalized()
            self.update()
            self.roiChanged.emit(self._rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            end = event.position().toPoint()
            self._rect = QRect(self._start, end).normalized()
            self.update()
            self.roiChanged.emit(self._rect)

    def setPixmap(self, pm: QPixmap) -> None:
        """保存最后一帧尺寸以便将 ROI 映射回原始分辨率坐标。"""
        super().setPixmap(pm)
        self._last_frame_size = pm.size()

    def get_roi_on_image(self, image_shape_hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """将控件坐标 ROI 映射到图像像素坐标 (x1,y1,x2,y2)。"""
        if self._rect.isNull() or self._last_frame_size.isEmpty():
            return None
        disp_w, disp_h = self._last_frame_size.width(), self._last_frame_size.height()
        img_h, img_w = image_shape_hw
        sx = img_w / max(1, disp_w)
        sy = img_h / max(1, disp_h)
        x1 = int(self._rect.left() * sx)
        y1 = int(self._rect.top() * sy)
        x2 = int(self._rect.right() * sx)
        y2 = int(self._rect.bottom() * sy)
        # 规范化边界
        x1, x2 = max(0, min(x1, img_w-1)), max(0, min(x2, img_w-1))
        y1, y2 = max(0, min(y1, img_h-1)), max(0, min(y2, img_h-1))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

# ---- 捕获线程：读取深度/彩色并输出渲染图像与点云 ----
class PercipioCaptureThread(QThread):
    frameReady = Signal(object, object, object)  # depth_render_bgr, color_bgr, p3d_nparray
    status = Signal(str)
    startedOk = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._handle = None
        self._cl = None
        self._depth_calib = None
        self._scale_unit = None
        self._depth_calib_data = None

    def run(self):
        if pcammls is None:
            self.status.emit("Percipio SDK 不可用，无法连接相机。")
            self.startedOk.emit(False)
            return

        try:
            self._cl = pcammls.PercipioSDK()
            dev_list = self._cl.ListDevice()
            if not dev_list:
                self.status.emit("未找到相机设备。")
                self.startedOk.emit(False)
                return
            self._handle = self._cl.Open(dev_list[0].id)
            if not self._cl.isValidHandle(self._handle):
                self.status.emit("打开设备失败。")
                self.startedOk.emit(False)
                return

            err = self._cl.DeviceStreamEnable(self._handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
            if err:
                self.status.emit(f"启用数据流失败: {err}")
                self.startedOk.emit(False)
                return

            self._cl.DeviceLoadDefaultParameters(self._handle)
            self._scale_unit = self._cl.DeviceReadCalibDepthScaleUnit(self._handle)
            self._depth_calib = self._cl.DeviceReadCalibData(self._handle, PERCIPIO_STREAM_DEPTH)
            self._depth_calib_data = self._cl.DeviceReadCalibData(self._handle, PERCIPIO_STREAM_DEPTH)

            self._cl.DeviceStreamOn(self._handle)
            self._running = True
            self.startedOk.emit(True)
            img_registration_depth = pcammls.image_data()
            img_registration_render = pcammls.image_data()
            img_parsed_color = pcammls.image_data()
            pointcloud_data_arr = pcammls.pointcloud_data_list()

            last_emit = time.time()
            while self._running:
                image_list = self._cl.DeviceStreamRead(self._handle, 2000)
                if not image_list:
                    continue

                img_depth = None
                img_color = None
                for frame in image_list:
                    if frame.streamID == PERCIPIO_STREAM_DEPTH:
                        img_depth = frame
                    elif frame.streamID == PERCIPIO_STREAM_COLOR:
                        img_color = frame

                if img_depth is None or img_color is None:
                    continue

                # 深度渲染
                self._cl.DeviceStreamDepthRender(img_depth, img_registration_render)
                depth_render = img_registration_render.as_nparray().copy()

                # 彩色解码
                self._cl.DeviceStreamImageDecode(img_color, img_parsed_color)
                color_bgr = img_parsed_color.as_nparray().copy()

                # 点云
                self._cl.DeviceStreamMapDepthImageToPoint3D(
                    img_depth, self._depth_calib_data, self._scale_unit, pointcloud_data_arr
                )
                p3d = pointcloud_data_arr.as_nparray().copy()  # (H, W, 3) mm

                # 控制发射频率，避免 UI 过载
                now = time.time()
                if now - last_emit > 0.03:
                    self.frameReady.emit(depth_render, color_bgr, p3d)
                    last_emit = now

            self._cl.DeviceStreamOff(self._handle)
            self._cl.Close(self._handle)
        except Exception as e:
            self.status.emit("采集线程异常: " + str(e))
            traceback.print_exc()
            self.startedOk.emit(False)

    def stop(self):
        self._running = False

# ---- 处理管线（封装在 ui_pipeline.py 中） ----
from ui_pipeline import Pipeline, ProcessingResult  # 本工程自定义

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摩擦焊 视觉-运动 一体化 | Centerline GUI")
        self.resize(1480, 900)

        # 状态
        self.depth_last = None
        self.color_last = None
        self.p3d_last = None
        self.depth_shape_hw = (0, 0)

        # 管线
        self.pipeline = Pipeline()

        # UI 搭建
        self._build_ui()
        self._connect_signals()

        # 自动加载参数状态
        self._refresh_asset_labels()

    # ---------- UI ----------
    def _build_ui(self):
        # 菜单
        menubar = self.menuBar()
        m_dev = menubar.addMenu("设备")
        m_tools = menubar.addMenu("工具/脚本")
        m_help = menubar.addMenu("帮助")

        self.act_connect = QAction("连接相机", self); m_dev.addAction(self.act_connect)
        self.act_disconnect = QAction("断开相机", self); m_dev.addAction(self.act_disconnect)

        self.act_roi = QAction("交互式 ROI（cal_roi.py）", self); m_tools.addAction(self.act_roi)
        self.act_draw = QAction("理论中轴线绘制/提取（draw_centerline.py）", self); m_tools.addAction(self.act_draw)
        self.act_tilt = QAction("平面倾斜校正（my_calibrate.py）", self); m_tools.addAction(self.act_tilt)
        self.act_handeye = QAction("手眼标定（calibrate_hand_eye.py）", self); m_tools.addAction(self.act_handeye)
        self.act_detect = QAction("原检测流程（detect_simplified.py）", self); m_tools.addAction(self.act_detect)

        self.act_readme = QAction("打开 README", self); m_help.addAction(self.act_readme)

        # 左：预览与 ROI
        left_box = QGroupBox("预览 / ROI")
        self.lbl_depth = ROIOverlayLabel()
        self.lbl_depth.setAlignment(Qt.AlignCenter)
        self.lbl_depth.setStyleSheet("background:#111;")
        self.lbl_color = QLabel(alignment=Qt.AlignCenter)
        self.lbl_color.setStyleSheet("background:#111;")

        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(QLabel("深度渲染"))
        left_layout.addWidget(self.lbl_depth, 4)
        left_layout.addWidget(QLabel("彩色图"))
        left_layout.addWidget(self.lbl_color, 3)

        # 右：处理结果
        right_box = QGroupBox("处理结果（UI 内可视化）")
        self.lbl_mask = QLabel(alignment=Qt.AlignCenter)
        self.lbl_mask.setStyleSheet("background:#222;")
        self.lbl_skel_overlay = QLabel(alignment=Qt.AlignCenter)
        self.lbl_skel_overlay.setStyleSheet("background:#222;")
        self.lbl_compare = QLabel(alignment=Qt.AlignCenter)
        self.lbl_compare.setStyleSheet("background:#222;")

        grid = QGridLayout(right_box)
        grid.addWidget(QLabel("表面掩码"), 0, 0)
        grid.addWidget(self.lbl_mask, 1, 0)
        grid.addWidget(QLabel("骨架叠加（深度图）"), 0, 1)
        grid.addWidget(self.lbl_skel_overlay, 1, 1)
        grid.addWidget(QLabel("中轴线比较（含偏移箭头）"), 2, 0, 1, 2)
        grid.addWidget(self.lbl_compare, 3, 0, 1, 2)

        # 控制区
        ctrl_box = QGroupBox("控制 / 状态")
        self.btn_process = QPushButton("处理选中 ROI")
        self.btn_export = QPushButton("导出 CSV")
        self.btn_reload_assets = QPushButton("重新加载参数(.npy)")

        self.ed_num_keypoints = QSpinBox()
        self.ed_num_keypoints.setRange(4, 200)
        self.ed_num_keypoints.setValue(12)

        form = QFormLayout(ctrl_box)
        form.addRow(self.btn_process, self.btn_export)
        form.addRow("偏移关键点数量", self.ed_num_keypoints)
        form.addRow(self.btn_reload_assets)

        # 资产状态
        asset_box = QGroupBox("自动加载参数状态")
        self.lab_theory = QLabel("theoretical_centerline.npy: 未加载")
        self.lab_tilt = QLabel("tilt_correction_matrix.npy: 未加载")
        self.lab_handeye = QLabel("hand_eye_transform.npy: 未加载")
        asset_layout = QVBoxLayout(asset_box)
        asset_layout.addWidget(self.lab_theory)
        asset_layout.addWidget(self.lab_tilt)
        asset_layout.addWidget(self.lab_handeye)

        # 偏移输出
        out_box = QGroupBox("偏移输出")
        self.lab_px = QLabel("平均像素偏移: -")
        self.lab_mm = QLabel("平均物理偏移: -")
        v_out = QVBoxLayout(out_box)
        v_out.addWidget(self.lab_px)
        v_out.addWidget(self.lab_mm)

        # 底部日志
        self.log = QLabel("Ready.")
        self.log.setStyleSheet("color:#aaa")

        # 主布局
        splitter = QSplitter()
        left_panel = QWidget()
        lv = QVBoxLayout(left_panel)
        lv.addWidget(left_box)
        lv.addWidget(ctrl_box)
        lv.addWidget(asset_box)
        lv.addWidget(out_box)
        lv.addWidget(self.log)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_box)
        splitter.setSizes([700, 780])

        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())

    def _connect_signals(self):
        # 菜单
        self.act_connect.triggered.connect(self.start_camera)
        self.act_disconnect.triggered.connect(self.stop_camera)
        self.act_roi.triggered.connect(lambda: self._run_script("cal_roi.py"))
        self.act_draw.triggered.connect(lambda: self._run_script("draw_centerline.py"))
        self.act_tilt.triggered.connect(lambda: self._run_script("my_calibrate.py"))
        self.act_handeye.triggered.connect(lambda: self._run_script("calibrate_hand_eye.py"))
        self.act_detect.triggered.connect(lambda: self._run_script("detect_simplified.py"))
        self.act_readme.triggered.connect(self._open_readme)

        # 控件
        self.btn_process.clicked.connect(self.on_process_clicked)
        self.btn_export.clicked.connect(self.on_export_clicked)
        self.btn_reload_assets.clicked.connect(self._reload_assets)

    # ---------- 设备 ----------
    def start_camera(self):
        if hasattr(self, "cap_thread") and self.cap_thread.isRunning():
            self._log("相机已连接。")
            return
        self.cap_thread = PercipioCaptureThread(self)
        self.cap_thread.frameReady.connect(self._on_frame)
        self.cap_thread.status.connect(self._log)
        self.cap_thread.startedOk.connect(lambda ok: self.statusBar().showMessage("已连接" if ok else "连接失败"))
        self.cap_thread.start()

    def stop_camera(self):
        if hasattr(self, "cap_thread"):
            self.cap_thread.stop()
        self.statusBar().showMessage("已断开")

    def _on_frame(self, depth_bgr, color_bgr, p3d):
        self.depth_last = depth_bgr
        self.color_last = color_bgr
        self.p3d_last = p3d
        self.depth_shape_hw = depth_bgr.shape[:2]

        self.lbl_depth.setPixmap(np_to_qpixmap(depth_bgr))
        self.lbl_color.setPixmap(np_to_qpixmap(color_bgr))

    # ---------- 处理 ----------
    def _get_current_roi(self) -> Optional[Tuple[int, int, int, int]]:
        if self.depth_last is None:
            return None
        roi = self.lbl_depth.get_roi_on_image(self.depth_last.shape[:2])
        return roi

    def on_process_clicked(self):
        if self.depth_last is None or self.p3d_last is None:
            QMessageBox.warning(self, "提示", "还没有相机帧。请先连接设备。")
            return
        roi = self._get_current_roi()
        if roi is None:
            QMessageBox.warning(self, "提示", "ROI 为空，请在左侧“深度渲染”区域拖拽选择。")
            return
        try:
            res: ProcessingResult = self.pipeline.process_frame(
                p3d_nparray=self.p3d_last,
                depth_render_bgr=self.depth_last,
                roi_xyxy=roi,
                num_key_points=self.ed_num_keypoints.value(),
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "处理失败", str(e))
            return

        # 可视化结果
        if res.surface_mask is not None:
            self.lbl_mask.setPixmap(np_to_qpixmap(res.surface_mask))
        if res.skeleton_overlay is not None:
            self.lbl_skel_overlay.setPixmap(np_to_qpixmap(res.skeleton_overlay))
        if res.comparison_with_vectors is not None:
            self.lbl_compare.setPixmap(np_to_qpixmap(res.comparison_with_vectors))

        # 偏移输出
        if res.avg_pixel_offset is not None:
            dx, dy = res.avg_pixel_offset
            self.lab_px.setText(f"平均像素偏移: ({dx:.2f}, {dy:.2f}) px")
        else:
            self.lab_px.setText("平均像素偏移: -")
        if res.avg_mm_offset is not None:
            mx, my = res.avg_mm_offset
            self.lab_mm.setText(f"平均物理偏移: ({mx:.3f}, {my:.3f}) mm")
        else:
            self.lab_mm.setText("平均物理偏移: -")

        # 日志
        self._log(res.msg or "处理完成")

    def on_export_clicked(self):
        if not self.pipeline.has_result():
            QMessageBox.information(self, "提示", "暂无可导出的检测结果。请先点击“处理选中 ROI”。")
            return
        default = os.path.join(os.getcwd(), time.strftime("centerline_offsets_%Y%m%d_%H%M%S.csv"))
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", default, "CSV (*.csv)")
        if not path:
            return
        try:
            self.pipeline.export_csv(path)
            QMessageBox.information(self, "导出成功", f"CSV 已导出：\n{path}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", str(e))

    # ---------- 资产 ----------
    def _reload_assets(self):
        self.pipeline.load_assets()
        self._refresh_asset_labels()
        self._log("参数重载完成。")

    def _refresh_asset_labels(self):
        ok, tip = self.pipeline.theory_ok, "theoretical_centerline.npy"
        self.lab_theory.setText(f"{tip}: {'已加载' if ok else '未找到'}")
        ok, tip = self.pipeline.tilt_ok, "tilt_correction_matrix.npy"
        self.lab_tilt.setText(f"{tip}: {'已加载' if ok else '未找到'}")
        ok, tip = self.pipeline.handeye_ok, "hand_eye_transform.npy"
        self.lab_handeye.setText(f"{tip}: {'已加载' if ok else '未找到'}")

    # ---------- 其他 ----------
    def _run_script(self, filename: str):
        # 保留原脚本交互/注释，通过独立进程运行
        script_path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "找不到脚本", f"{filename} 不存在。")
            return
        try:
            # 使用与当前 Python 相同的解释器
            subprocess.Popen([sys.executable, script_path])
            self._log(f"已启动脚本：{filename}")
        except Exception as e:
            QMessageBox.critical(self, "启动失败", str(e))

    def _open_readme(self):
        path = os.path.join(os.getcwd(), "README.md")
        if os.path.exists(path):
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore
            elif sys.platform == "darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
        else:
            QMessageBox.information(self, "README 缺失", "项目目录下未找到 README.md")

    def _log(self, text: str):
        self.log.setText(text)
        print(text)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
