# -*- coding: utf-8 -*-
"""
Welding Centerline UI (PySide6)

修复：
- ROI 映射精确化：考虑等比缩放与居中留黑边(offset)，避免框选错位。
- 窗体过大：所有图像控件等比缩放 + 右侧结果区滚动容器，防止被大图撑大。

保持：
- 不更动既有脚本的注释/函数，仍以 import/子进程复用。
- 全链路可视化与导出逻辑不变。
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
    QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QGridLayout, QStatusBar, QSplitter, QSpinBox, QScrollArea
)

# ---- Percipio SDK ----
try:
    import pcammls
    from pcammls import *
except Exception as e:
    pcammls = None
    print("[WARN] Percipio SDK not available in this environment:", e)

# ---- detect 模块导入（不改动原文件） ----
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
    # fallback
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
        super().resizeEvent(e)
        self._rescale_and_set()

    def _rescale_and_set(self):
        if self._orig_pm is None or self.width() <= 2 or self.height() <= 2:
            self.setPixmap(QPixmap())
            return
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
        self._display_rect = QRect()   # 实际显示的 pixmap 的矩形（相对控件坐标）
        self._orig_size = QSize()      # 原始图像尺寸

    # 覆盖 set_image，保留原图尺寸
    def set_image(self, img):
        if isinstance(img, np.ndarray):
            pm = np_to_qpixmap(img)
            self._orig_size = QSize(pm.width(), pm.height())
            self._orig_pm = pm
        elif isinstance(img, QPixmap):
            self._orig_size = QSize(img.width(), img.height())
            self._orig_pm = img
        else:
            self._orig_pm = None
            self._orig_size = QSize()
        self._rescale_and_set()  # 内部会设置 _display_rect

    def _rescale_and_set(self):
        if self._orig_pm is None or self.width() <= 2 or self.height() <= 2:
            self.setPixmap(QPixmap())
            self._display_rect = QRect()
            return
        scaled = self._orig_pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_pm = scaled
        # 计算显示矩形：居中
        xoff = (self.width() - scaled.width()) // 2
        yoff = (self.height() - scaled.height()) // 2
        self._display_rect = QRect(xoff, yoff, scaled.width(), scaled.height())
        self.setPixmap(scaled)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._rect.isNull():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            # 限制只在显示区域内描绘框，观感更准确
            roi_draw = self._rect & self._display_rect
            painter.drawRect(roi_draw if not roi_draw.isNull() else self._rect)

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

    def get_roi_on_image(self, _ignored_shape=None) -> Optional[Tuple[int, int, int, int]]:
        """
        将当前控件坐标下的 ROI 映射到原始图像像素坐标 (x1,y1,x2,y2)。
        考虑显示区域偏移(_display_rect)与缩放比。
        """
        if self._orig_size.isEmpty() or self._rect.isNull() or self._display_rect.isNull():
            return None

        # 与显示区域求交，避免越界
        roi = self._rect & self._display_rect
        if roi.isNull():
            return None

        # 缩放比：显示像素 -> 原图像素
        sx = self._orig_size.width() / max(1, self._display_rect.width())
        sy = self._orig_size.height() / max(1, self._display_rect.height())

        # 去掉显示区域的偏移（居中）
        def map_pt(p: QPoint):
            x = (p.x() - self._display_rect.left()) * sx
            y = (p.y() - self._display_rect.top()) * sy
            return int(np.clip(x, 0, self._orig_size.width()-1)), int(np.clip(y, 0, self._orig_size.height()-1))

        x1, y1 = map_pt(roi.topLeft())
        x2, y2 = map_pt(roi.bottomRight())
        # 归一化
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

# ---- 取流线程 ----
class PercipioCaptureThread(QThread):
    frameReady = Signal(object, object, object)  # depth_render_bgr, color_bgr, p3d_nparray
    status = Signal(str)
    startedOk = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False

    def run(self):
        if pcammls is None:
            self.status.emit("Percipio SDK 不可用，无法连接相机。")
            self.startedOk.emit(False)
            return

        try:
            cl = pcammls.PercipioSDK()
            dev_list = cl.ListDevice()
            if not dev_list:
                self.status.emit("未找到相机设备。")
                self.startedOk.emit(False)
                return
            handle = cl.Open(dev_list[0].id)
            if not cl.isValidHandle(handle):
                self.status.emit("打开设备失败。")
                self.startedOk.emit(False)
                return

            err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
            if err:
                self.status.emit(f"启用数据流失败: {err}")
                self.startedOk.emit(False)
                return

            cl.DeviceLoadDefaultParameters(handle)
            scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
            depth_calib_data = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)

            cl.DeviceStreamOn(handle)
            self._running = True
            self.startedOk.emit(True)

            img_render = pcammls.image_data()
            img_color = pcammls.image_data()
            pointcloud_data_arr = pcammls.pointcloud_data_list()

            last_emit = time.time()
            while self._running:
                image_list = cl.DeviceStreamRead(handle, 2000)
                if not image_list:
                    continue

                depth_frame = None
                color_frame = None
                for frame in image_list:
                    if frame.streamID == PERCIPIO_STREAM_DEPTH:
                        depth_frame = frame
                    elif frame.streamID == PERCIPIO_STREAM_COLOR:
                        color_frame = frame

                if depth_frame is None or color_frame is None:
                    continue

                cl.DeviceStreamDepthRender(depth_frame, img_render)
                depth_bgr = img_render.as_nparray().copy()

                cl.DeviceStreamImageDecode(color_frame, img_color)
                color_bgr = img_color.as_nparray().copy()

                cl.DeviceStreamMapDepthImageToPoint3D(
                    depth_frame, depth_calib_data, scale_unit, pointcloud_data_arr
                )
                p3d = pointcloud_data_arr.as_nparray().copy()  # (H, W, 3) mm

                now = time.time()
                if now - last_emit > 0.03:
                    self.frameReady.emit(depth_bgr, color_bgr, p3d)
                    last_emit = now

            cl.DeviceStreamOff(handle)
            cl.Close(handle)
        except Exception as e:
            self.status.emit("采集线程异常: " + str(e))
            traceback.print_exc()
            self.startedOk.emit(False)

    def stop(self):
        self._running = False

# ---- 管线 ----
from ui_pipeline import Pipeline, ProcessingResult

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摩擦焊 视觉-运动 一体化 | Centerline GUI")
        self.resize(1280, 800)  # 更合理的初始尺寸

        self.depth_last = None
        self.color_last = None
        self.p3d_last = None

        self.pipeline = Pipeline()

        self._build_ui()
        self._connect_signals()
        self._refresh_asset_labels()

    def _build_ui(self):
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

        # 左侧：预览/ROI
        left_box = QGroupBox("预览 / ROI")
        self.lbl_depth = ROIOverlayLabel(min_h=360)
        self.lbl_color = ScaledImageLabel(min_h=260)

        left_layout = QVBoxLayout(left_box)
        left_layout.addWidget(QLabel("深度渲染"))
        left_layout.addWidget(self.lbl_depth, 4)
        left_layout.addWidget(QLabel("彩色图"))
        left_layout.addWidget(self.lbl_color, 3)

        # 右侧：结果（滚动区域，防止撑大窗口）
        right_content = QWidget()
        right_layout = QGridLayout(right_content)
        right_content.setStyleSheet("background:#0b0b0b;")

        self.lbl_mask = ScaledImageLabel(min_h=240)
        self.lbl_skel_overlay = ScaledImageLabel(min_h=240)
        self.lbl_compare = ScaledImageLabel(min_h=340)

        right_layout.addWidget(QLabel("表面掩码"), 0, 0)
        right_layout.addWidget(self.lbl_mask, 1, 0)
        right_layout.addWidget(QLabel("骨架叠加（深度图）"), 0, 1)
        right_layout.addWidget(self.lbl_skel_overlay, 1, 1)
        right_layout.addWidget(QLabel("中轴线比较（含偏移箭头）"), 2, 0, 1, 2)
        right_layout.addWidget(self.lbl_compare, 3, 0, 1, 2)

        scroll = QScrollArea()
        scroll.setWidget(right_content)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: #111; }")

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

        # 左侧组合
        left_panel = QWidget()
        lv = QVBoxLayout(left_panel)
        lv.addWidget(left_box)
        lv.addWidget(ctrl_box)
        lv.addWidget(asset_box)
        lv.addWidget(out_box)
        lv.addWidget(self.log)

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(scroll)
        splitter.setSizes([600, 680])  # 默认左略小右略大，且整体不会超出窗口

        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())

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

    # ---- 设备 ----
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
        # 使用 set_image（等比缩放 + 内部记录显示矩形），防止撑大窗口并确保 ROI 映射正确
        self.depth_last = depth_bgr
        self.color_last = color_bgr
        self.p3d_last = p3d
        self.lbl_depth.set_image(depth_bgr)
        self.lbl_color.set_image(color_bgr)

    # ---- 处理 ----
    def _get_current_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self.lbl_depth.get_roi_on_image()

    def on_process_clicked(self):
        if self.depth_last is None or self.p3d_last is None:
            QMessageBox.warning(self, "提示", "还没有相机帧。请先连接设备。")
            return
        roi = self._get_current_roi()
        if roi is None:
            QMessageBox.warning(self, "提示", "ROI 为空或越界，请在左侧“深度渲染”区域框选可见区域内的 ROI。")
            return

        from ui_pipeline import Pipeline, ProcessingResult  # 确保最新
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

        # UI 内可视化（等比缩放）
        if res.surface_mask is not None:
            self.lbl_mask.set_image(res.surface_mask)
        else:
            self.lbl_mask.set_image(None)

        if res.skeleton_overlay is not None:
            self.lbl_skel_overlay.set_image(res.skeleton_overlay)
        else:
            self.lbl_skel_overlay.set_image(None)

        if res.comparison_with_vectors is not None:
            self.lbl_compare.set_image(res.comparison_with_vectors)
        else:
            self.lbl_compare.set_image(None)

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

    # ---- 资产 ----
    def _reload_assets(self):
        self.pipeline.load_assets()
        self._refresh_asset_labels()
        self._log("参数重载完成。")

    def _refresh_asset_labels(self):
        self.lab_theory.setText(f"theoretical_centerline.npy: {'已加载' if self.pipeline.theory_ok else '未找到'}")
        self.lab_tilt.setText(f"tilt_correction_matrix.npy: {'已加载' if self.pipeline.tilt_ok else '未找到'}")
        self.lab_handeye.setText(f"hand_eye_transform.npy: {'已加载' if self.pipeline.handeye_ok else '未找到'}")

    # ---- 其他 ----
    def _run_script(self, filename: str):
        script_path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "找不到脚本", f"{filename} 不存在。")
            return
        try:
            subprocess.Popen([sys.executable, script_path])
            self._log(f"已启动脚本：{filename}")
        except Exception:
            QMessageBox.critical(self, "启动失败", f"无法启动：{filename}")

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
