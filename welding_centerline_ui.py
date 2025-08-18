import sys, os, threading
from typing import Optional, Tuple
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget, QGroupBox, QSpinBox, QDoubleSpinBox,
    QFormLayout, QFrame
)
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QPixmap, QImage, QAction

import ui_pipeline as pipe

# 将 numpy 图像(BGR或灰度) 转为 QPixmap，用于 QLabel 显示
def to_qpix(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w, c = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w*c, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class RubberBandLabel(QLabel):
    """可在 QLabel 上用鼠标拖拽绘制矩形 ROI。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.origin: Optional[QPoint] = None
        self.current: Optional[QPoint] = None
        self.roi: Optional[QRect] = None
        self.drawing = False

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and not self.pixmap().isNull():
            self.origin = e.position().toPoint()
            self.current = self.origin
            self.drawing = True
            self.update()

    def mouseMoveEvent(self, e):
        if self.drawing:
            self.current = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.origin and self.current:
                self.roi = QRect(self.origin, self.current).normalized()
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.drawing and self.origin and self.current:
            from PySide6.QtGui import QPainter, QPen
            p = QPainter(self)
            p.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            p.drawRect(QRect(self.origin, self.current))
            p.end()

    def get_roi_xyxy(self) -> Optional[Tuple[int,int,int,int]]:
        if not self.roi:
            return None
        # 将 QLabel 的坐标映射为像素坐标（图像铺满 label，无缩放拉伸时可直接用）
        r = self.roi
        return r.left(), r.top(), r.right(), r.bottom()

    def clear_roi(self):
        self.roi = None
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("摩擦焊中轴线提取与偏移计算")
        self.resize(1400, 900)

        self.snapshot: Optional[pipe.CameraSnapshot] = None

        # 顶部菜单动作（外部流程）
        act_handeye = QAction("启动手眼标定（外部）", self)
        act_handeye.triggered.connect(self.launch_handeye)
        act_plane = QAction("启动平面校准（外部）", self)
        act_plane.triggered.connect(self.launch_plane)
        act_theory = QAction("启动理论中轴线工具（外部）", self)
        act_theory.triggered.connect(self.launch_theory)

        menu = self.menuBar().addMenu("工具")
        menu.addAction(act_handeye)
        menu.addAction(act_plane)
        menu.addAction(act_theory)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.init_detect_tab()

    def init_detect_tab(self):
        page = QWidget()
        main_layout = QHBoxLayout(page)

        # 左侧：主流程
        left = QVBoxLayout()
        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("连接并抓取一帧")
        self.btn_connect.clicked.connect(self.on_capture)
        btn_row.addWidget(self.btn_connect)

        self.btn_calc = QPushButton("计算中轴线与偏移")
        self.btn_calc.setEnabled(False)
        self.btn_calc.clicked.connect(self.on_compute)
        btn_row.addWidget(self.btn_calc)

        left.addLayout(btn_row)

        # 处理参数
        param_box = QGroupBox("处理参数")
        form = QFormLayout(param_box)
        self.sp_depth_margin = QDoubleSpinBox()
        self.sp_depth_margin.setRange(0.1, 20.0); self.sp_depth_margin.setSingleStep(0.1); self.sp_depth_margin.setValue(3.5)
        self.sp_pixel_size = QDoubleSpinBox()
        self.sp_pixel_size.setRange(0.05, 5.0); self.sp_pixel_size.setSingleStep(0.05); self.sp_pixel_size.setValue(0.5)
        self.sp_keypoints = QSpinBox(); self.sp_keypoints.setRange(4, 100); self.sp_keypoints.setValue(12)
        form.addRow("深度容差(mm)：", self.sp_depth_margin)
        form.addRow("正交投影像素尺寸(mm/px)：", self.sp_pixel_size)
        form.addRow("关键点数量：", self.sp_keypoints)
        left.addWidget(param_box)

        # 深度渲染 + 选择 ROI
        self.lbl_depth = RubberBandLabel()
        self.lbl_depth.setFrameShape(QFrame.Box)
        self.lbl_depth.setAlignment(Qt.AlignCenter)
        left.addWidget(QLabel("深度渲染 - 在此拖拽选择 ROI"))
        left.addWidget(self.lbl_depth, 4)

        # 结果与导出
        result_box = QGroupBox("结果")
        grid = QGridLayout(result_box)
        self.lbl_mask = QLabel(); self.lbl_mask.setAlignment(Qt.AlignCenter); self.lbl_mask.setFrameShape(QFrame.Box)
        self.lbl_skel = QLabel(); self.lbl_skel.setAlignment(Qt.AlignCenter); self.lbl_skel.setFrameShape(QFrame.Box)
        self.lbl_comp = QLabel(); self.lbl_comp.setAlignment(Qt.AlignCenter); self.lbl_comp.setFrameShape(QFrame.Box)
        self.lbl_final = QLabel(); self.lbl_final.setAlignment(Qt.AlignCenter); self.lbl_final.setFrameShape(QFrame.Box)

        grid.addWidget(QLabel("表面掩码"), 0,0); grid.addWidget(self.lbl_mask, 1,0)
        grid.addWidget(QLabel("骨架（中轴线）"), 0,1); grid.addWidget(self.lbl_skel, 1,1)
        grid.addWidget(QLabel("中轴线比较"), 2,0); grid.addWidget(self.lbl_comp, 3,0)
        grid.addWidget(QLabel("偏移箭头"), 2,1); grid.addWidget(self.lbl_final, 3,1)
        left.addWidget(result_box, 6)

        # 导出按钮
        export_row = QHBoxLayout()
        self.btn_export = QPushButton("导出偏移 CSV")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.on_export)
        export_row.addWidget(self.btn_export)
        left.addLayout(export_row)

        # 右侧：指标
        right = QGroupBox("指标")
        rlay = QFormLayout(right)
        self.lbl_match = QLabel("-")
        self.lbl_avg_px = QLabel("-")
        self.lbl_avg_mm = QLabel("-")
        rlay.addRow("匹配度(%)：", self.lbl_match)
        rlay.addRow("平均像素偏移 (dx, dy)：", self.lbl_avg_px)
        rlay.addRow("平均物理偏移 (dx, dy) mm：", self.lbl_avg_mm)

        main_layout.addLayout(left, 3)
        main_layout.addWidget(right, 1)

        self.tabs.addTab(page, "检测与比较")

    def launch_handeye(self):
        # 直接调用原脚本流程（保留其交互与注释）
        try:
            import calibrate_hand_eye as mod
            threading.Thread(target=mod.main, daemon=True).start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法启动 calibrate_hand_eye.py：\n{e}")

    def launch_plane(self):
        try:
            import my_calibrate as mod
            threading.Thread(target=mod.main, daemon=True).start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法启动 my_calibrate.py：\n{e}")

    def launch_theory(self):
        try:
            import draw_centerline as mod
            threading.Thread(target=mod.draw_centerline, daemon=True).start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法启动 draw_centerline.py：\n{e}")

    def on_capture(self):
        self.btn_connect.setEnabled(False)
        def task():
            try:
                self.snapshot = pipe.capture_one_frame()
                pix = to_qpix(self.snapshot.depth_render_bgr)
                self.lbl_depth.setPixmap(pix.scaled(self.lbl_depth.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.btn_calc.setEnabled(True)
                QMessageBox.information(self, "成功", "已抓取一帧数据。请拖拽选择 ROI。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"抓取失败：\n{e}")
            finally:
                self.btn_connect.setEnabled(True)
        threading.Thread(target=task, daemon=True).start()

    def on_compute(self):
        if self.snapshot is None:
            QMessageBox.warning(self, "提示", "请先抓取一帧。")
            return
        roi = self.lbl_depth.get_roi_xyxy()
        if roi is None:
            QMessageBox.warning(self, "提示", "请在深度图上拖拽选择 ROI。")
            return

        self.btn_calc.setEnabled(False)
        def task():
            try:
                out = pipe.process_roi(
                    self.snapshot,
                    roi,
                    pixel_size_mm=self.sp_pixel_size.value(),
                    depth_margin=self.sp_depth_margin.value(),
                    keypoints=int(self.sp_keypoints.value())
                )
                # 显示图像
                self.lbl_mask.setPixmap(to_qpix(out["surface_mask"]).scaled(self.lbl_mask.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.lbl_skel.setPixmap(to_qpix(out["skeleton"]).scaled(self.lbl_skel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.lbl_comp.setPixmap(to_qpix(out["comparison"]).scaled(self.lbl_comp.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.lbl_final.setPixmap(to_qpix(out["final"]).scaled(self.lbl_final.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                # 指标
                self.lbl_match.setText(f"{out['match_score']*100:.2f}")
                if out['avg_pixel_offset'] is not None:
                    dx, dy = out['avg_pixel_offset']
                    self.lbl_avg_px.setText(f"({dx:.2f}, {dy:.2f})")
                else:
                    self.lbl_avg_px.setText("-")
                if out['avg_physical_offset'] is not None:
                    dx, dy = out['avg_physical_offset']
                    self.lbl_avg_mm.setText(f"({dx:.3f}, {dy:.3f})")
                else:
                    self.lbl_avg_mm.setText("-")

                # 保存以便导出
                self._last_deviation = out['deviation_vectors']
                self._last_avg_px = out['avg_pixel_offset']
                self._last_avg_mm = out['avg_physical_offset']
                self.btn_export.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"计算失败：\n{e}")
            finally:
                self.btn_calc.setEnabled(True)
        threading.Thread(target=task, daemon=True).start()

    def on_export(self):
        if not hasattr(self, "_last_deviation") or self._last_deviation is None:
            QMessageBox.information(self, "提示", "暂无可导出的结果。")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "保存 CSV", "deviation_vectors.csv", "CSV (*.csv)")
        if not fn:
            return
        try:
            pipe.export_deviation_csv(fn, self._last_deviation, self._last_avg_px, self._last_avg_mm)
            QMessageBox.information(self, "完成", "已导出。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
