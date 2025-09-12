# -*- coding: utf-8 -*-
"""
PyQt5 GUI 入口：
- 左侧参数（带简短说明/tooltip），右侧图像预览。
- 支持导入 G 代码、选择 roi_mode、遮挡设置、偏差补偿、导出偏移和 G 代码。
- 程序启动时预先启动相机（后台线程），点击"预览单帧"即可直接取图处理。
运行：
    python -m gui.main
或：
    python gui/main.py
"""
from __future__ import annotations
import os, sys, json, threading
from typing import Optional

# 添加当前目录到路径，确保能找到controller模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox,
    QComboBox, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, QTextEdit,
    QSplitter, QTabWidget
)
from PyQt5.QtGui import QPixmap

# 本地导入 - 改为绝对导入
from controller import AlignController, GUIConfig, np_to_qimage

# ---- 后台线程：相机预启动 ----
class CameraBootThread(QThread):
    status_changed = pyqtSignal(str)
    def __init__(self, ctrl: AlignController):
        super().__init__()
        self.ctrl = ctrl
    def run(self):
        msg = self.ctrl.start_camera()
        self.status_changed.emit(msg)

# ---- 后台线程：单帧处理 ----
class ProcessThread(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    def __init__(self, ctrl: AlignController):
        super().__init__()
        self.ctrl = ctrl
    def run(self):
        try:
            out = self.ctrl.process_single_frame()
            self.finished_ok.emit(out)
        except Exception as e:
            self.failed.emit(str(e))

# ---- 主窗口 ----
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guided Centerline — GUI")
        self.resize(1280, 800)

        self.ctrl = AlignController(GUIConfig())

        # 预启相机
        self.statusBar().showMessage("正在启动相机…")
        self.cam_thread = CameraBootThread(self.ctrl)
        self.cam_thread.status_changed.connect(self.on_camera_status)
        self.cam_thread.start()

        # 布局
        root = QSplitter()
        root.setOrientation(Qt.Horizontal)
        root.setChildrenCollapsible(False)
        self.setCentralWidget(root)

        # 左侧：参数
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(8,8,8,8); lv.setSpacing(8)
        root.addWidget(left)

        # 文件组
        g_file = QGroupBox("文件 / 连接")
        f = QFormLayout(g_file)
        self.ed_T = QLineEdit(self.ctrl.cfg.T_path); self.ed_T.setToolTip("相机→机床外参 .npy（包含 R,t,T）")
        self.btn_T = QPushButton("浏览…"); self.btn_T.clicked.connect(self.pick_T)
        row = QHBoxLayout(); row.addWidget(self.ed_T); row.addWidget(self.btn_T)
        f.addRow("T_cam2machine:", row)

        self.ed_g = QLineEdit(self.ctrl.cfg.gcode_path); self.ed_g.setToolTip("理论 G 代码路径（支持 G0/G1/G2/G3）")
        self.btn_g = QPushButton("浏览…"); self.btn_g.clicked.connect(self.pick_gcode)
        row2 = QHBoxLayout(); row2.addWidget(self.ed_g); row2.addWidget(self.btn_g)
        f.addRow("G-code:", row2)

        self.chk_bias = QCheckBox("启用偏差补偿 (bias_comp.json)")
        self.chk_bias.setChecked(self.ctrl.cfg.bias_enable)
        self.ed_bias = QLineEdit(self.ctrl.cfg.bias_path)
        self.btn_bias = QPushButton("浏览…"); self.btn_bias.clicked.connect(self.pick_bias)
        row3 = QHBoxLayout(); row3.addWidget(self.ed_bias); row3.addWidget(self.btn_bias)
        f.addRow(self.chk_bias)
        f.addRow("bias_comp:", row3)
        lv.addWidget(g_file)

        # ROI 组
        g_roi = QGroupBox("ROI / 投影")
        fr = QFormLayout(g_roi)
        self.cmb_roi = QComboBox(); self.cmb_roi.addItems(["none","camera_rect","machine","gcode_bounds"])
        self.cmb_roi.setCurrentText(self.ctrl.cfg.roi_mode)
        fr.addRow("roi_mode:", self.cmb_roi)
        self.ed_xywh = QLineEdit("%d,%d,%d,%d"%self.ctrl.cfg.cam_roi_xywh); self.ed_xywh.setToolTip("当 roi_mode=camera_rect 时生效：x,y,w,h 像素")
        fr.addRow("cam_roi_xywh:", self.ed_xywh)
        self.ed_center = QLineEdit("%.3f,%.3f"%self.ctrl.cfg.roi_center_xy); self.ed_center.setToolTip("当 roi_mode=machine 时生效：ROI 中心 (mm)")
        fr.addRow("roi_center_xy:", self.ed_center)
        self.spn_size = QDoubleSpinBox(); self.spn_size.setRange(1, 5000); self.spn_size.setValue(self.ctrl.cfg.roi_size_mm)
        fr.addRow("roi_size_mm:", self.spn_size)
        self.spn_margin = QDoubleSpinBox(); self.spn_margin.setRange(0, 1000); self.spn_margin.setValue(self.ctrl.cfg.bounds_margin_mm)
        fr.addRow("bounds_margin_mm:", self.spn_margin)
        self.spn_pix = QDoubleSpinBox(); self.spn_pix.setRange(0.05, 10.0); self.spn_pix.setDecimals(3); self.spn_pix.setValue(self.ctrl.cfg.pixel_size_mm)
        fr.addRow("pixel_size_mm:", self.spn_pix)
        lv.addWidget(g_roi)

        # 遮挡组
        g_occ = QGroupBox("遮挡 (固定设备区域)")
        fo = QFormLayout(g_occ)
        self.chk_occ = QCheckBox("启用遮挡区域"); self.chk_occ.setChecked(self.ctrl.cfg.occ_enable)
        fo.addRow(self.chk_occ)
        self.ed_polys = QTextEdit()
        self.ed_polys.setPlaceholderText("每行一个多边形：x1,y1; x2,y2; ...  例如:\n-50,-50; 30,-30; 30,200; -50,200")
        # 默认一行
        def_poly = "; ".join([f"{x},{y}" for (x,y) in self.ctrl.cfg.occ_polys[0]])
        self.ed_polys.setPlainText(def_poly)
        self.spn_dilate = QDoubleSpinBox(); self.spn_dilate.setRange(0, 50); self.spn_dilate.setValue(self.ctrl.cfg.occ_dilate_mm)
        self.chk_band = QCheckBox("遮挡内按 G 代码合成环带掩码"); self.chk_band.setChecked(self.ctrl.cfg.occ_synthesize_band)
        self.spn_band = QDoubleSpinBox(); self.spn_band.setRange(0, 100); self.spn_band.setDecimals(2); self.spn_band.setValue(self.ctrl.cfg.occ_band_halfwidth_mm or 0.0)
        self.spn_band.setToolTip("若为 0 则自动从可见区估计半宽；否则用此值(mm)")
        fo.addRow("polys(mm):", self.ed_polys)
        fo.addRow("dilate_mm:", self.spn_dilate)
        fo.addRow(self.chk_band)
        fo.addRow("band_halfwidth_mm:", self.spn_band)
        lv.addWidget(g_occ)

        # 引导/偏移组
        g_guide = QGroupBox("引导中心线 / 偏移")
        fg = QFormLayout(g_guide)
        self.spn_step = QDoubleSpinBox(); self.spn_step.setRange(0.1, 10.0); self.spn_step.setDecimals(2); self.spn_step.setValue(self.ctrl.cfg.guide_step_mm)
        fg.addRow("guide_step_mm:", self.spn_step)
        self.spn_half = QDoubleSpinBox(); self.spn_half.setRange(0.5, 50.0); self.spn_half.setValue(self.ctrl.cfg.guide_halfwidth_mm)
        fg.addRow("guide_halfwidth_mm:", self.spn_half)
        self.spn_win = QSpinBox(); self.spn_win.setRange(1, 99); self.spn_win.setValue(self.ctrl.cfg.guide_smooth_win)
        fg.addRow("guide_smooth_win:", self.spn_win)
        self.spn_maxoff = QDoubleSpinBox(); self.spn_maxoff.setRange(0.1, 50.0); self.spn_maxoff.setValue(self.ctrl.cfg.guide_max_offset_mm)
        fg.addRow("guide_max_offset_mm:", self.spn_maxoff)
        self.spn_grad = QDoubleSpinBox(); self.spn_grad.setRange(0.001, 1.0); self.spn_grad.setDecimals(3); self.spn_grad.setValue(self.ctrl.cfg.guide_max_grad_mm_per_mm)
        fg.addRow("guide_max_grad (mm/mm):", self.spn_grad)
        self.spn_gap = QSpinBox(); self.spn_gap.setRange(1, 100); self.spn_gap.setValue(self.ctrl.cfg.max_gap_pts)
        fg.addRow("max_gap_pts:", self.spn_gap)
        lv.addWidget(g_guide)

        # Guard & 输出
        g_out = QGroupBox("Guard & 导出")
        fo2 = QFormLayout(g_out)
        self.spn_vr = QDoubleSpinBox(); self.spn_vr.setRange(0.0, 1.0); self.spn_vr.setSingleStep(0.05); self.spn_vr.setValue(self.ctrl.cfg.guard_min_valid_ratio)
        fo2.addRow("min_valid_ratio:", self.spn_vr)
        self.spn_p95 = QDoubleSpinBox(); self.spn_p95.setRange(0.1, 50.0); self.spn_p95.setValue(self.ctrl.cfg.guard_max_abs_p95_mm)
        fo2.addRow("max_abs_p95_mm:", self.spn_p95)
        self.ed_outdir = QLineEdit(self.ctrl.cfg.out_dir)
        fo2.addRow("out_dir:", self.ed_outdir)
        self.ed_csv = QLineEdit(self.ctrl.cfg.offset_csv)
        fo2.addRow("offset_csv:", self.ed_csv)
        self.ed_gc = QLineEdit(self.ctrl.cfg.corrected_gcode)
        fo2.addRow("corrected_gcode:", self.ed_gc)
        lv.addWidget(g_out)

        # 操作按钮
        btns = QHBoxLayout()
        self.btn_preview = QPushButton("预览单帧")
        self.btn_export  = QPushButton("导出纠偏 (CSV + GCode)")
        self.btn_bias    = QPushButton("保存 BiasComp (当前帧)")
        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_bias.clicked.connect(self.on_save_bias)
        btns.addWidget(self.btn_preview); btns.addWidget(self.btn_export); btns.addWidget(self.btn_bias)
        lv.addLayout(btns)
        lv.addStretch(1)

        # 右侧：图像
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(8,8,8,8); rv.setSpacing(6)
        root.addWidget(right)

        self.tabs = QTabWidget()
        self.lbl_vis = QLabel("预览图"); self.lbl_vis.setAlignment(Qt.AlignCenter); self.lbl_vis.setMinimumHeight(360)
        self.lbl_probe = QLabel("Normal-Probes"); self.lbl_probe.setAlignment(Qt.AlignCenter); self.lbl_probe.setMinimumHeight(240)
        self.lbl_hist = QLabel("Bias Δn 直方图"); self.lbl_hist.setAlignment(Qt.AlignCenter); self.lbl_hist.setMinimumHeight(240)
        w1 = QWidget(); v1 = QVBoxLayout(w1); v1.addWidget(self.lbl_vis)
        w2 = QWidget(); v2 = QVBoxLayout(w2); v2.addWidget(self.lbl_probe)
        w3 = QWidget(); v3 = QVBoxLayout(w3); v3.addWidget(self.lbl_hist)
        self.tabs.addTab(w1, "对齐叠加")
        self.tabs.addTab(w2, "法线采样")
        self.tabs.addTab(w3, "Bias 直方图")
        rv.addWidget(self.tabs, 1)

        # 使用提示
        tips = QLabel(
            "<b>提示：</b> 先确认 T_cam2machine 与 G-code 路径，等待状态栏显示“camera_ready”。\n"
            "点击 <i>预览单帧</i> 进行处理；若 Guard=PASS，再点击 <i>导出纠偏</i> 生成 CSV 与 corrected.gcode。"
        )
        tips.setWordWrap(True)
        rv.addWidget(tips)

        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)

    # ---- 文件选择 ----
    def pick_T(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 T_cam2machine.npy", "", "NumPy NPY (*.npy)")
        if p:
            self.ed_T.setText(p)
    def pick_gcode(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 G-code", "", "G-code (*.gcode *.nc *.tap *.ngc *.txt);;All files (*)")
        if p:
            self.ed_g.setText(p)
    def pick_bias(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择 bias_comp.json", "", "JSON (*.json)")
        if p:
            self.ed_bias.setText(p)

    # ---- 将控件值写回配置 ----
    def flush_cfg(self):
        c = self.ctrl.cfg
        c.T_path = self.ed_T.text().strip()
        c.gcode_path = self.ed_g.text().strip()
        c.bias_enable = self.chk_bias.isChecked()
        c.bias_path = self.ed_bias.text().strip()
        c.roi_mode = self.cmb_roi.currentText()
        try:
            x,y,w,h = [int(v) for v in self.ed_xywh.text().replace(' ','').split(',')[:4]]
            c.cam_roi_xywh = (x,y,w,h)
        except Exception:
            pass
        try:
            cx,cy = [float(v) for v in self.ed_center.text().replace(' ','').split(',')[:2]]
            c.roi_center_xy = (cx,cy)
        except Exception:
            pass
        c.roi_size_mm = float(self.spn_size.value())
        c.bounds_margin_mm = float(self.spn_margin.value())
        c.pixel_size_mm = float(self.spn_pix.value())
        # occ
        c.occ_enable = self.chk_occ.isChecked()
        c.occ_dilate_mm = float(self.spn_dilate.value())
        c.occ_synthesize_band = self.chk_band.isChecked()
        bh = float(self.spn_band.value())
        c.occ_band_halfwidth_mm = None if bh == 0.0 else bh
        polys: list = []
        for line in self.ed_polys.toPlainText().splitlines():
            line = line.strip()
            if not line: continue
            pts = []
            for seg in line.split(';'):
                seg = seg.strip()
                if not seg: continue
                x_str, y_str = seg.split(',')[:2]
                pts.append((float(x_str), float(y_str)))
            if len(pts) >= 3:
                polys.append(pts)
        if polys:
            c.occ_polys = polys
        # guide
        c.guide_step_mm = float(self.spn_step.value())
        c.guide_halfwidth_mm = float(self.spn_half.value())
        c.guide_smooth_win = int(self.spn_win.value())
        c.guide_max_offset_mm = float(self.spn_maxoff.value())
        c.guide_max_grad_mm_per_mm = float(self.spn_grad.value())
        c.max_gap_pts = int(self.spn_gap.value())
        # guard
        c.guard_min_valid_ratio = float(self.spn_vr.value())
        c.guard_max_abs_p95_mm = float(self.spn_p95.value())
        c.out_dir = self.ed_outdir.text().strip()
        c.offset_csv = self.ed_csv.text().strip()
        c.corrected_gcode = self.ed_gc.text().strip()

    # ---- 槽：相机状态 ----
    def on_camera_status(self, msg: str):
        if msg == "camera_ready":
            self.statusBar().showMessage("camera_ready")
        else:
            self.statusBar().showMessage(msg)
            if msg.startswith("camera_error"):
                QMessageBox.critical(self, "相机错误", msg)

    # ---- 槽：预览 ----
    def on_preview(self):
        self.flush_cfg()
        self.statusBar().showMessage("处理中…")
        self.btn_preview.setEnabled(False)
        self.proc_thread = ProcessThread(self.ctrl)
        self.proc_thread.finished_ok.connect(self.on_preview_done)
        self.proc_thread.failed.connect(self.on_preview_failed)
        self.proc_thread.start()

    def on_preview_done(self, out: dict):
        self.btn_preview.setEnabled(True)
        self.statusBar().showMessage("预览完成")
        # 更新三张图
        for (key, label) in [("vis_cmp", self.lbl_vis), ("vis_probe", self.lbl_probe), ("hist_panel", self.lbl_hist)]:
            img = out.get(key)
            if img is not None:
                q = np_to_qimage(img)
                if q is not None:
                    label.setPixmap(QPixmap.fromImage(q).scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    label.setText("(无法显示图像)")
            else:
                label.setText("(无图像)")

    def on_preview_failed(self, err: str):
        self.btn_preview.setEnabled(True)
        self.statusBar().showMessage("预览失败")
        QMessageBox.warning(self, "预览失败", err)

    # ---- 槽：导出 ----
    def on_export(self):
        try:
            self.flush_cfg()
            res = self.ctrl.export_corrected()
            self.statusBar().showMessage("导出完成")
            QMessageBox.information(self, "导出完成", f"CSV: {res['offset_csv']}\nGCode: {res['corrected_gcode']}")
        except Exception as e:
            self.statusBar().showMessage("导出失败")
            QMessageBox.warning(self, "导出失败", str(e))

    # ---- 槽：保存 bias_comp ----
    def on_save_bias(self):
        try:
            self.flush_cfg()
            p = self.ctrl.save_bias_from_current()
            QMessageBox.information(self, "已保存", f"bias_comp 已写入:\n{p}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

# ---- 入口 ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
