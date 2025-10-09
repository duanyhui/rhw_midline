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
# 顶部 import 区域补充
from PyQt5.QtWidgets import QScrollArea, QSizePolicy, QFrame

# 添加当前目录到路径，确保能找到controller模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
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

        # 左列真实容器：所有参数组都加到 left_body（原来加到 lv 的代码不变）
        left_body = QWidget()
        lv = QVBoxLayout(left_body)
        lv.setContentsMargins(8, 8, 8, 8)
        lv.setSpacing(8)
        # 1) 左栏不要被压扁
        left_body.setMinimumWidth(320)  # 或者 left_scroll.setMinimumWidth(320)

        # 2) 控制分配策略：右侧更“贪”，但左侧保底
        root.setCollapsible(0, False)  # 左右都不可被完全折叠
        root.setCollapsible(1, False)
        root.setStretchFactor(0, 0)  # 左：权重小一些
        root.setStretchFactor(1, 1)  # 右：权重大一些

        # 3) 设定一个合适的初始宽度（需要在窗口显示后再设，避免被布局覆盖）
        self.resize(1360, 860)  # 可按需调整窗口默认大小
        QTimer.singleShot(0, lambda: root.setSizes([380, max(700, self.width() - 380)]))

        #  滚动区包装
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_body)
        left_scroll.setWidgetResizable(True)  # 关键：随窗口变化自适应
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 只竖向滚
        left_scroll.setFrameShape(QFrame.NoFrame)


        # 把滚动区挂到分割器
        # 左栏顶层容器：上=scroll，下=footer
        left_col = QWidget()
        left_col_v = QVBoxLayout(left_col)
        left_col_v.setContentsMargins(0, 0, 0, 0)
        left_col_v.setSpacing(0)

        # 上：滚动区（占满可用空间）
        left_col_v.addWidget(left_scroll, 1)

        # 先把左栏放进分割器（页脚稍后再 add）
        root.addWidget(left_col)

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

        # === Corner ignoring（拐角点附近忽略取点） ===
        self.chk_corner = QCheckBox("拐角忽略 (corner_ignore_enable)")
        self.chk_corner.setChecked(self.ctrl.cfg.corner_ignore_enable)
        self.chk_corner.setToolTip("开启后，在拐角点两侧按弧长范围忽略取点，避免拐角对平滑/限幅产生拖尾影响。")

        self.spn_cang = QDoubleSpinBox()
        self.spn_cang.setRange(1.0, 180.0)
        self.spn_cang.setDecimals(1)
        self.spn_cang.setValue(self.ctrl.cfg.corner_angle_thr_deg)
        self.spn_cang.setToolTip("拐角判定阈值（度）。相邻两段的转角≥该值视为拐角。")

        self.spn_cspan = QDoubleSpinBox()
        self.spn_cspan.setRange(0.0, 50.0)
        self.spn_cspan.setDecimals(2)
        self.spn_cspan.setValue(self.ctrl.cfg.corner_ignore_span_mm)
        self.spn_cspan.setToolTip("在每个拐角两侧各忽略的弧长半径（mm），按 guide_step_mm 转为点数。")

        f.addRow(self.chk_corner)
        f.addRow("corner_angle_thr_deg:", self.spn_cang)
        f.addRow("corner_ignore_span_mm:", self.spn_cspan)

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
        # ====== 展平 / 最近表面 ======
        g_flat = QGroupBox("展平 / 最近表面")
        ff = QFormLayout(g_flat)

        self.chk_plane = QCheckBox("启用平面展平");
        self.chk_plane.setChecked(self.ctrl.cfg.plane_enable)
        self.spn_pransac = QDoubleSpinBox();
        self.spn_pransac.setRange(0.1, 10.0);
        self.spn_pransac.setValue(self.ctrl.cfg.plane_ransac_thresh_mm)
        self.spn_piters = QSpinBox();
        self.spn_piters.setRange(50, 5000);
        self.spn_piters.setValue(self.ctrl.cfg.plane_ransac_iters)
        self.spn_pcap = QSpinBox();
        self.spn_pcap.setRange(1000, 500000);
        self.spn_pcap.setValue(self.ctrl.cfg.plane_sample_cap)
        self.cmb_zsel = QComboBox();
        self.cmb_zsel.addItems(["max", "min"]);
        self.cmb_zsel.setCurrentText(self.ctrl.cfg.z_select)
        self.spn_qlo = QDoubleSpinBox();
        self.spn_qlo.setRange(0.0, 50.0);
        self.spn_qlo.setDecimals(1);
        self.spn_qlo.setValue(self.ctrl.cfg.nearest_qlo)
        self.spn_qhi = QDoubleSpinBox();
        self.spn_qhi.setRange(50.0, 100.0);
        self.spn_qhi.setDecimals(1);
        self.spn_qhi.setValue(self.ctrl.cfg.nearest_qhi)
        self.spn_dmargin = QDoubleSpinBox();
        self.spn_dmargin.setRange(0.1, 50.0);
        self.spn_dmargin.setValue(self.ctrl.cfg.depth_margin_mm)
        self.spn_openk = QSpinBox();
        self.spn_openk.setRange(0, 31);
        self.spn_openk.setValue(self.ctrl.cfg.morph_open)
        self.spn_closek = QSpinBox();
        self.spn_closek.setRange(0, 31);
        self.spn_closek.setValue(self.ctrl.cfg.morph_close)
        self.spn_minarea = QSpinBox();
        self.spn_minarea.setRange(1, 100000);
        self.spn_minarea.setValue(self.ctrl.cfg.min_component_area_px)

        # tooltips（中文说明）
        self.chk_plane.setToolTip("开启平面拟合并展平高度图，提高最近表面稳定性；若工件平整，建议开启。")
        self.spn_pransac.setToolTip("RANSAC 内点阈值 mm；值越大越宽松（默认 0.8mm）。")
        self.spn_piters.setToolTip("RANSAC 迭代次数；越大越稳，越慢（默认 500）。")
        self.spn_pcap.setToolTip("RANSAC 采样点上限，控制耗时（默认 120k）。")
        self.cmb_zsel.setToolTip("最近表面选取：max=更靠近相机/更高处；min=更靠近下方。")
        self.spn_qlo.setToolTip("分位下界（%）；与上界一起限定取层范围。")
        self.spn_qhi.setToolTip("分位上界（%）；常用 95~99。")
        self.spn_dmargin.setToolTip("相对参考层的厚度边界（mm），越大越宽。")
        self.spn_openk.setToolTip("开运算核大小（像素）；去小噪点。0=关闭。")
        self.spn_closek.setToolTip("闭运算核大小（像素）；补小孔洞。0=关闭。")
        self.spn_minarea.setToolTip("连通域保留的最小面积；避免误检。")

        ff.addRow(self.chk_plane)
        ff.addRow("plane_ransac_thresh_mm:", self.spn_pransac)
        ff.addRow("plane_ransac_iters:", self.spn_piters)
        ff.addRow("plane_sample_cap:", self.spn_pcap)
        ff.addRow("z_select:", self.cmb_zsel)
        ff.addRow("nearest_qlo%:", self.spn_qlo)
        ff.addRow("nearest_qhi%:", self.spn_qhi)
        ff.addRow("depth_margin_mm:", self.spn_dmargin)
        ff.addRow("morph_open(px):", self.spn_openk)
        ff.addRow("morph_close(px):", self.spn_closek)
        ff.addRow("min_component_area_px:", self.spn_minarea)
        lv.addWidget(g_flat)

        # ====== 调试 / 可视化 ======
        g_dbg = QGroupBox("调试 / 可视化")
        fd = QFormLayout(g_dbg)
        self.chk_probe = QCheckBox("显示法向采样线");
        self.chk_probe.setChecked(self.ctrl.cfg.draw_normal_probes)
        self.spn_arrow = QSpinBox();
        self.spn_arrow.setRange(1, 200);
        self.spn_arrow.setValue(self.ctrl.cfg.arrow_stride)
        self.chk_dbgwin = QCheckBox("法线-交点小窗");
        self.chk_dbgwin.setChecked(self.ctrl.cfg.debug_normals_window)
        self.spn_dbgstr = QSpinBox();
        self.spn_dbgstr.setRange(1, 200);
        self.spn_dbgstr.setValue(self.ctrl.cfg.debug_normals_stride)
        self.spn_dbgmax = QSpinBox();
        self.spn_dbgmax.setRange(1, 200);
        self.spn_dbgmax.setValue(self.ctrl.cfg.debug_normals_max)
        self.spn_dbgl = QDoubleSpinBox();
        self.spn_dbgl.setRange(0.0, 100.0);
        self.spn_dbgl.setDecimals(2)
        self.spn_dbgl.setValue(self.ctrl.cfg.debug_normals_len_mm or 0.0)
        self.chk_dbgtext = QCheckBox("在交点标注 Δn");
        self.chk_dbgtext.setChecked(self.ctrl.cfg.debug_normals_text)

        self.chk_probe.setToolTip("在叠加图上显示每个采样点的法向扫描线，用于查错对齐。")
        self.spn_arrow.setToolTip("偏差箭头抽样步长（点）；数值越大，箭头越稀疏。")
        self.chk_dbgwin.setToolTip("开启小窗：稀疏显示法线与交点，便于核查。")
        self.spn_dbgstr.setToolTip("小窗法线抽样步长。")
        self.spn_dbgmax.setToolTip("小窗最多显示的法线根数。")
        self.spn_dbgl.setToolTip("小窗法线长度（mm）；0=自动。")
        self.chk_dbgtext.setToolTip("是否在交点旁标注 Δn 文本。")

        fd.addRow(self.chk_probe)
        fd.addRow("arrow_stride:", self.spn_arrow)
        fd.addRow(self.chk_dbgwin)
        fd.addRow("debug_stride:", self.spn_dbgstr)
        fd.addRow("debug_max:", self.spn_dbgmax)
        fd.addRow("debug_len_mm(0=auto):", self.spn_dbgl)
        fd.addRow(self.chk_dbgtext)
        lv.addWidget(g_dbg)

        # ====== 参数说明（速查） ======
        g_help = QGroupBox("参数说明（速查）")
        vh = QVBoxLayout(g_help)
        self.txt_help = QTextEdit();
        self.txt_help.setReadOnly(True)
        self.txt_help.setHtml("""
              <h3>核心参数解释</h3>
              <ul>
                <li><b>plane_ransac_thresh_mm</b>：平面拟合内点阈值（越大越宽松）。</li>
                <li><b>z_select / nearest_qlo/qhi / depth_margin_mm</b>：决定“最近表面”所在高度带。</li>
                <li><b>morph_open/close</b>：形态学净化，去噪与补洞。</li>
                <li><b>guide_halfwidth_mm</b>：法向扫描半宽；太小易丢；太大易误匹配。</li>
                <li><b>guide_smooth_win / curvature_adaptive</b>：平滑窗口与曲率自适应，抑制抖动并保护拐角。</li>
                <li><b>guide_max_grad_mm_per_mm</b>：梯度限幅，限制相邻点 Δn 跳变。</li>
                <li><b>max_gap_pts</b>：仅对短缺口插值；长缺口由 Guard 拦截。</li>
                <li><b>occ_*</b>：遮挡区配置；可在遮挡内按 G 代码合成环带掩码。</li>
                <li><b>Guard</b>：导出安全门槛（命中率、p95、平面内点率、长缺失、梯度）。</li>
              </ul>
              """)
        vh.addWidget(self.txt_help)
        lv.addWidget(g_help)

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
        foot = QFrame()
        foot.setFrameShape(QFrame.StyledPanel)
        foot.setStyleSheet("QFrame { background: #fafafa; border-top: 1px solid #d9d9d9; }")
        foot.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        btns = QHBoxLayout(foot)
        btns.setContentsMargins(8, 6, 8, 6)
        btns.setSpacing(8)
        self.btn_preview = QPushButton("预览单帧")
        self.btn_export = QPushButton("导出纠偏 (CSV + GCode)")
        self.btn_bias = QPushButton("保存 BiasComp (当前帧)")
        self.btn_advanced_params = QPushButton("高级参数调节")
        self.btn_advanced_params.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_bias.clicked.connect(self.on_save_bias)
        self.btn_advanced_params.clicked.connect(self.open_advanced_params)

        btns.addWidget(self.btn_preview)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_bias)
        btns.addWidget(self.btn_advanced_params)  # 添加高级参数按钮

        # 把页脚加到左栏容器的底部（固定，不随滚动）
        left_col_v.addWidget(foot, 0)
        # lv.addLayout(btns)
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

        self.lbl_top = QLabel("顶视高度图");
        self.lbl_top.setAlignment(Qt.AlignCenter);
        self.lbl_top.setMinimumHeight(240)
        self.lbl_nearest = QLabel("最近表面掩码");
        self.lbl_nearest.setAlignment(Qt.AlignCenter);
        self.lbl_nearest.setMinimumHeight(240)
        w4 = QWidget();
        v4 = QVBoxLayout(w4);
        v4.addWidget(self.lbl_top)
        w5 = QWidget();
        v5 = QVBoxLayout(w5);
        v5.addWidget(self.lbl_nearest)
        self.tabs.addTab(w4, "顶视高度")
        self.tabs.addTab(w5, "最近表面")

        self.lbl_corr = QLabel("纠偏叠加预览");
        self.lbl_corr.setAlignment(Qt.AlignCenter);
        self.lbl_corr.setMinimumHeight(360)
        w6 = QWidget();
        v6 = QVBoxLayout(w6);
        v6.addWidget(self.lbl_corr)
        self.tabs.addTab(w6, "纠偏叠加")

        # 指标卡
        self.lbl_metrics = QLabel();
        self.lbl_metrics.setAlignment(Qt.AlignLeft)
        self.lbl_metrics.setStyleSheet("QLabel { font-family: Consolas, Menlo, monospace; }")
        rv.addWidget(self.lbl_metrics)

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


        # corner ignoring
        try:
            c.corner_ignore_enable = self.chk_corner.isChecked()
            c.corner_angle_thr_deg = float(self.spn_cang.value())
            c.corner_ignore_span_mm = float(self.spn_cspan.value())
        except Exception:
            pass
        # guard
        c.guard_min_valid_ratio = float(self.spn_vr.value())
        c.guard_max_abs_p95_mm = float(self.spn_p95.value())
        c.out_dir = self.ed_outdir.text().strip()
        c.offset_csv = self.ed_csv.text().strip()
        c.corrected_gcode = self.ed_gc.text().strip()

        # 平面/展平
        c.plane_enable = self.chk_plane.isChecked()
        c.plane_ransac_thresh_mm = float(self.spn_pransac.value())
        c.plane_ransac_iters = int(self.spn_piters.value())
        c.plane_sample_cap = int(self.spn_pcap.value())
        c.z_select = self.cmb_zsel.currentText()
        c.nearest_qlo = float(self.spn_qlo.value())
        c.nearest_qhi = float(self.spn_qhi.value())
        c.depth_margin_mm = float(self.spn_dmargin.value())
        c.morph_open = int(self.spn_openk.value())
        c.morph_close = int(self.spn_closek.value())
        c.min_component_area_px = int(self.spn_minarea.value())

        # 调试/可视化
        c.draw_normal_probes = self.chk_probe.isChecked()
        c.arrow_stride = int(self.spn_arrow.value())
        c.debug_normals_window = self.chk_dbgwin.isChecked()
        c.debug_normals_stride = int(self.spn_dbgstr.value())
        c.debug_normals_max = int(self.spn_dbgmax.value())
        val_len = float(self.spn_dbgl.value())
        c.debug_normals_len_mm = (None if abs(val_len) < 1e-9 else val_len)
        c.debug_normals_text = self.chk_dbgtext.isChecked()

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

        def set_img(label: QLabel, img):
            if img is None: label.setText("(无图像)"); return
            q = np_to_qimage(img)
            if q is None: label.setText("(无法显示图像)"); return
            label.setPixmap(QPixmap.fromImage(q).scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        set_img(self.lbl_vis, out.get("vis_cmp"))
        set_img(self.lbl_probe, out.get("vis_probe"))
        set_img(self.lbl_hist, out.get("hist_panel"))
        set_img(self.lbl_top, out.get("vis_top"))
        set_img(self.lbl_nearest, out.get("vis_nearest"))
        set_img(self.lbl_corr, out.get("vis_corr"))

        m = out.get("metrics", {})
        if m:
            self.lbl_metrics.setText(
                "valid_ratio: {:>5.2f}    p95(mm): {:>6.3f}    plane_inlier: {:>5}\n"
                "longest_missing(mm): {:>6.2f}".format(
                    float(m.get("valid_ratio", 0.0)),
                    float(m.get("dev_p95", 0.0)),
                    ("{:.2f}".format(m["plane_inlier_ratio"]) if str(m.get("plane_inlier_ratio"))!="nan" else "nan"),
                    float(m.get("longest_missing_mm", 0.0))
                )
            )
        else:
            self.lbl_metrics.setText("")

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
    
    # ---- 槽：打开高级参数调节窗口 ----
    def open_advanced_params(self):
        """打开高级参数调节窗口"""
        try:
            # 先更新当前配置
            self.flush_cfg()
            
            # 导入高级参数对话框
            from simple_advanced_params import SimpleAdvancedParametersDialog
            
            # 创建对话框
            dialog = SimpleAdvancedParametersDialog(self.ctrl, self)
            
            # 连接参数应用信号
            dialog.parameters_applied.connect(self.on_advanced_params_applied)
            
            # 显示对话框
            dialog.exec_()
            
        except ImportError:
            QMessageBox.warning(self, "错误", "无法加载高级参数对话框模块")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开高级参数窗口失败: {e}")
    
    def on_advanced_params_applied(self, params):
        """高级参数应用后的回调"""
        try:
            # 参数已经在对话框中应用到控制器，这里可以做一些后续处理
            
            # 自动触发一次预览以显示参数效果
            if hasattr(self, 'btn_preview') and self.btn_preview.isEnabled():
                QTimer.singleShot(500, self.on_preview)  # 0.5秒后自动预览
                
            self.statusBar().showMessage("高级参数已应用")
            
        except Exception as e:
            print(f"高级参数应用回调错误: {e}")

# ---- 入口 ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
