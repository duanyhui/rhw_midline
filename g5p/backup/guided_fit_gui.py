# -*- coding: utf-8 -*-
"""
guided_fit_gui.py (enhanced)
----------------------------
- 预留“外部偏差输入”接口：
  1) 导入 CSV / JSON 文件；
  2) 内置 HTTP 端点（127.0.0.1:8765/offsets）接收 JSON。

- 丰富的偏差可视化：
  * 顶视叠加（多层箭头 + 热力线）
  * 偏差直方图
  * 偏差-弧长 曲线
  * Run 列表（可见性切换、颜色标识、来源）

运行：python guided_fit_gui.py
"""
import sys, os, time, json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import guided_fit_pipeline as gfp
from param_help import HELP
import offset_inlet as inlet
import align_centerline_to_gcode_pro_edit as core


# ---------------------- Matplotlib 画布 ----------------------
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4.6, height=3.2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)

    def plot_hist(self, data: np.ndarray, bins:int=40, title='Histogram'):
        self.fig.clear(); self.ax = self.fig.add_subplot(111)
        if data is None or len(data)==0:
            self.ax.text(0.5,0.5,'No data', ha='center', va='center')
        else:
            self.ax.hist(data, bins=bins)
            self.ax.set_xlabel('delta_n (mm)')
            self.ax.set_ylabel('count')
            self.ax.set_title(title)
        self.draw()

    def plot_curve(self, s: np.ndarray, y: np.ndarray, title='Offset vs s (mm)'):
        self.fig.clear(); self.ax = self.fig.add_subplot(111)
        if y is None or len(y)==0:
            self.ax.text(0.5,0.5,'No data', ha='center', va='center')
        else:
            if s is None or len(s) != len(y):
                s = np.arange(len(y), dtype=float)
            self.ax.plot(s, y)
            self.ax.set_xlabel('s (mm)')
            self.ax.set_ylabel('delta_n (mm)')
            self.ax.set_title(title)
            self.ax.grid(True, alpha=0.35)
        self.draw()


# ---------------------- 叠加层数据结构 ----------------------
class OverlayStack:
    def __init__(self):
        self.base_img = None
        self.origin_xy = None
        self.pix_mm = None
        self.HW = None
        self.g_xy = None
        self.N_ref = None
        self.layers: List[Dict[str, Any]] = []  # each: {name, delta, color, stride, visible, s}

    def reset(self):
        self.base_img = None; self.origin_xy = None; self.pix_mm = None
        self.HW = None; self.g_xy = None; self.N_ref = None; self.layers.clear()

    def init_base(self, base_img: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float,
                  height_shape: Tuple[int,int], g_xy: np.ndarray, N_ref: np.ndarray):
        if self.base_img is None:
            self.base_img = base_img.copy()
            self.origin_xy = origin_xy; self.pix_mm = float(pix_mm)
            self.HW = (int(height_shape[0]), int(height_shape[1]))
            self.g_xy = g_xy.copy() if g_xy is not None else None
            self.N_ref = N_ref.copy() if N_ref is not None else None

    def add_layer(self, name: str, delta_n: np.ndarray, color=(0,255,255), stride:int=12, s: np.ndarray=None):
        if self.g_xy is None or self.N_ref is None or delta_n is None or len(delta_n)==0:
            return
        M = min(len(delta_n), len(self.g_xy), len(self.N_ref))
        self.layers.append(dict(name=name, delta=delta_n[:M].copy(),
                                color=tuple(map(int,color)), stride=int(stride),
                                visible=True, s=s[:M].copy() if s is not None else None))

    def set_visible(self, idx:int, on:bool):
        if 0 <= idx < len(self.layers):
            self.layers[idx]['visible'] = bool(on)

    def _xy_to_px(self, xy: np.ndarray) -> np.ndarray:
        x0, y0 = self.origin_xy; H, W = self.base_img.shape[:2]
        y1 = y0 + H * self.pix_mm
        xs = np.clip(((xy[:,0]-x0)/self.pix_mm).astype(np.int32), 0, W-1)
        ys = np.clip(((y1 - xy[:,1])/self.pix_mm).astype(np.int32), 0, H-1)
        return np.stack([xs,ys], axis=1)

    def render(self, draw_heat=False) -> np.ndarray:
        if self.base_img is None: return None
        vis = self.base_img.copy()
        if self.g_xy is None or self.N_ref is None: return vis
        # 可选：沿路径画“热力线”（根据 delta 绝对值着色）
        if draw_heat and self.layers:
            # 叠加所有可见层的绝对偏差（取最近一层为主，或取均值/最大值，这里用均值）
            M = len(self.g_xy)
            agg = np.zeros(M, float); cnt = np.zeros(M, int)
            for L in self.layers:
                if not L['visible']: continue
                dn = L['delta']; m = min(len(dn), M)
                agg[:m] += np.abs(dn[:m]); cnt[:m] += 1
            ok = cnt > 0
            if ok.any():
                agg[ok] /= cnt[ok]
                vmin, vmax = np.percentile(agg[ok], 5), np.percentile(agg[ok], 95)
                vmax = max(vmax, vmin + 1e-6)
                norm = (np.clip(agg, vmin, vmax) - vmin) / (vmax - vmin)
                cmap = cv2.COLORMAP_TURBO if hasattr(cv2, 'COLORMAP_TURBO') else cv2.COLORMAP_JET
                colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)[:,0,:]  # (M,3) BGR
                P = self._xy_to_px(self.g_xy)
                for i in range(len(P)-1):
                    cv2.line(vis, tuple(P[i]), tuple(P[i+1]), tuple(int(x) for x in colors[i]), 2, cv2.LINE_AA)

        # 箭头层
        for li, L in enumerate(self.layers, start=1):
            if not L['visible']: continue
            dn = L['delta']; color = L['color']; stride = max(1, int(L['stride']))
            M = min(len(dn), len(self.g_xy), len(self.N_ref))
            g = self.g_xy[:M]; N = self.N_ref[:M]
            for i in range(0, M, stride):
                p0 = g[i]; p1 = g[i] + N[i] * dn[i]
                P = self._xy_to_px(np.vstack([p0, p1]))
                cv2.arrowedLine(vis, tuple(P[0]), tuple(P[1]), color, 2, cv2.LINE_AA, tipLength=0.28)
            cv2.putText(vis, f'run#{li}:{L["name"]}', (12, 24+20*li), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f'run#{li}:{L["name"]}', (12, 24+20*li), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        return vis


# ---------------------- GUI 主窗口 ----------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Centerline ⇄ G-code — GUI 封装（增强版）")
        self.resize(1480, 920)

        self.engine = gfp.GuidedFitEngine()
        self.overlay = OverlayStack()
        self.last_result = None
        self.http_server: Optional[inlet.HttpOffsetServer] = None

        # 中心视图与右侧面板
        self.lbl_img = QtWidgets.QLabel("结果图在这里显示")
        self.lbl_img.setAlignment(QtCore.Qt.AlignCenter); self.lbl_img.setMinimumSize(900, 680)
        scr = QtWidgets.QScrollArea(); scr.setWidget(self.lbl_img); scr.setWidgetResizable(True)

        right = self._build_right_panel()

        splitter = QtWidgets.QSplitter(); splitter.addWidget(scr); splitter.addWidget(right)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("就绪")

        try:
            self.engine.open_camera()
            self.statusBar().showMessage("相机已连接")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "相机未就绪", f"打开相机失败（可先不连相机调试界面）：\n{e}")

    # ---------- 右侧面板（含 Tab） ----------
    def _build_right_panel(self):
        panel = QtWidgets.QTabWidget()

        # --- Tab1：参数与运行 ---
        tab_run = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(tab_run)
        box_files = QtWidgets.QGroupBox("文件"); f_lay = QtWidgets.QFormLayout(box_files)
        self.ed_T = QtWidgets.QLineEdit(str(core.PARAMS.get("T_path","T_cam2machine.npy")))
        self.ed_g = QtWidgets.QLineEdit(str(core.PARAMS.get("gcode_path","example.gcode")))
        btn_T = QtWidgets.QPushButton("..."); btn_T.clicked.connect(self._pick_T)
        btn_g = QtWidgets.QPushButton("..."); btn_g.clicked.connect(self._pick_g)
        rw1 = QtWidgets.QHBoxLayout(); rw1.addWidget(self.ed_T); rw1.addWidget(btn_T)
        rw2 = QtWidgets.QHBoxLayout(); rw2.addWidget(self.ed_g); rw2.addWidget(btn_g)
        f_lay.addRow("外参 T_path：", self._with_help(rw1, "T_path"))
        f_lay.addRow("G 代码路径：", self._with_help(rw2, "gcode_path"))

        box_params = QtWidgets.QGroupBox("关键参数"); p_lay = QtWidgets.QFormLayout(box_params)
        self.cmb_roi = QtWidgets.QComboBox(); self.cmb_roi.addItems(["gcode_bounds","machine","camera_rect","none"])
        self.cmb_roi.setCurrentText(str(core.PARAMS.get("roi_mode","gcode_bounds")))
        self.spn_bounds_margin = self._mk_dspin(core.PARAMS.get("bounds_margin_mm",20.0), 0.0, 200.0, 1.0)
        self.spn_pix = self._mk_dspin(core.PARAMS.get("pixel_size_mm",0.8), 0.05, 5.0, 0.05)
        self.spn_half = self._mk_dspin(core.PARAMS.get("guide_halfwidth_mm",6.0), 0.5, 50.0, 0.5)
        self.spn_step = self._mk_dspin(core.PARAMS.get("guide_step_mm",2.0), 0.1, 20.0, 0.1)
        self.spn_smooth = QtWidgets.QSpinBox(); self.spn_smooth.setRange(1, 99); self.spn_smooth.setSingleStep(2)
        self.spn_smooth.setValue(int(core.PARAMS.get("guide_smooth_win",7)))
        self.spn_maxoff = self._mk_dspin(core.PARAMS.get("guide_max_offset_mm",8.0), 0.5, 50.0, 0.5)
        self.chk_plane = QtWidgets.QCheckBox("启用平面展平"); self.chk_plane.setChecked(bool(core.PARAMS.get("plane_enable",True)))
        self.spn_plane_thr = self._mk_dspin(core.PARAMS.get("plane_ransac_thresh_mm",0.8), 0.05, 5.0, 0.05)
        self.spn_depth = self._mk_dspin(core.PARAMS.get("depth_margin_mm",3.0), 0.1, 20.0, 0.1)
        self.cmb_apply = QtWidgets.QComboBox(); self.cmb_apply.addItems(["invert","follow"])
        self.cmb_apply.setCurrentText(str(core.PARAMS.get("offset_apply_mode","invert")))
        self.spn_stride = QtWidgets.QSpinBox(); self.spn_stride.setRange(1, 100); self.spn_stride.setValue(int(core.PARAMS.get("arrow_stride",12)))
        p_lay.addRow("ROI 模式：", self._with_help(self.cmb_roi, "roi_mode"))
        p_lay.addRow("边界外扩 (mm)：", self._with_help(self.spn_bounds_margin, "bounds_margin_mm"))
        p_lay.addRow("像素尺寸 (mm/px)：", self._with_help(self.spn_pix, "pixel_size_mm"))
        p_lay.addRow("引导法向半宽 (mm)：", self._with_help(self.spn_half, "guide_halfwidth_mm"))
        p_lay.addRow("G 代码重采样步长 (mm)：", self._with_help(self.spn_step, "guide_step_mm"))
        p_lay.addRow("偏移移动平均窗口：", self._with_help(self.spn_smooth, "guide_smooth_win"))
        p_lay.addRow("偏移限幅 (mm)：", self._with_help(self.spn_maxoff, "guide_max_offset_mm"))
        p_lay.addRow(self.chk_plane)
        p_lay.addRow("平面 RANSAC 阈值 (mm)：", self._with_help(self.spn_plane_thr, "plane_ransac_thresh_mm"))
        p_lay.addRow("最近层厚度 (mm)：", self._with_help(self.spn_depth, "depth_margin_mm"))
        p_lay.addRow("纠偏应用策略：", self._with_help(self.cmb_apply, "offset_apply_mode"))
        p_lay.addRow("箭头步长：", self._with_help(self.spn_stride, "arrow_stride"))

        box_out = QtWidgets.QGroupBox("输出"); o_lay = QtWidgets.QFormLayout(box_out)
        self.ed_outdir = QtWidgets.QLineEdit(str(core.PARAMS.get("out_dir","out")))
        self.ed_csv = QtWidgets.QLineEdit(str(core.PARAMS.get("offset_csv","out/offset_table.csv")))
        self.ed_corr = QtWidgets.QLineEdit(str(core.PARAMS.get("corrected_gcode","out/corrected.gcode")))
        btn_od = QtWidgets.QPushButton("..."); btn_od.clicked.connect(self._pick_outdir)
        rw3 = QtWidgets.QHBoxLayout(); rw3.addWidget(self.ed_outdir); rw3.addWidget(btn_od)
        o_lay.addRow("输出目录：", rw3); o_lay.addRow("偏差 CSV：", self.ed_csv); o_lay.addRow("纠偏 G 代码：", self.ed_corr)

        box_ops = QtWidgets.QGroupBox("操作"); grid = QtWidgets.QGridLayout(box_ops)
        self.btn_run = QtWidgets.QPushButton("采集一帧并计算")
        self.btn_export = QtWidgets.QPushButton("导出偏差 + 纠偏G代码（叠加）")
        self.btn_clear = QtWidgets.QPushButton("清空叠加")
        self.btn_saveimg = QtWidgets.QPushButton("保存当前图")
        self.chk_heat = QtWidgets.QCheckBox("显示热力线"); self.chk_heat.setChecked(True)
        grid.addWidget(self.btn_run, 0, 0, 1, 2)
        grid.addWidget(self.btn_export, 1, 0, 1, 2)
        grid.addWidget(self.btn_clear, 2, 0); grid.addWidget(self.btn_saveimg, 2, 1)
        grid.addWidget(self.chk_heat, 3, 0)

        self.btn_run.clicked.connect(self.on_run_once)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_saveimg.clicked.connect(self.on_save_img)

        self.txt_log = QtWidgets.QPlainTextEdit(); self.txt_log.setReadOnly(True); self.txt_log.setMinimumHeight(120)

        lay.addWidget(box_files); lay.addWidget(box_params); lay.addWidget(box_out)
        lay.addWidget(box_ops); lay.addWidget(self.txt_log, 1)

        # --- Tab2：外部偏差 ---
        tab_ext = QtWidgets.QWidget(); ly2 = QtWidgets.QVBoxLayout(tab_ext)
        row_imp = QtWidgets.QHBoxLayout()
        btn_imp_csv = QtWidgets.QPushButton("导入 CSV 偏差…"); btn_imp_csv.clicked.connect(self.on_import_csv)
        btn_imp_json = QtWidgets.QPushButton("导入 JSON 偏差…"); btn_imp_json.clicked.connect(self.on_import_json)
        row_imp.addWidget(btn_imp_csv); row_imp.addWidget(btn_imp_json); row_imp.addStretch(1)

        grp_http = QtWidgets.QGroupBox("HTTP 输入（本机 127.0.0.1:8765/offsets）")
        f2 = QtWidgets.QFormLayout(grp_http)
        self.ed_http_port = QtWidgets.QSpinBox(); self.ed_http_port.setRange(1024, 65500); self.ed_http_port.setValue(8765)
        self.btn_http = QtWidgets.QPushButton("启动服务"); self.btn_http.setCheckable(True)
        self.btn_http.toggled.connect(self.on_toggle_http)
        f2.addRow("端口：", self.ed_http_port); f2.addRow(self.btn_http)

        # Run 列表（可见性）
        grp_list = QtWidgets.QGroupBox("叠加层（run 列表）")
        v = QtWidgets.QVBoxLayout(grp_list)
        self.tbl_layers = QtWidgets.QTableWidget(0, 4)
        self.tbl_layers.setHorizontalHeaderLabels(["可见","名称","颜色","长度"])
        self.tbl_layers.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.tbl_layers.setColumnWidth(0, 48); self.tbl_layers.setColumnWidth(2, 64); self.tbl_layers.setColumnWidth(3, 64)
        v.addWidget(self.tbl_layers)
        row_btn = QtWidgets.QHBoxLayout()
        self.btn_showall = QtWidgets.QPushButton("全显"); self.btn_hideall = QtWidgets.QPushButton("全隐")
        self.btn_showall.clicked.connect(lambda: self._set_all_layers(True))
        self.btn_hideall.clicked.connect(lambda: self._set_all_layers(False))
        row_btn.addWidget(self.btn_showall); row_btn.addWidget(self.btn_hideall); row_btn.addStretch(1)
        v.addLayout(row_btn)

        ly2.addLayout(row_imp); ly2.addWidget(grp_http); ly2.addWidget(grp_list); ly2.addStretch(1)

        # --- Tab3：偏差图表 ---
        tab_plot = QtWidgets.QWidget(); ly3 = QtWidgets.QVBoxLayout(tab_plot)
        self.plot_hist = PlotCanvas(tab_plot); self.plot_curve = PlotCanvas(tab_plot)
        ly3.addWidget(self.plot_hist); ly3.addWidget(self.plot_curve)

        # 放入 TabWidget
        panel.addTab(tab_run, "运行与参数")
        panel.addTab(tab_ext, "外部偏差")
        panel.addTab(tab_plot, "偏差图表")
        return panel

    # ====== 工具 ======
    def _with_help(self, w: Union[QtWidgets.QWidget, QtWidgets.QLayout], key: str):
        tip = HELP.get(key, key)
        if isinstance(w, QtWidgets.QLayout):
            cont = QtWidgets.QWidget(); cont.setLayout(w); cont.setToolTip(tip); return cont
        else:
            w.setToolTip(tip); return w

    def _mk_dspin(self, val: float, lo: float, hi: float, step: float):
        s = QtWidgets.QDoubleSpinBox(); s.setDecimals(3); s.setRange(lo, hi); s.setSingleStep(step); s.setValue(float(val)); return s

    def _pick_T(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择外参 T.npy", "", "NumPy (*.npy);;All (*)")
        if fn: self.ed_T.setText(fn)

    def _pick_g(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 G 代码", "", "G-code (*.gcode *.nc *.txt);;All (*)")
        if fn: self.ed_g.setText(fn)

    def _pick_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录", self.ed_outdir.text().strip() or ".")
        if d: self.ed_outdir.setText(d)

    def _collect_cfg(self) -> Dict[str, Any]:
        cfg = dict(core.PARAMS)
        cfg["T_path"] = self.ed_T.text().strip()
        cfg["gcode_path"] = self.ed_g.text().strip()
        cfg["roi_mode"] = self.cmb_roi.currentText()
        cfg["bounds_margin_mm"] = float(self.spn_bounds_margin.value())
        cfg["pixel_size_mm"] = float(self.spn_pix.value())
        cfg["guide_halfwidth_mm"] = float(self.spn_half.value())
        cfg["guide_step_mm"] = float(self.spn_step.value())
        cfg["guide_smooth_win"] = int(self.spn_smooth.value())
        cfg["guide_max_offset_mm"] = float(self.spn_maxoff.value())
        cfg["plane_enable"] = bool(self.chk_plane.isChecked())
        cfg["plane_ransac_thresh_mm"] = float(self.spn_plane_thr.value())
        cfg["depth_margin_mm"] = float(self.spn_depth.value())
        cfg["offset_apply_mode"] = self.cmb_apply.currentText()
        cfg["arrow_stride"] = int(self.spn_stride.value())
        cfg["out_dir"] = self.ed_outdir.text().strip()
        cfg["offset_csv"] = self.ed_csv.text().strip()
        cfg["corrected_gcode"] = self.ed_corr.text().strip()
        return cfg

    # ====== 事件：采集一帧 ======
    def on_run_once(self):
        cfg = self._collect_cfg()
        try:
            res = self.engine.run_once(cfg)
            self.last_result = res
            self.overlay.init_base(res["vis_base"], res["origin_xy"], res["pix_mm"], res["height_shape"], res["g_xy"], res["N_ref"])
            self._refresh_view()
            st = res["stats"]
            self._log(f"run ok | valid_ratio={st['valid_ratio']:.3f}  p95={st['dev_p95']:.3f}  plane_inlier={st['plane_inlier_ratio']:.3f}")
        except Exception as e:
            self._log(f"[ERR] {e}")
            QtWidgets.QMessageBox.critical(self, "运行失败", str(e))

    # ====== 事件：导出 ======
    def on_export(self):
        if self.last_result is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先点击“采集一帧并计算”。")
            return
        cfg = self._collect_cfg()
        try:
            rep = self.engine.export_offsets_and_gcode(self.last_result, cfg, out_dir=Path(cfg["out_dir"]))
            color_cycle = [(255,255,0), (0,255,255), (255,128,0), (0,255,128), (255,0,255), (0,200,255)]
            color = color_cycle[len(self.overlay.layers) % len(color_cycle)]
            self.overlay.add_layer(f'export#{len(self.overlay.layers)+1}', rep["delta_n_apply"], color=color, stride=int(cfg["arrow_stride"]), s=rep["s"])
            self._refresh_view()
            self._log(f"导出完成：CSV={rep['csv_path']}  GCODE={rep['gcode_path']}")
        except Exception as e:
            self._log(f"[ERR] 导出失败：{e}")
            QtWidgets.QMessageBox.critical(self, "导出失败", str(e))

    # ====== 外部偏差：导入 CSV/JSON ======
    def on_import_csv(self):
        if self.overlay.g_xy is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先运行一次以加载参考 G 代码。")
            return
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 CSV 偏差文件", "", "CSV (*.csv);;All (*)")
        if not fn: return
        pkt = inlet.load_offsets_from_csv(fn)
        self._consume_external_packet(pkt, Path(fn).name)

    def on_import_json(self):
        if self.overlay.g_xy is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先运行一次以加载参考 G 代码。")
            return
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 JSON 偏差文件", "", "JSON (*.json);;All (*)")
        if not fn: return
        pkt = inlet.load_offsets_from_json(fn)
        self._consume_external_packet(pkt, Path(fn).name)

    # ====== 外部偏差：HTTP ======
    def on_toggle_http(self, checked: bool):
        if checked:
            port = int(self.ed_http_port.value())
            self.http_server = inlet.HttpOffsetServer(port=port, callback=self._consume_packet_from_http)
            self.http_server.start()
            self.btn_http.setText("停止服务")
            self._log(f"[HTTP] 监听 http://127.0.0.1:{port}/offsets  （POST JSON）")
        else:
            if self.http_server: self.http_server.stop(); self.http_server = None
            self.btn_http.setText("启动服务")
            self._log("[HTTP] 已停止")

    def _consume_packet_from_http(self, pkt: inlet.OffsetPacket):
        name = pkt.meta.get('source','http')
        QtCore.QTimer.singleShot(0, lambda pkt=pkt, name=name: self._consume_external_packet(pkt, name))

    def _consume_external_packet(self, pkt: inlet.OffsetPacket, name: str):
        if not pkt.is_valid():
            QtWidgets.QMessageBox.warning(self, "数据无效", "未检测到有效偏差数据。"); return
        dn, g_corr = gfp.align_external_offsets(self.overlay.g_xy, self.overlay.N_ref,
                                               packet=pkt, apply_mode=str(pkt.meta.get('apply_mode','as_is')))
        color_cycle = [(255,255,0), (0,255,255), (255,128,0), (0,255,128), (255,0,255), (0,200,255)]
        color = color_cycle[len(self.overlay.layers) % len(color_cycle)]
        # 叠加
        self.overlay.add_layer(f'ext:{name}', dn, color=color, stride=int(self.spn_stride.value()), s=None)
        self._refresh_view()
        self._log(f"[EXT] 已叠加外部偏差：{name}  模式={pkt.mode}  n={len(dn)}")

    # ====== Run 列表 ======
    def _set_all_layers(self, on:bool):
        for i in range(len(self.overlay.layers)):
            self.overlay.layers[i]['visible'] = on
            chk = self.tbl_layers.cellWidget(i, 0)
            if isinstance(chk, QtWidgets.QCheckBox): chk.setChecked(on)
        self._refresh_view()

    def _refresh_layer_table(self):
        self.tbl_layers.setRowCount(len(self.overlay.layers))
        for i, L in enumerate(self.overlay.layers):
            # 可见
            chk = QtWidgets.QCheckBox(); chk.setChecked(L['visible']); chk.stateChanged.connect(lambda _, idx=i: self._on_layer_vis(idx))
            self.tbl_layers.setCellWidget(i, 0, chk)
            # 名称
            self.tbl_layers.setItem(i, 1, QtWidgets.QTableWidgetItem(str(L['name'])))
            # 颜色
            cell = QtWidgets.QTableWidgetItem(' ')
            col = QtGui.QColor(int(L['color'][2]), int(L['color'][1]), int(L['color'][0]))
            cell.setBackground(col); self.tbl_layers.setItem(i, 2, cell)
            # 长度
            self.tbl_layers.setItem(i, 3, QtWidgets.QTableWidgetItem(str(len(L['delta']))))

    def _on_layer_vis(self, idx: int):
        if 0 <= idx < len(self.overlay.layers):
            self.overlay.layers[idx]['visible'] = not self.overlay.layers[idx]['visible']
            self._refresh_view()

    # ====== 刷新视图 & 图表 ======
    def _refresh_view(self):
        vis = self.overlay.render(draw_heat=self.chk_heat.isChecked())
        if vis is None and self.last_result is not None:
            vis = self.last_result["vis_base"]
        if vis is not None:
            self.lbl_img.setPixmap(self._cvimg_to_qpixmap(vis))
        self._refresh_layer_table()
        # 图表：取最后一层或叠加均值
        if self.overlay.layers:
            dn = self.overlay.layers[-1]['delta']
            s = self.overlay.layers[-1].get('s', None)
            self.plot_hist.plot_hist(dn, bins=40, title='delta_n histogram (last layer)')
            s_ref = gfp.compute_arclength(self.overlay.g_xy)
            if s is None or len(s) != len(dn): s = np.linspace(0, s_ref[-1] if len(s_ref) else len(dn), len(dn))
            self.plot_curve.plot_curve(s, dn, title='delta_n vs s (last layer)')

    # ====== 其他操作 ======
    def on_clear(self):
        self.overlay.reset(); self.lbl_img.clear(); self.tbl_layers.setRowCount(0)
        self._log("已清空叠加。")

    def on_save_img(self):
        if self.overlay.base_img is None:
            QtWidgets.QMessageBox.information(self, "提示", "没有可保存的图像。先运行一次或导入一次。"); return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存当前可视化", "overlay.png", "PNG (*.png)")
        if fn:
            vis = self.overlay.render(draw_heat=self.chk_heat.isChecked()) or self.overlay.base_img
            cv2.imwrite(fn, vis); self._log(f"[SAVE] {fn}")

    def _cvimg_to_qpixmap(self, img_bgr: np.ndarray) -> QtGui.QPixmap:
        if img_bgr is None: return QtGui.QPixmap()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg)

    def _log(self, s: str):
        self.txt_log.appendPlainText(s)
        sb = self.txt_log.verticalScrollBar(); sb.setValue(sb.maximum())


# ---------------------- main ----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(); win.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
