# -*- coding: utf-8 -*-
"""
多层加工系统 - 高级参数调节界面
基于simple_advanced_params.py，专为多层系统定制
"""
import os
import sys
import json
import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

# 本地导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class MultilayerAdvancedParametersDialog(QDialog):
    """多层加工系统高级参数调节对话框"""
    
    parameters_applied = pyqtSignal(dict)  # 参数应用信号
    
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("多层加工系统 - 高级参数调节")
        self.setModal(True)
        self.resize(1200, 800)
        
        self.setup_ui()
        self.load_current_parameters()
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        tabs = QTabWidget()
        
        # ROI投影参数选项卡
        roi_tab = self.create_roi_tab()
        tabs.addTab(roi_tab, "ROI投影")
        
        # 最近表面参数选项卡
        surface_tab = self.create_surface_tab()
        tabs.addTab(surface_tab, "最近表面")
        
        # 可视化参数选项卡
        visual_tab = self.create_visualization_tab()
        tabs.addTab(visual_tab, "法线可视化")
        
        # 遮挡区域参数选项卡
        occlusion_tab = self.create_occlusion_tab()
        tabs.addTab(occlusion_tab, "遮挡区域")
        
        # 引导中心线选项卡
        guide_tab = self.create_guide_tab()
        tabs.addTab(guide_tab, "引导中心线")
        
        # 多层特有参数选项卡
        multilayer_tab = self.create_multilayer_tab()
        tabs.addTab(multilayer_tab, "多层配置")
        
        layout.addWidget(tabs)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        
        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset_parameters)
        btn_layout.addWidget(self.btn_reset)
        
        self.btn_save_preset = QPushButton("保存预设")
        self.btn_save_preset.clicked.connect(self.save_preset)
        btn_layout.addWidget(self.btn_save_preset)
        
        self.btn_load_preset = QPushButton("加载预设")
        self.btn_load_preset.clicked.connect(self.load_preset)
        btn_layout.addWidget(self.btn_load_preset)
        
        btn_layout.addStretch()
        
        self.btn_preview = QPushButton("预览效果")
        self.btn_preview.clicked.connect(self.preview_changes)
        btn_layout.addWidget(self.btn_preview)
        
        self.btn_apply = QPushButton("应用")
        self.btn_apply.clicked.connect(self.apply_parameters)
        btn_layout.addWidget(self.btn_apply)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(btn_layout)
        
    def create_roi_tab(self):
        """创建ROI投影参数选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # ROI模式设置
        roi_group = QGroupBox("ROI模式设置")
        roi_layout = QFormLayout(roi_group)
        
        self.cmb_roi_mode = QComboBox()
        self.cmb_roi_mode.addItems(["none", "camera_rect", "machine", "gcode_bounds"])
        self.cmb_roi_mode.setToolTip("ROI模式:\nnone-无ROI限制\ncamera_rect-相机矩形ROI\nmachine-机床坐标ROI\ngcode_bounds-G代码边界ROI")
        roi_layout.addRow("ROI模式:", self.cmb_roi_mode)
        
        self.spn_pixel_size = QDoubleSpinBox()
        self.spn_pixel_size.setRange(0.05, 10.0)
        self.spn_pixel_size.setDecimals(3)
        self.spn_pixel_size.setToolTip("像素物理尺寸，决定顶视投影分辨率，影响精度和计算量")
        roi_layout.addRow("像素尺寸(mm/pixel):", self.spn_pixel_size)
        
        self.spn_bounds_margin = QDoubleSpinBox()
        self.spn_bounds_margin.setRange(0, 100)
        self.spn_bounds_margin.setDecimals(1)
        self.spn_bounds_margin.setToolTip("投影边界额外扩展距离，确保完整覆盖工件")
        roi_layout.addRow("边界扩展(mm):", self.spn_bounds_margin)
        
        layout.addWidget(roi_group)
        
        # 相机矩形ROI
        camera_group = QGroupBox("相机矩形ROI设置")
        camera_layout = QFormLayout(camera_group)
        
        self.spn_roi_x = QSpinBox()
        self.spn_roi_x.setRange(0, 4000)
        self.spn_roi_x.setToolTip("ROI矩形左上角X坐标(像素)")
        camera_layout.addRow("X坐标(像素):", self.spn_roi_x)
        
        self.spn_roi_y = QSpinBox()
        self.spn_roi_y.setRange(0, 4000)
        self.spn_roi_y.setToolTip("ROI矩形左上角Y坐标(像素)")
        camera_layout.addRow("Y坐标(像素):", self.spn_roi_y)
        
        self.spn_roi_w = QSpinBox()
        self.spn_roi_w.setRange(50, 4000)
        self.spn_roi_w.setToolTip("ROI矩形宽度(像素)")
        camera_layout.addRow("宽度(像素):", self.spn_roi_w)
        
        self.spn_roi_h = QSpinBox()
        self.spn_roi_h.setRange(50, 4000)
        self.spn_roi_h.setToolTip("ROI矩形高度(像素)")
        camera_layout.addRow("高度(像素):", self.spn_roi_h)
        
        layout.addWidget(camera_group)
        
        # 机床坐标ROI
        machine_group = QGroupBox("机床坐标ROI设置")
        machine_layout = QFormLayout(machine_group)
        
        self.spn_roi_center_x = QDoubleSpinBox()
        self.spn_roi_center_x.setRange(-1000, 1000)
        self.spn_roi_center_x.setDecimals(3)
        self.spn_roi_center_x.setToolTip("ROI中心X坐标(机床坐标系)")
        machine_layout.addRow("中心X(mm):", self.spn_roi_center_x)
        
        self.spn_roi_center_y = QDoubleSpinBox()
        self.spn_roi_center_y.setRange(-1000, 1000)
        self.spn_roi_center_y.setDecimals(3)
        self.spn_roi_center_y.setToolTip("ROI中心Y坐标(机床坐标系)")
        machine_layout.addRow("中心Y(mm):", self.spn_roi_center_y)
        
        self.spn_roi_size = QDoubleSpinBox()
        self.spn_roi_size.setRange(1, 5000)
        self.spn_roi_size.setDecimals(1)
        self.spn_roi_size.setToolTip("ROI正方形尺寸(mm)")
        machine_layout.addRow("ROI尺寸(mm):", self.spn_roi_size)
        
        layout.addWidget(machine_group)
        layout.addStretch()
        
        return widget
        
    def create_surface_tab(self):
        """创建最近表面参数选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 平面展平
        plane_group = QGroupBox("平面展平算法")
        plane_layout = QFormLayout(plane_group)
        
        self.chk_plane_enable = QCheckBox("启用平面展平")
        self.chk_plane_enable.setToolTip("对高度图进行平面拟合并展平，提高稳定性")
        plane_layout.addRow(self.chk_plane_enable)
        
        self.spn_plane_thresh = QDoubleSpinBox()
        self.spn_plane_thresh.setRange(0.1, 10.0)
        self.spn_plane_thresh.setDecimals(2)
        self.spn_plane_thresh.setToolTip("RANSAC平面拟合内点阈值，值越大越宽松")
        plane_layout.addRow("RANSAC阈值(mm):", self.spn_plane_thresh)
        
        self.spn_plane_iters = QSpinBox()
        self.spn_plane_iters.setRange(50, 5000)
        self.spn_plane_iters.setToolTip("RANSAC迭代次数，越大越稳但越慢")
        plane_layout.addRow("RANSAC迭代次数:", self.spn_plane_iters)
        
        self.spn_plane_sample_cap = QSpinBox()
        self.spn_plane_sample_cap.setRange(1000, 500000)
        self.spn_plane_sample_cap.setToolTip("采样点上限，控制计算耗时")
        plane_layout.addRow("采样点上限:", self.spn_plane_sample_cap)
        
        layout.addWidget(plane_group)
        
        # 最近表面提取
        surface_group = QGroupBox("最近表面提取")
        surface_layout = QFormLayout(surface_group)
        
        self.cmb_z_select = QComboBox()
        self.cmb_z_select.addItems(["max", "min"])
        self.cmb_z_select.setToolTip("Z方向选择：max=靠近相机，min=靠近下方")
        surface_layout.addRow("Z选择模式:", self.cmb_z_select)
        
        self.spn_nearest_qlo = QDoubleSpinBox()
        self.spn_nearest_qlo.setRange(0.0, 50.0)
        self.spn_nearest_qlo.setDecimals(1)
        self.spn_nearest_qlo.setToolTip("高度分布下分位数，与上分位数一起限定取层范围")
        surface_layout.addRow("下分位数(%):", self.spn_nearest_qlo)
        
        self.spn_nearest_qhi = QDoubleSpinBox()
        self.spn_nearest_qhi.setRange(50.0, 100.0)
        self.spn_nearest_qhi.setDecimals(1)
        self.spn_nearest_qhi.setToolTip("高度分布上分位数，常用95~99")
        surface_layout.addRow("上分位数(%):", self.spn_nearest_qhi)
        
        self.spn_depth_margin = QDoubleSpinBox()
        self.spn_depth_margin.setRange(0.1, 50.0)
        self.spn_depth_margin.setDecimals(2)
        self.spn_depth_margin.setToolTip("相对参考层的厚度边界，越大越宽")
        surface_layout.addRow("深度边界(mm):", self.spn_depth_margin)
        
        layout.addWidget(surface_group)
        
        # 形态学处理
        morph_group = QGroupBox("形态学处理")
        morph_layout = QFormLayout(morph_group)
        
        self.spn_morph_open = QSpinBox()
        self.spn_morph_open.setRange(0, 31)
        self.spn_morph_open.setToolTip("开运算核大小，去小噪点，0=关闭")
        morph_layout.addRow("开运算核大小(像素):", self.spn_morph_open)
        
        self.spn_morph_close = QSpinBox()
        self.spn_morph_close.setRange(0, 31)
        self.spn_morph_close.setToolTip("闭运算核大小，补小孔洞，0=关闭")
        morph_layout.addRow("闭运算核大小(像素):", self.spn_morph_close)
        
        self.spn_min_area = QSpinBox()
        self.spn_min_area.setRange(1, 100000)
        self.spn_min_area.setToolTip("连通域保留的最小面积，避免误检")
        morph_layout.addRow("最小连通域(像素):", self.spn_min_area)
        
        layout.addWidget(morph_group)
        layout.addStretch()
        
        return widget
        
    def create_visualization_tab(self):
        """创建法线可视化参数选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 法线显示
        normal_group = QGroupBox("法线显示设置")
        normal_layout = QFormLayout(normal_group)
        
        self.chk_draw_normals = QCheckBox("显示法向采样线")
        self.chk_draw_normals.setToolTip("在叠加图上显示每个采样点的法向扫描线")
        normal_layout.addRow(self.chk_draw_normals)
        
        self.spn_arrow_stride = QSpinBox()
        self.spn_arrow_stride.setRange(1, 200)
        self.spn_arrow_stride.setToolTip("偏差箭头抽样步长，数值越大箭头越稀疏")
        normal_layout.addRow("箭头间距:", self.spn_arrow_stride)
        
        layout.addWidget(normal_group)
        
        # 调试窗口
        debug_group = QGroupBox("法线调试窗口")
        debug_layout = QFormLayout(debug_group)
        
        self.chk_debug_window = QCheckBox("启用法线调试窗口")
        self.chk_debug_window.setToolTip("开启小窗：稀疏显示法线与交点")
        debug_layout.addRow(self.chk_debug_window)
        
        self.spn_debug_stride = QSpinBox()
        self.spn_debug_stride.setRange(1, 200)
        self.spn_debug_stride.setToolTip("小窗法线抽样步长")
        debug_layout.addRow("调试步长:", self.spn_debug_stride)
        
        self.spn_debug_max = QSpinBox()
        self.spn_debug_max.setRange(1, 200)
        self.spn_debug_max.setToolTip("小窗最多显示的法线根数")
        debug_layout.addRow("最大显示数:", self.spn_debug_max)
        
        self.spn_debug_length = QDoubleSpinBox()
        self.spn_debug_length.setRange(0.0, 100.0)
        self.spn_debug_length.setDecimals(2)
        self.spn_debug_length.setToolTip("小窗法线长度，0=自动")
        debug_layout.addRow("法线长度(mm):", self.spn_debug_length)
        
        self.chk_debug_text = QCheckBox("显示Δn文本标注")
        self.chk_debug_text.setToolTip("是否在交点旁标注Δn文本")
        debug_layout.addRow(self.chk_debug_text)
        
        layout.addWidget(debug_group)
        layout.addStretch()
        
        return widget
        
    def create_occlusion_tab(self):
        """创建遮挡区域参数选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 遮挡设置
        occlusion_group = QGroupBox("遮挡区域设置")
        occlusion_layout = QFormLayout(occlusion_group)
        
        self.chk_occlusion_enable = QCheckBox("启用遮挡区域")
        self.chk_occlusion_enable.setToolTip("启用遮挡区域处理，在固定设备区域遮挡")
        occlusion_layout.addRow(self.chk_occlusion_enable)
        
        self.spn_occlusion_dilate = QDoubleSpinBox()
        self.spn_occlusion_dilate.setRange(0, 50)
        self.spn_occlusion_dilate.setDecimals(1)
        self.spn_occlusion_dilate.setToolTip("遮挡区域向外扩张距离")
        occlusion_layout.addRow("扩张距离(mm):", self.spn_occlusion_dilate)
        
        layout.addWidget(occlusion_group)
        
        # 多边形定义
        polygon_group = QGroupBox("多边形定义")
        polygon_layout = QVBoxLayout(polygon_group)
        
        help_label = QLabel("格式：每行一个多边形，坐标用逗号分隔，点用分号分隔\n例如：-50,-50; 30,-30; 30,200; -50,200")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #666; font-size: 11px;")
        polygon_layout.addWidget(help_label)
        
        self.txt_polygons = QTextEdit()
        self.txt_polygons.setMaximumHeight(100)
        self.txt_polygons.setToolTip("定义遮挡区域的多边形顶点坐标")
        polygon_layout.addWidget(self.txt_polygons)
        
        layout.addWidget(polygon_group)
        
        # 环带合成
        band_group = QGroupBox("环带合成")
        band_layout = QFormLayout(band_group)
        
        self.chk_synthesize_band = QCheckBox("遮挡内按G代码合成环带")
        self.chk_synthesize_band.setToolTip("在遮挡区域内按G代码路径合成环带掩模")
        band_layout.addRow(self.chk_synthesize_band)
        
        self.spn_band_halfwidth = QDoubleSpinBox()
        self.spn_band_halfwidth.setRange(0, 100)
        self.spn_band_halfwidth.setDecimals(2)
        self.spn_band_halfwidth.setToolTip("环带半宽，0=自动估计")
        band_layout.addRow("环带半宽(mm):", self.spn_band_halfwidth)
        
        layout.addWidget(band_group)
        layout.addStretch()
        
        return widget
        
    def create_guide_tab(self):
        """创建引导中心线参数选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 基础参数
        basic_group = QGroupBox("基础参数")
        basic_layout = QFormLayout(basic_group)
        
        self.spn_guide_step = QDoubleSpinBox()
        self.spn_guide_step.setRange(0.1, 10.0)
        self.spn_guide_step.setDecimals(2)
        self.spn_guide_step.setToolTip("引导中心线采样步长，影响精度和计算速度")
        basic_layout.addRow("引导步长(mm):", self.spn_guide_step)
        
        self.spn_guide_halfwidth = QDoubleSpinBox()
        self.spn_guide_halfwidth.setRange(0.5, 50.0)
        self.spn_guide_halfwidth.setDecimals(1)
        self.spn_guide_halfwidth.setToolTip("法向搜索半宽，太小易丢失，太大易误匹配")
        basic_layout.addRow("搜索半宽(mm):", self.spn_guide_halfwidth)
        
        self.spn_guide_max_offset = QDoubleSpinBox()
        self.spn_guide_max_offset.setRange(0.1, 50.0)
        self.spn_guide_max_offset.setDecimals(1)
        self.spn_guide_max_offset.setToolTip("允许的最大偏移量，超出则被丢弃")
        basic_layout.addRow("最大偏移(mm):", self.spn_guide_max_offset)
        
        layout.addWidget(basic_group)
        
        # 平滑参数
        smooth_group = QGroupBox("平滑参数")
        smooth_layout = QFormLayout(smooth_group)
        
        self.spn_smooth_win = QSpinBox()
        self.spn_smooth_win.setRange(1, 99)
        self.spn_smooth_win.setToolTip("平滑窗口大小，抵抗噪声但可能过度平滑")
        smooth_layout.addRow("平滑窗口:", self.spn_smooth_win)
        
        self.chk_curvature_adaptive = QCheckBox("曲率自适应平滑")
        self.chk_curvature_adaptive.setToolTip("在弯曲处减少平滑，保持细节")
        smooth_layout.addRow(self.chk_curvature_adaptive)
        
        self.spn_curvature_gamma = QDoubleSpinBox()
        self.spn_curvature_gamma.setRange(1.0, 100.0)
        self.spn_curvature_gamma.setDecimals(1)
        self.spn_curvature_gamma.setToolTip("曲率敏感度，值越大对弯曲越敏感")
        smooth_layout.addRow("曲率敏感度:", self.spn_curvature_gamma)
        
        self.spn_min_smooth_win = QSpinBox()
        self.spn_min_smooth_win.setRange(1, 21)
        self.spn_min_smooth_win.setToolTip("自适应时的最小平滑窗口")
        smooth_layout.addRow("最小平滑窗口:", self.spn_min_smooth_win)
        
        layout.addWidget(smooth_group)
        
        # 梯度控制
        gradient_group = QGroupBox("梯度控制")
        gradient_layout = QFormLayout(gradient_group)
        
        self.spn_max_gradient = QDoubleSpinBox()
        self.spn_max_gradient.setRange(0.001, 1.0)
        self.spn_max_gradient.setDecimals(3)
        self.spn_max_gradient.setToolTip("相邻点间最大允许梯度，限制突变")
        gradient_layout.addRow("最大梯度(mm/mm):", self.spn_max_gradient)
        
        self.spn_max_gap = QSpinBox()
        self.spn_max_gap.setRange(1, 100)
        self.spn_max_gap.setToolTip("允许插值的最大缺口点数")
        gradient_layout.addRow("最大缺口点数:", self.spn_max_gap)
        
        layout.addWidget(gradient_group)
        
        # 拐角处理
        corner_group = QGroupBox("拐角处理")
        corner_layout = QFormLayout(corner_group)
        
        self.chk_corner_ignore = QCheckBox("启用拐角忽略")
        self.chk_corner_ignore.setToolTip("在拐角点附近忽略取点，避免对平滑产生影响")
        corner_layout.addRow(self.chk_corner_ignore)
        
        self.spn_corner_angle = QDoubleSpinBox()
        self.spn_corner_angle.setRange(1.0, 180.0)
        self.spn_corner_angle.setDecimals(1)
        self.spn_corner_angle.setToolTip("拐角判定角度阈值，转角≥该值视为拐角")
        corner_layout.addRow("拐角角度阈值(度):", self.spn_corner_angle)
        
        self.spn_corner_span = QDoubleSpinBox()
        self.spn_corner_span.setRange(0.0, 50.0)
        self.spn_corner_span.setDecimals(2)
        self.spn_corner_span.setToolTip("在拐角点两侧各忽略的弧长半径")
        corner_layout.addRow("忽略半径(mm):", self.spn_corner_span)
        
        layout.addWidget(corner_group)
        layout.addStretch()
        
        return widget
        
    def create_multilayer_tab(self):
        """创建多层特有参数选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 偏差补偿策略
        bias_group = QGroupBox("偏差补偿策略")
        bias_layout = QFormLayout(bias_group)
        
        self.chk_bias_enable = QCheckBox("启用偏差补偿")
        self.chk_bias_enable.setToolTip("第一层用于标定，从第二层开始应用前一层的偏差补偿")
        bias_layout.addRow(self.chk_bias_enable)
        
        self.edit_bias_path = QLineEdit()
        self.btn_browse_bias = QPushButton("浏览")
        self.btn_browse_bias.clicked.connect(self.browse_bias_file)
        bias_file_layout = QHBoxLayout()
        bias_file_layout.addWidget(self.edit_bias_path)
        bias_file_layout.addWidget(self.btn_browse_bias)
        bias_layout.addRow("偏差补偿文件:", bias_file_layout)
        
        layout.addWidget(bias_group)
        
        # 质量控制
        quality_group = QGroupBox("质量控制阈值")
        quality_layout = QFormLayout(quality_group)
        
        self.spn_min_valid_ratio = QDoubleSpinBox()
        self.spn_min_valid_ratio.setRange(0.0, 1.0)
        self.spn_min_valid_ratio.setDecimals(2)
        self.spn_min_valid_ratio.setToolTip("最小有效率阈值，低于此值则Guard报警")
        quality_layout.addRow("最小有效率:", self.spn_min_valid_ratio)
        
        self.spn_max_dev_p95 = QDoubleSpinBox()
        self.spn_max_dev_p95.setRange(0.1, 100.0)
        self.spn_max_dev_p95.setDecimals(2)
        self.spn_max_dev_p95.setToolTip("最大P95偏差阈值，超过此值则Guard报警")
        quality_layout.addRow("最大P95偏差(mm):", self.spn_max_dev_p95)
        
        self.spn_min_plane_inlier = QDoubleSpinBox()
        self.spn_min_plane_inlier.setRange(0.0, 1.0)
        self.spn_min_plane_inlier.setDecimals(2)
        self.spn_min_plane_inlier.setToolTip("最小平面内点率阈值")
        quality_layout.addRow("最小平面内点率:", self.spn_min_plane_inlier)
        
        self.spn_max_long_missing = QDoubleSpinBox()
        self.spn_max_long_missing.setRange(1.0, 200.0)
        self.spn_max_long_missing.setDecimals(1)
        self.spn_max_long_missing.setToolTip("最大允许的长缺失距离")
        quality_layout.addRow("最大长缺失(mm):", self.spn_max_long_missing)
        
        self.spn_max_grad_limit = QDoubleSpinBox()
        self.spn_max_grad_limit.setRange(0.001, 1.0)
        self.spn_max_grad_limit.setDecimals(3)
        self.spn_max_grad_limit.setToolTip("梯度限制阈值")
        quality_layout.addRow("梯度限制(mm/mm):", self.spn_max_grad_limit)
        
        layout.addWidget(quality_group)
        
        # 处理策略
        processing_group = QGroupBox("处理策略")
        processing_layout = QFormLayout(processing_group)
        
        self.chk_auto_next = QCheckBox("自动处理下一层")
        self.chk_auto_next.setToolTip("处理完成后自动进入下一层")
        processing_layout.addRow(self.chk_auto_next)
        
        self.spn_process_delay = QSpinBox()
        self.spn_process_delay.setRange(0, 60)
        self.spn_process_delay.setSuffix(" 秒")
        self.spn_process_delay.setToolTip("自动处理间隔时间")
        processing_layout.addRow("处理延迟:", self.spn_process_delay)
        
        self.spn_max_retry = QSpinBox()
        self.spn_max_retry.setRange(0, 10)
        self.spn_max_retry.setToolTip("处理失败时最大重试次数")
        processing_layout.addRow("最大重试次数:", self.spn_max_retry)
        
        layout.addWidget(processing_group)
        layout.addStretch()
        
        return widget
        
    def browse_bias_file(self):
        """浏览偏差补偿文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择偏差补偿文件", "", "JSON文件 (*.json);;所有文件 (*)"
        )
        if file_path:
            self.edit_bias_path.setText(file_path)
            
    def load_current_parameters(self):
        """加载当前参数到界面控件"""
        cfg = self.controller.cfg
        
        # ROI参数
        self.cmb_roi_mode.setCurrentText(cfg.roi_mode)
        self.spn_roi_x.setValue(cfg.cam_roi_xywh[0])
        self.spn_roi_y.setValue(cfg.cam_roi_xywh[1])
        self.spn_roi_w.setValue(cfg.cam_roi_xywh[2])
        self.spn_roi_h.setValue(cfg.cam_roi_xywh[3])
        
        # 机床坐标ROI
        self.spn_roi_center_x.setValue(cfg.roi_center_xy[0])
        self.spn_roi_center_y.setValue(cfg.roi_center_xy[1])
        self.spn_roi_size.setValue(cfg.roi_size_mm)
        
        self.spn_pixel_size.setValue(cfg.pixel_size_mm)
        self.spn_bounds_margin.setValue(cfg.bounds_margin_mm)
        
        # 最近表面参数
        self.chk_plane_enable.setChecked(cfg.plane_enable)
        self.spn_plane_thresh.setValue(cfg.plane_ransac_thresh_mm)
        self.spn_plane_iters.setValue(cfg.plane_ransac_iters)
        self.spn_plane_sample_cap.setValue(getattr(cfg, 'plane_sample_cap', 120000))
        self.cmb_z_select.setCurrentText(cfg.z_select)
        self.spn_nearest_qlo.setValue(cfg.nearest_qlo)
        self.spn_nearest_qhi.setValue(cfg.nearest_qhi)
        self.spn_depth_margin.setValue(cfg.depth_margin_mm)
        self.spn_morph_open.setValue(cfg.morph_open)
        self.spn_morph_close.setValue(cfg.morph_close)
        self.spn_min_area.setValue(cfg.min_component_area_px)
        
        # 可视化参数
        self.chk_draw_normals.setChecked(cfg.draw_normal_probes)
        self.spn_arrow_stride.setValue(cfg.arrow_stride)
        self.chk_debug_window.setChecked(cfg.debug_normals_window)
        self.spn_debug_stride.setValue(cfg.debug_normals_stride)
        self.spn_debug_max.setValue(cfg.debug_normals_max)
        self.spn_debug_length.setValue(cfg.debug_normals_len_mm or 0.0)
        self.chk_debug_text.setChecked(cfg.debug_normals_text)
        
        # 遮挡参数
        self.chk_occlusion_enable.setChecked(cfg.occ_enable)
        self.spn_occlusion_dilate.setValue(cfg.occ_dilate_mm)
        self.chk_synthesize_band.setChecked(cfg.occ_synthesize_band)
        self.spn_band_halfwidth.setValue(cfg.occ_band_halfwidth_mm or 0.0)
        
        # 多边形文本
        if cfg.occ_polys and len(cfg.occ_polys) > 0:
            poly_lines = []
            for poly in cfg.occ_polys:
                poly_str = "; ".join([f"{x},{y}" for (x, y) in poly])
                poly_lines.append(poly_str)
            self.txt_polygons.setPlainText('\n'.join(poly_lines))
        
        # 引导中心线参数
        self.spn_guide_step.setValue(cfg.guide_step_mm)
        self.spn_guide_halfwidth.setValue(cfg.guide_halfwidth_mm)
        self.spn_guide_max_offset.setValue(cfg.guide_max_offset_mm)
        self.spn_smooth_win.setValue(cfg.guide_smooth_win)
        self.chk_curvature_adaptive.setChecked(cfg.curvature_adaptive)
        self.spn_curvature_gamma.setValue(cfg.curvature_gamma)
        self.spn_min_smooth_win.setValue(cfg.min_smooth_win)
        self.spn_max_gradient.setValue(cfg.guide_max_grad_mm_per_mm)
        self.spn_max_gap.setValue(cfg.max_gap_pts)
        
        # 拐角参数
        self.chk_corner_ignore.setChecked(cfg.corner_ignore_enable)
        self.spn_corner_angle.setValue(cfg.corner_angle_thr_deg)
        self.spn_corner_span.setValue(cfg.corner_ignore_span_mm)
        
        # 多层特有参数
        self.chk_bias_enable.setChecked(cfg.bias_enable)
        self.edit_bias_path.setText(cfg.bias_path)
        
        # 质量控制参数
        self.spn_min_valid_ratio.setValue(cfg.guard_min_valid_ratio)
        self.spn_max_dev_p95.setValue(cfg.guard_max_abs_p95_mm)
        self.spn_min_plane_inlier.setValue(getattr(cfg, 'guard_min_plane_inlier_ratio', 0.25))
        self.spn_max_long_missing.setValue(getattr(cfg, 'guard_long_missing_max_mm', 20.0))
        self.spn_max_grad_limit.setValue(getattr(cfg, 'guard_grad_max_mm_per_mm', 0.08))
        
        # 处理策略参数（默认值）
        self.chk_auto_next.setChecked(False)
        self.spn_process_delay.setValue(2)
        self.spn_max_retry.setValue(3)
        
    def collect_current_parameters(self):
        """收集当前界面参数"""
        params = {}
        
        # ROI参数
        params['roi_mode'] = self.cmb_roi_mode.currentText()
        params['cam_roi_xywh'] = (
            self.spn_roi_x.value(),
            self.spn_roi_y.value(),
            self.spn_roi_w.value(),
            self.spn_roi_h.value()
        )
        params['roi_center_xy'] = (
            self.spn_roi_center_x.value(),
            self.spn_roi_center_y.value()
        )
        params['roi_size_mm'] = self.spn_roi_size.value()
        params['pixel_size_mm'] = self.spn_pixel_size.value()
        params['bounds_margin_mm'] = self.spn_bounds_margin.value()
        
        # 最近表面参数
        params['plane_enable'] = self.chk_plane_enable.isChecked()
        params['plane_ransac_thresh_mm'] = self.spn_plane_thresh.value()
        params['plane_ransac_iters'] = self.spn_plane_iters.value()
        params['plane_sample_cap'] = self.spn_plane_sample_cap.value()
        params['z_select'] = self.cmb_z_select.currentText()
        params['nearest_qlo'] = self.spn_nearest_qlo.value()
        params['nearest_qhi'] = self.spn_nearest_qhi.value()
        params['depth_margin_mm'] = self.spn_depth_margin.value()
        params['morph_open'] = self.spn_morph_open.value()
        params['morph_close'] = self.spn_morph_close.value()
        params['min_component_area_px'] = self.spn_min_area.value()
        
        # 可视化参数
        params['draw_normal_probes'] = self.chk_draw_normals.isChecked()
        params['arrow_stride'] = self.spn_arrow_stride.value()
        params['debug_normals_window'] = self.chk_debug_window.isChecked()
        params['debug_normals_stride'] = self.spn_debug_stride.value()
        params['debug_normals_max'] = self.spn_debug_max.value()
        debug_len = self.spn_debug_length.value()
        params['debug_normals_len_mm'] = None if debug_len == 0.0 else debug_len
        params['debug_normals_text'] = self.chk_debug_text.isChecked()
        
        # 遮挡参数
        params['occ_enable'] = self.chk_occlusion_enable.isChecked()
        params['occ_dilate_mm'] = self.spn_occlusion_dilate.value()
        params['occ_synthesize_band'] = self.chk_synthesize_band.isChecked()
        band_hw = self.spn_band_halfwidth.value()
        params['occ_band_halfwidth_mm'] = None if band_hw == 0.0 else band_hw
        
        # 解析多边形
        polys = []
        for line in self.txt_polygons.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                points = []
                for segment in line.split(';'):
                    segment = segment.strip()
                    if not segment:
                        continue
                    x_str, y_str = segment.split(',')[:2]
                    points.append((float(x_str), float(y_str)))
                if len(points) >= 3:
                    polys.append(points)
            except:
                pass
        params['occ_polys'] = polys if polys else [[(-50, -50), (30, -30), (30, 200), (-50, 200)]]
        
        # 引导中心线参数
        params['guide_step_mm'] = self.spn_guide_step.value()
        params['guide_halfwidth_mm'] = self.spn_guide_halfwidth.value()
        params['guide_max_offset_mm'] = self.spn_guide_max_offset.value()
        params['guide_smooth_win'] = self.spn_smooth_win.value()
        params['curvature_adaptive'] = self.chk_curvature_adaptive.isChecked()
        params['curvature_gamma'] = self.spn_curvature_gamma.value()
        params['min_smooth_win'] = self.spn_min_smooth_win.value()
        params['guide_max_grad_mm_per_mm'] = self.spn_max_gradient.value()
        params['max_gap_pts'] = self.spn_max_gap.value()
        
        # 拐角参数
        params['corner_ignore_enable'] = self.chk_corner_ignore.isChecked()
        params['corner_angle_thr_deg'] = self.spn_corner_angle.value()
        params['corner_ignore_span_mm'] = self.spn_corner_span.value()
        
        # 多层特有参数
        params['bias_enable'] = self.chk_bias_enable.isChecked()
        params['bias_path'] = self.edit_bias_path.text()
        
        # 质量控制参数
        params['guard_min_valid_ratio'] = self.spn_min_valid_ratio.value()
        params['guard_max_abs_p95_mm'] = self.spn_max_dev_p95.value()
        params['guard_min_plane_inlier_ratio'] = self.spn_min_plane_inlier.value()
        params['guard_long_missing_max_mm'] = self.spn_max_long_missing.value()
        params['guard_grad_max_mm_per_mm'] = self.spn_max_grad_limit.value()
        
        # 处理策略参数
        params['auto_next_layer'] = self.chk_auto_next.isChecked()
        params['process_delay_sec'] = self.spn_process_delay.value()
        params['max_retry_count'] = self.spn_max_retry.value()
        
        return params
        
    def save_preset(self):
        """保存参数预设"""
        try:
            preset_name, ok = QInputDialog.getText(self, "保存预设", "预设名称:")
            if ok and preset_name:
                params = self.collect_current_parameters()
                preset_data = {
                    "name": preset_name,
                    "params": params,
                    "save_time": str(time.time())
                }
                
                preset_file = f"multilayer_preset_{preset_name}.json"
                with open(preset_file, 'w', encoding='utf-8') as f:
                    json.dump(preset_data, f, ensure_ascii=False, indent=2)
                    
                QMessageBox.information(self, "成功", f"预设 '{preset_name}' 已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存预设失败: {e}")
            
    def load_preset(self):
        """加载参数预设"""
        try:
            preset_file, _ = QFileDialog.getOpenFileName(
                self, "加载预设", "", "JSON文件 (*.json)"
            )
            if preset_file:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                    
                params = preset_data["params"]
                # 应用参数到控件
                for key, value in params.items():
                    if hasattr(self.controller.cfg, key):
                        setattr(self.controller.cfg, key, value)
                        
                # 重新加载界面
                self.load_current_parameters()
                
                preset_name = preset_data.get("name", "未知")
                QMessageBox.information(self, "成功", f"预设 '{preset_name}' 已加载")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载预设失败: {e}")
            
    def preview_changes(self):
        """预览参数变化效果"""
        try:
            # 应用参数到控制器（临时）
            params = self.collect_current_parameters()
            for key, value in params.items():
                if hasattr(self.controller.cfg, key):
                    setattr(self.controller.cfg, key, value)
            
            QMessageBox.information(self, "预览", "参数已临时应用，可在主界面查看效果。\n点击'应用'保存更改，或'取消'恢复原值。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预览失败: {e}")
            
    def apply_parameters(self):
        """应用参数到控制器"""
        try:
            params = self.collect_current_parameters()
            
            # 应用参数到控制器配置
            for key, value in params.items():
                if hasattr(self.controller.cfg, key):
                    setattr(self.controller.cfg, key, value)
            
            # 发射参数应用信号
            self.parameters_applied.emit(params)
            
            # 关闭对话框
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用参数失败: {e}")
            
    def reset_parameters(self):
        """重置参数"""
        try:
            # 重新加载当前参数
            self.load_current_parameters()
            QMessageBox.information(self, "重置", "参数已重置")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重置失败: {e}")