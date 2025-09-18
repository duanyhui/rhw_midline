# -*- coding: utf-8 -*-
"""
高级参数调节界面 - 第二部分
包含遮挡区域、引导中心线参数和其他功能方法
"""

def create_occlusion_tab(self):
    """创建遮挡区域参数选项卡"""
    tab = QWidget()
    scroll = QScrollArea()
    scroll.setWidget(tab)
    scroll.setWidgetResizable(True)
    layout = QVBoxLayout(tab)
    
    # 遮挡设置
    occlusion_group = QGroupBox("遮挡区域设置")
    occlusion_layout = QFormLayout(occlusion_group)
    
    self.chk_occlusion_enable = QCheckBox("启用遮挡区域")
    self.chk_occlusion_enable.stateChanged.connect(self.on_parameter_changed)
    occlusion_layout.addRow(self.chk_occlusion_enable)
    
    self.spn_occlusion_dilate = QDoubleSpinBox()
    self.spn_occlusion_dilate.setRange(0, 50)
    self.spn_occlusion_dilate.setDecimals(1)
    self.spn_occlusion_dilate.valueChanged.connect(self.on_parameter_changed)
    occlusion_layout.addRow("扩张距离(mm):", self.spn_occlusion_dilate)
    
    # 多边形定义
    polygon_group = QGroupBox("多边形定义")
    polygon_layout = QVBoxLayout(polygon_group)
    
    polygon_help = QLabel("格式：每行一个多边形，坐标用逗号分隔，点用分号分隔\n例如：-50,-50; 30,-30; 30,200; -50,200")
    polygon_help.setWordWrap(True)
    polygon_help.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
    polygon_layout.addWidget(polygon_help)
    
    self.txt_polygons = QTextEdit()
    self.txt_polygons.setMaximumHeight(120)
    self.txt_polygons.textChanged.connect(self.on_parameter_changed)
    polygon_layout.addWidget(self.txt_polygons)
    
    # 环带合成
    band_group = QGroupBox("环带合成")
    band_layout = QFormLayout(band_group)
    
    self.chk_synthesize_band = QCheckBox("遮挡内按G代码合成环带")
    self.chk_synthesize_band.stateChanged.connect(self.on_parameter_changed)
    band_layout.addRow(self.chk_synthesize_band)
    
    self.spn_band_halfwidth = QDoubleSpinBox()
    self.spn_band_halfwidth.setRange(0, 100)
    self.spn_band_halfwidth.setDecimals(2)
    self.spn_band_halfwidth.valueChanged.connect(self.on_parameter_changed)
    band_layout.addRow("环带半宽(mm):", self.spn_band_halfwidth)
    
    layout.addWidget(occlusion_group)
    layout.addWidget(polygon_group)
    layout.addWidget(band_group)
    layout.addStretch()
    
    return scroll
    
def create_guide_tab(self):
    """创建引导中心线参数选项卡"""
    tab = QWidget()
    scroll = QScrollArea()
    scroll.setWidget(tab)
    scroll.setWidgetResizable(True)
    layout = QVBoxLayout(tab)
    
    # 基础参数
    basic_group = QGroupBox("基础参数")
    basic_layout = QFormLayout(basic_group)
    
    self.spn_guide_step = QDoubleSpinBox()
    self.spn_guide_step.setRange(0.1, 10.0)
    self.spn_guide_step.setDecimals(2)
    self.spn_guide_step.valueChanged.connect(self.on_parameter_changed)
    basic_layout.addRow("引导步长(mm):", self.spn_guide_step)
    
    self.spn_guide_halfwidth = QDoubleSpinBox()
    self.spn_guide_halfwidth.setRange(0.5, 50.0)
    self.spn_guide_halfwidth.setDecimals(1)
    self.spn_guide_halfwidth.valueChanged.connect(self.on_parameter_changed)
    basic_layout.addRow("搜索半宽(mm):", self.spn_guide_halfwidth)
    
    self.spn_guide_max_offset = QDoubleSpinBox()
    self.spn_guide_max_offset.setRange(0.1, 50.0)
    self.spn_guide_max_offset.setDecimals(1)
    self.spn_guide_max_offset.valueChanged.connect(self.on_parameter_changed)
    basic_layout.addRow("最大偏移(mm):", self.spn_guide_max_offset)
    
    # 平滑参数
    smooth_group = QGroupBox("平滑参数")
    smooth_layout = QFormLayout(smooth_group)
    
    self.spn_smooth_win = QSpinBox()
    self.spn_smooth_win.setRange(1, 99)
    self.spn_smooth_win.valueChanged.connect(self.on_parameter_changed)
    smooth_layout.addRow("平滑窗口:", self.spn_smooth_win)
    
    self.chk_curvature_adaptive = QCheckBox("曲率自适应平滑")
    self.chk_curvature_adaptive.stateChanged.connect(self.on_parameter_changed)
    smooth_layout.addRow(self.chk_curvature_adaptive)
    
    self.spn_curvature_gamma = QDoubleSpinBox()
    self.spn_curvature_gamma.setRange(1.0, 100.0)
    self.spn_curvature_gamma.setDecimals(1)
    self.spn_curvature_gamma.valueChanged.connect(self.on_parameter_changed)
    smooth_layout.addRow("曲率敏感度:", self.spn_curvature_gamma)
    
    self.spn_min_smooth_win = QSpinBox()
    self.spn_min_smooth_win.setRange(1, 21)
    self.spn_min_smooth_win.valueChanged.connect(self.on_parameter_changed)
    smooth_layout.addRow("最小平滑窗口:", self.spn_min_smooth_win)
    
    # 梯度控制
    gradient_group = QGroupBox("梯度控制")
    gradient_layout = QFormLayout(gradient_group)
    
    self.spn_max_gradient = QDoubleSpinBox()
    self.spn_max_gradient.setRange(0.001, 1.0)
    self.spn_max_gradient.setDecimals(3)
    self.spn_max_gradient.valueChanged.connect(self.on_parameter_changed)
    gradient_layout.addRow("最大梯度(mm/mm):", self.spn_max_gradient)
    
    self.spn_max_gap = QSpinBox()
    self.spn_max_gap.setRange(1, 100)
    self.spn_max_gap.valueChanged.connect(self.on_parameter_changed)
    gradient_layout.addRow("最大缺口点数:", self.spn_max_gap)
    
    # 拐角处理
    corner_group = QGroupBox("拐角处理")
    corner_layout = QFormLayout(corner_group)
    
    self.chk_corner_ignore = QCheckBox("启用拐角忽略")
    self.chk_corner_ignore.stateChanged.connect(self.on_parameter_changed)
    corner_layout.addRow(self.chk_corner_ignore)
    
    self.spn_corner_angle = QDoubleSpinBox()
    self.spn_corner_angle.setRange(1.0, 180.0)
    self.spn_corner_angle.setDecimals(1)
    self.spn_corner_angle.valueChanged.connect(self.on_parameter_changed)
    corner_layout.addRow("拐角阈值(度):", self.spn_corner_angle)
    
    self.spn_corner_span = QDoubleSpinBox()
    self.spn_corner_span.setRange(0.0, 50.0)
    self.spn_corner_span.setDecimals(2)
    self.spn_corner_span.valueChanged.connect(self.on_parameter_changed)
    corner_layout.addRow("忽略半径(mm):", self.spn_corner_span)
    
    layout.addWidget(basic_group)
    layout.addWidget(smooth_group)
    layout.addWidget(gradient_group)
    layout.addWidget(corner_group)
    layout.addStretch()
    
    return scroll

def create_preview_panel(self):
    """创建预览面板"""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    
    # 顶部控制按钮
    control_layout = QHBoxLayout()
    
    self.btn_preview = QPushButton("实时预览")
    self.btn_preview.setCheckable(True)
    self.btn_preview.clicked.connect(self.toggle_preview)
    control_layout.addWidget(self.btn_preview)
    
    self.chk_auto_preview = QCheckBox("自动预览")
    self.chk_auto_preview.setChecked(True)
    control_layout.addWidget(self.chk_auto_preview)
    
    control_layout.addStretch()
    
    self.lbl_preview_status = QLabel("就绪")
    self.lbl_preview_status.setStyleSheet("QLabel { color: #666; }")
    control_layout.addWidget(self.lbl_preview_status)
    
    layout.addLayout(control_layout)
    
    # 预览选项卡
    self.preview_tabs = QTabWidget()
    
    # 对齐叠加预览
    self.lbl_overlay = QLabel("预览图像")
    self.lbl_overlay.setAlignment(Qt.AlignCenter)
    self.lbl_overlay.setMinimumHeight(400)
    self.lbl_overlay.setStyleSheet("QLabel { border: 1px solid #ccc; background: white; }")
    overlay_tab = QWidget()
    overlay_layout = QVBoxLayout(overlay_tab)
    overlay_layout.addWidget(self.lbl_overlay)
    self.preview_tabs.addTab(overlay_tab, "对齐叠加")
    
    # 法线采样预览
    self.lbl_normals = QLabel("法线预览")
    self.lbl_normals.setAlignment(Qt.AlignCenter)
    self.lbl_normals.setMinimumHeight(400)
    self.lbl_normals.setStyleSheet("QLabel { border: 1px solid #ccc; background: white; }")
    normals_tab = QWidget()
    normals_layout = QVBoxLayout(normals_tab)
    normals_layout.addWidget(self.lbl_normals)
    self.preview_tabs.addTab(normals_tab, "法线采样")
    
    # 顶视图预览
    self.lbl_topview = QLabel("顶视图")
    self.lbl_topview.setAlignment(Qt.AlignCenter)
    self.lbl_topview.setMinimumHeight(400)
    self.lbl_topview.setStyleSheet("QLabel { border: 1px solid #ccc; background: white; }")
    topview_tab = QWidget()
    topview_layout = QVBoxLayout(topview_tab)
    topview_layout.addWidget(self.lbl_topview)
    self.preview_tabs.addTab(topview_tab, "顶视图")
    
    # 最近表面预览
    self.lbl_surface = QLabel("最近表面")
    self.lbl_surface.setAlignment(Qt.AlignCenter)
    self.lbl_surface.setMinimumHeight(400)
    self.lbl_surface.setStyleSheet("QLabel { border: 1px solid #ccc; background: white; }")
    surface_tab = QWidget()
    surface_layout = QVBoxLayout(surface_tab)
    surface_layout.addWidget(self.lbl_surface)
    self.preview_tabs.addTab(surface_tab, "最近表面")
    
    layout.addWidget(self.preview_tabs)
    
    # 参数统计信息
    stats_group = QGroupBox("实时统计")
    stats_layout = QVBoxLayout(stats_group)
    
    self.lbl_stats = QLabel("暂无统计信息")
    self.lbl_stats.setFont(QFont("Consolas", 10))
    self.lbl_stats.setAlignment(Qt.AlignLeft)
    stats_layout.addWidget(self.lbl_stats)
    
    layout.addWidget(stats_group)
    
    return panel

# 将新方法添加到 AdvancedParametersDialog 类中
AdvancedParametersDialog.create_occlusion_tab = create_occlusion_tab
AdvancedParametersDialog.create_guide_tab = create_guide_tab  
AdvancedParametersDialog.create_preview_panel = create_preview_panel

# 继续定义其他方法
def setup_connections(self):
    """设置信号连接"""
    # 参数选项卡切换时自动预览
    self.params_tabs.currentChanged.connect(self.on_tab_changed)
    
def on_tab_changed(self, index):
    """选项卡切换时触发预览"""
    if self.chk_auto_preview.isChecked():
        self.schedule_preview()

def on_parameter_changed(self):
    """参数变化时触发预览"""
    if self.chk_auto_preview.isChecked():
        self.schedule_preview()

def schedule_preview(self):
    """计划预览（延迟执行）"""
    self.preview_timer.stop()
    self.preview_timer.start(500)  # 500ms延迟

def toggle_preview(self, checked):
    """切换预览模式"""
    if checked:
        self.start_preview()
        self.btn_preview.setText("停止预览")
    else:
        self.stop_preview()
        self.btn_preview.setText("实时预览")

def start_preview(self):
    """开始预览"""
    if self.preview_thread and self.preview_thread.isRunning():
        return
        
    try:
        self.lbl_preview_status.setText("处理中...")
        self.lbl_preview_status.setStyleSheet("QLabel { color: orange; }")
        
        # 收集当前参数
        params = self.collect_current_parameters()
        
        # 启动预览线程
        self.preview_thread = ParameterPreviewThread(self.controller, params)
        self.preview_thread.preview_ready.connect(self.on_preview_ready)
        self.preview_thread.preview_failed.connect(self.on_preview_failed)
        self.preview_thread.start()
        
    except Exception as e:
        self.lbl_preview_status.setText(f"预览失败: {e}")
        self.lbl_preview_status.setStyleSheet("QLabel { color: red; }")

def stop_preview(self):
    """停止预览"""
    if self.preview_thread and self.preview_thread.isRunning():
        self.preview_thread.terminate()
        self.preview_thread.wait()
    self.lbl_preview_status.setText("已停止")
    self.lbl_preview_status.setStyleSheet("QLabel { color: #666; }")

def on_preview_ready(self, result):
    """预览完成"""
    self.current_preview_result = result
    self.update_preview_images(result)
    self.update_preview_stats(result)
    
    self.lbl_preview_status.setText("预览完成")
    self.lbl_preview_status.setStyleSheet("QLabel { color: green; }")

def on_preview_failed(self, error):
    """预览失败"""
    self.lbl_preview_status.setText(f"预览失败: {error}")
    self.lbl_preview_status.setStyleSheet("QLabel { color: red; }")

def update_preview_images(self, result):
    """更新预览图像"""
    # 更新对齐叠加图
    if 'vis_cmp' in result:
        self.display_image_in_label(self.lbl_overlay, result['vis_cmp'])
    
    # 更新法线预览图
    if 'vis_probe' in result:
        self.display_image_in_label(self.lbl_normals, result['vis_probe'])
    
    # 更新顶视图
    if 'vis_top' in result:
        self.display_image_in_label(self.lbl_topview, result['vis_top'])
        
    # 更新最近表面
    if 'vis_nearest' in result:
        self.display_image_in_label(self.lbl_surface, result['vis_nearest'])

def display_image_in_label(self, label, img_array):
    """在标签中显示图像"""
    try:
        if img_array is None:
            label.setText("无图像数据")
            return
            
        qimage = np_to_qimage(img_array)
        if qimage:
            pixmap = QPixmap.fromImage(qimage)
            scaled_pixmap = pixmap.scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
        else:
            label.setText("图像格式错误")
    except Exception as e:
        label.setText(f"显示错误: {e}")

def update_preview_stats(self, result):
    """更新预览统计"""
    try:
        metrics = result.get('metrics', {})
        if not metrics:
            self.lbl_stats.setText("暂无统计信息")
            return
            
        stats_text = []
        stats_text.append(f"有效率: {metrics.get('valid_ratio', 0):.2%}")
        stats_text.append(f"偏差P95: {metrics.get('dev_p95', 0):.3f} mm")
        stats_text.append(f"偏差均值: {metrics.get('dev_mean', 0):+.3f} mm")
        stats_text.append(f"偏差中位数: {metrics.get('dev_median', 0):+.3f} mm")
        
        if 'plane_inlier_ratio' in metrics:
            stats_text.append(f"平面内点率: {metrics['plane_inlier_ratio']:.2%}")
            
        stats_text.append(f"最长缺失: {metrics.get('longest_missing_mm', 0):.1f} mm")
        
        self.lbl_stats.setText('\n'.join(stats_text))
        
    except Exception as e:
        self.lbl_stats.setText(f"统计错误: {e}")

# 将这些方法绑定到类
AdvancedParametersDialog.setup_connections = setup_connections
AdvancedParametersDialog.on_tab_changed = on_tab_changed
AdvancedParametersDialog.on_parameter_changed = on_parameter_changed
AdvancedParametersDialog.schedule_preview = schedule_preview
AdvancedParametersDialog.toggle_preview = toggle_preview
AdvancedParametersDialog.start_preview = start_preview
AdvancedParametersDialog.stop_preview = stop_preview
AdvancedParametersDialog.on_preview_ready = on_preview_ready
AdvancedParametersDialog.on_preview_failed = on_preview_failed
AdvancedParametersDialog.update_preview_images = update_preview_images
AdvancedParametersDialog.display_image_in_label = display_image_in_label
AdvancedParametersDialog.update_preview_stats = update_preview_stats