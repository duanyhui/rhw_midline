# -*- coding: utf-8 -*-
"""
高级参数调节界面 - 工具方法
包含参数加载、保存、重置等功能
"""

def _backup_config(self):
    """备份当前配置"""
    return {
        'roi_mode': self.controller.cfg.roi_mode,
        'cam_roi_xywh': self.controller.cfg.cam_roi_xywh,
        'roi_center_xy': self.controller.cfg.roi_center_xy,
        'roi_size_mm': self.controller.cfg.roi_size_mm,
        'pixel_size_mm': self.controller.cfg.pixel_size_mm,
        'bounds_margin_mm': self.controller.cfg.bounds_margin_mm,
        'plane_enable': self.controller.cfg.plane_enable,
        'plane_ransac_thresh_mm': self.controller.cfg.plane_ransac_thresh_mm,
        'plane_ransac_iters': self.controller.cfg.plane_ransac_iters,
        'z_select': self.controller.cfg.z_select,
        'nearest_qlo': self.controller.cfg.nearest_qlo,
        'nearest_qhi': self.controller.cfg.nearest_qhi,
        'depth_margin_mm': self.controller.cfg.depth_margin_mm,
        'morph_open': self.controller.cfg.morph_open,
        'morph_close': self.controller.cfg.morph_close,
        'min_component_area_px': self.controller.cfg.min_component_area_px,
        'draw_normal_probes': self.controller.cfg.draw_normal_probes,
        'arrow_stride': self.controller.cfg.arrow_stride,
        'debug_normals_window': self.controller.cfg.debug_normals_window,
        'debug_normals_stride': self.controller.cfg.debug_normals_stride,
        'debug_normals_max': self.controller.cfg.debug_normals_max,
        'debug_normals_len_mm': self.controller.cfg.debug_normals_len_mm,
        'debug_normals_text': self.controller.cfg.debug_normals_text,
        'occ_enable': self.controller.cfg.occ_enable,
        'occ_dilate_mm': self.controller.cfg.occ_dilate_mm,
        'occ_polys': self.controller.cfg.occ_polys,
        'occ_synthesize_band': self.controller.cfg.occ_synthesize_band,
        'occ_band_halfwidth_mm': self.controller.cfg.occ_band_halfwidth_mm,
        'guide_step_mm': self.controller.cfg.guide_step_mm,
        'guide_halfwidth_mm': self.controller.cfg.guide_halfwidth_mm,
        'guide_smooth_win': self.controller.cfg.guide_smooth_win,
        'guide_max_offset_mm': self.controller.cfg.guide_max_offset_mm,
        'guide_max_grad_mm_per_mm': self.controller.cfg.guide_max_grad_mm_per_mm,
        'max_gap_pts': self.controller.cfg.max_gap_pts,
        'curvature_adaptive': self.controller.cfg.curvature_adaptive,
        'curvature_gamma': self.controller.cfg.curvature_gamma,
        'min_smooth_win': self.controller.cfg.min_smooth_win,
        'corner_ignore_enable': self.controller.cfg.corner_ignore_enable,
        'corner_angle_thr_deg': self.controller.cfg.corner_angle_thr_deg,
        'corner_ignore_span_mm': self.controller.cfg.corner_ignore_span_mm,
        'guard_min_valid_ratio': self.controller.cfg.guard_min_valid_ratio,
        'guard_max_abs_p95_mm': self.controller.cfg.guard_max_abs_p95_mm,
    }

def load_current_parameters(self):
    """加载当前参数到界面控件"""
    cfg = self.controller.cfg
    
    # ROI参数
    self.cmb_roi_mode.setCurrentText(cfg.roi_mode)
    self.spn_roi_x.setValue(cfg.cam_roi_xywh[0])
    self.spn_roi_y.setValue(cfg.cam_roi_xywh[1])
    self.spn_roi_w.setValue(cfg.cam_roi_xywh[2])
    self.spn_roi_h.setValue(cfg.cam_roi_xywh[3])
    self.spn_pixel_size.setValue(cfg.pixel_size_mm)
    self.spn_bounds_margin.setValue(cfg.bounds_margin_mm)
    
    # 最近表面参数
    self.chk_plane_enable.setChecked(cfg.plane_enable)
    self.spn_plane_thresh.setValue(cfg.plane_ransac_thresh_mm)
    self.spn_plane_iters.setValue(cfg.plane_ransac_iters)
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
    params['pixel_size_mm'] = self.spn_pixel_size.value()
    params['bounds_margin_mm'] = self.spn_bounds_margin.value()
    
    # 最近表面参数
    params['plane_enable'] = self.chk_plane_enable.isChecked()
    params['plane_ransac_thresh_mm'] = self.spn_plane_thresh.value()
    params['plane_ransac_iters'] = self.spn_plane_iters.value()
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
    
    return params

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
        
        QMessageBox.information(self, "成功", "参数已应用")
        
    except Exception as e:
        QMessageBox.critical(self, "错误", f"应用参数失败: {e}")

def reset_parameters(self):
    """重置参数到原始值"""
    try:
        # 恢复原始配置
        for key, value in self.original_config.items():
            if hasattr(self.controller.cfg, key):
                setattr(self.controller.cfg, key, value)
        
        # 重新加载到界面
        self.load_current_parameters()
        
        # 触发预览
        if self.chk_auto_preview.isChecked():
            self.schedule_preview()
            
        QMessageBox.information(self, "重置", "参数已重置为打开对话框时的值")
        
    except Exception as e:
        QMessageBox.critical(self, "错误", f"重置参数失败: {e}")

def load_preset(self):
    """加载预设参数"""
    try:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载参数预设", "", "JSON文件 (*.json)"
        )
        
        if not file_path:
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            preset_params = json.load(f)
            
        # 应用预设参数到控制器
        for key, value in preset_params.items():
            if hasattr(self.controller.cfg, key):
                setattr(self.controller.cfg, key, value)
        
        # 重新加载到界面
        self.load_current_parameters()
        
        # 触发预览
        if self.chk_auto_preview.isChecked():
            self.schedule_preview()
            
        QMessageBox.information(self, "成功", f"已加载预设: {file_path}")
        
    except Exception as e:
        QMessageBox.critical(self, "错误", f"加载预设失败: {e}")

def save_preset(self):
    """保存当前参数为预设"""
    try:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存参数预设", "advanced_params_preset.json", "JSON文件 (*.json)"
        )
        
        if not file_path:
            return
            
        params = self.collect_current_parameters()
        
        # 添加元数据
        preset_data = {
            'metadata': {
                'name': '高级参数预设',
                'description': '包含ROI、最近表面、可视化、遮挡、引导等所有高级参数',
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0'
            },
            'parameters': params
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(preset_data, f, ensure_ascii=False, indent=2)
            
        QMessageBox.information(self, "成功", f"预设已保存: {file_path}")
        
    except Exception as e:
        QMessageBox.critical(self, "错误", f"保存预设失败: {e}")

def closeEvent(self, event):
    """关闭事件"""
    # 停止预览线程
    self.stop_preview()
    
    # 恢复原始配置（如果用户没有点击应用）
    if not hasattr(self, '_applied'):
        for key, value in self.original_config.items():
            if hasattr(self.controller.cfg, key):
                setattr(self.controller.cfg, key, value)
    
    event.accept()

def reject(self):
    """取消按钮"""
    # 恢复原始配置
    for key, value in self.original_config.items():
        if hasattr(self.controller.cfg, key):
            setattr(self.controller.cfg, key, value)
    
    super().reject()

def accept(self):
    """确认按钮"""
    self._applied = True
    super().accept()

# 绑定方法到类（需要在主文件中import这些方法）
def bind_utils_to_class():
    """将工具方法绑定到AdvancedParametersDialog类"""
    import advanced_params_dialog
    
    # 绑定所有工具方法
    advanced_params_dialog.AdvancedParametersDialog._backup_config = _backup_config
    advanced_params_dialog.AdvancedParametersDialog.load_current_parameters = load_current_parameters
    advanced_params_dialog.AdvancedParametersDialog.collect_current_parameters = collect_current_parameters
    advanced_params_dialog.AdvancedParametersDialog.apply_parameters = apply_parameters
    advanced_params_dialog.AdvancedParametersDialog.reset_parameters = reset_parameters
    advanced_params_dialog.AdvancedParametersDialog.load_preset = load_preset
    advanced_params_dialog.AdvancedParametersDialog.save_preset = save_preset
    advanced_params_dialog.AdvancedParametersDialog.closeEvent = closeEvent
    advanced_params_dialog.AdvancedParametersDialog.reject = reject
    advanced_params_dialog.AdvancedParametersDialog.accept = accept