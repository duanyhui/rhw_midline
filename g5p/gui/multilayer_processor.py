# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - 处理线程模块
"""
import os
import json
import time
import traceback
from typing import Optional, Dict
from PyQt5.QtCore import QThread, pyqtSignal

from multilayer_data import LayerInfo
from controller import AlignController

# ==================== 单层处理线程 ====================

class LayerProcessingThread(QThread):
    """单层处理线程"""
    processing_finished = pyqtSignal(int, dict)  # layer_id, result
    processing_failed = pyqtSignal(int, str)     # layer_id, error
    progress_updated = pyqtSignal(int, str)      # layer_id, status
    
    def __init__(self, layer_info: LayerInfo, controller: AlignController, 
                 previous_bias: Optional[Dict] = None):
        super().__init__()
        self.layer_info = layer_info
        self.controller = controller
        self.previous_bias = previous_bias
        self.start_time = time.time()
        
    def run(self):
        """运行处理"""
        temp_bias_path = None  # 初始化临时文件路径
        
        try:
            layer_id = self.layer_info.layer_id
            self.progress_updated.emit(layer_id, "开始处理...")
            
            # 1. 设置G代码路径
            self.controller.cfg.gcode_path = self.layer_info.gcode_path
            self.progress_updated.emit(layer_id, f"加载G代码: {os.path.basename(self.layer_info.gcode_path)}")
            
            # 2. 如果有前层偏差补偿，则应用
            temp_bias_path = None
            if self.previous_bias and layer_id > 1:
                self.controller.cfg.bias_enable = True
                # 临时保存bias文件
                temp_bias_path = f"temp_bias_layer_{layer_id}_{int(time.time())}.json"
                with open(temp_bias_path, 'w', encoding='utf-8') as f:
                    json.dump(self.previous_bias, f, ensure_ascii=False, indent=2)
                self.controller.cfg.bias_path = temp_bias_path
                self.progress_updated.emit(layer_id, "应用前层偏差补偿...")
            else:
                self.controller.cfg.bias_enable = False
                self.progress_updated.emit(layer_id, "无前层偏差补偿")
                
            # 3. 相机采图和处理
            self.progress_updated.emit(layer_id, "相机采图中...")
            result = self.controller.process_single_frame(for_export=False)
            
            # 4. 检查处理结果质量
            metrics = result.get('metrics', {})
            valid_ratio = metrics.get('valid_ratio', 0.0)
            dev_p95 = metrics.get('dev_p95', 0.0)
            
            # 基础质量检查
            quality_issues = []
            if valid_ratio < 0.3:
                quality_issues.append(f"有效率过低: {valid_ratio:.1%}")
            if dev_p95 > 15.0:
                quality_issues.append(f"偏差过大: {dev_p95:.2f}mm")
                
            if quality_issues:
                self.progress_updated.emit(layer_id, f"质量警告: {'; '.join(quality_issues)}")
            
            # 5. 生成当前层的偏差补偿
            self.progress_updated.emit(layer_id, "生成偏差补偿...")
            
            if layer_id == 1:
                # 第一层：只记录偏差补偿，不导出纠偏
                bias_path = self.controller.save_bias_from_current()
                result['bias_comp_path'] = bias_path
                result['is_calibration_layer'] = True
                result['layer_type'] = 'calibration'
                self.progress_updated.emit(layer_id, "第一层标定完成")
            else:
                # 后续层：导出纠偏并记录偏差补偿
                self.progress_updated.emit(layer_id, "导出纠偏数据...")
                
                try:
                    correction = self.controller.export_corrected()
                    result['correction'] = correction
                    result['is_calibration_layer'] = False
                    result['layer_type'] = 'correction'
                    
                    # 保存当前层的偏差补偿（用于下一层）
                    bias_path = self.controller.save_bias_from_current()
                    result['bias_comp_path'] = bias_path
                    
                    self.progress_updated.emit(layer_id, "纠偏数据导出完成")
                    
                except Exception as e:
                    # 导出失败，但仍然保存偏差补偿
                    self.progress_updated.emit(layer_id, f"导出警告: {str(e)}")
                    bias_path = self.controller.save_bias_from_current()
                    result['bias_comp_path'] = bias_path
                    result['export_error'] = str(e)
            
            # 6. 添加处理时间和质量指标
            processing_time = time.time() - self.start_time
            result['processing_time'] = processing_time
            result['layer_id'] = layer_id
            result['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 7. 读取并缓存生成的偏差补偿数据
            try:
                if 'bias_comp_path' in result:
                    with open(result['bias_comp_path'], 'r', encoding='utf-8') as f:
                        bias_data = json.load(f)
                        result['bias_comp_data'] = bias_data
            except Exception as e:
                self.progress_updated.emit(layer_id, f"偏差补偿读取警告: {e}")
            
            # 8. 清理临时文件
            if temp_bias_path and os.path.exists(temp_bias_path):
                try:
                    os.remove(temp_bias_path)
                except Exception as e:
                    print(f"清理临时文件失败: {e}")
                    
            self.progress_updated.emit(layer_id, "处理完成")
            self.processing_finished.emit(layer_id, result)
            
        except Exception as e:
            error_msg = f"第{self.layer_info.layer_id}层处理失败: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
            self.processing_failed.emit(self.layer_info.layer_id, error_msg)
            
            # 清理临时文件
            if temp_bias_path and os.path.exists(temp_bias_path):
                try:
                    os.remove(temp_bias_path)
                except:
                    pass

# ==================== 批量处理线程 ====================

class BatchProcessingThread(QThread):
    """批量处理线程"""
    batch_progress = pyqtSignal(int, int, str)  # current_layer, total_layers, status
    layer_completed = pyqtSignal(int, dict)     # layer_id, result
    layer_failed = pyqtSignal(int, str)         # layer_id, error
    batch_finished = pyqtSignal(bool, str)      # success, message
    
    def __init__(self, layers: Dict[int, LayerInfo], controller: AlignController,
                 start_layer: int = 1, end_layer: Optional[int] = None):
        super().__init__()
        self.layers = layers
        self.controller = controller
        self.start_layer = start_layer
        self.end_layer = end_layer or max(layers.keys())
        self.should_stop = False
        
    def run(self):
        """批量处理运行"""
        try:
            total_layers = self.end_layer - self.start_layer + 1
            processed_count = 0
            failed_count = 0
            
            previous_bias = None
            
            for layer_id in range(self.start_layer, self.end_layer + 1):
                if self.should_stop:
                    self.batch_finished.emit(False, "用户取消")
                    return
                    
                if layer_id not in self.layers:
                    self.layer_failed.emit(layer_id, f"第{layer_id}层不存在")
                    failed_count += 1
                    continue
                    
                self.batch_progress.emit(layer_id, total_layers, f"处理第{layer_id}层...")
                
                # 处理单层
                layer_info = self.layers[layer_id]
                single_processor = LayerProcessingThread(layer_info, self.controller, previous_bias)
                
                # 连接信号（同步处理）
                result_received = False
                processing_result = None
                processing_error = None
                
                def on_finished(lid, result):
                    nonlocal result_received, processing_result
                    processing_result = result
                    result_received = True
                    
                def on_failed(lid, error):
                    nonlocal result_received, processing_error
                    processing_error = error
                    result_received = True
                    
                single_processor.processing_finished.connect(on_finished)
                single_processor.processing_failed.connect(on_failed)
                
                # 启动并等待完成
                single_processor.start()
                single_processor.wait()  # 等待线程完成
                
                if processing_result:
                    self.layer_completed.emit(layer_id, processing_result)
                    processed_count += 1
                    
                    # 提取偏差补偿用于下一层
                    if 'bias_comp_data' in processing_result:
                        previous_bias = processing_result['bias_comp_data']
                        
                elif processing_error:
                    self.layer_failed.emit(layer_id, processing_error)
                    failed_count += 1
                    
                # 更新进度
                progress_msg = f"已完成 {processed_count}/{total_layers} 层"
                if failed_count > 0:
                    progress_msg += f", 失败 {failed_count} 层"
                self.batch_progress.emit(layer_id, total_layers, progress_msg)
                
            # 批量处理完成
            if failed_count == 0:
                self.batch_finished.emit(True, f"批量处理完成，共处理 {processed_count} 层")
            else:
                self.batch_finished.emit(False, f"批量处理完成，成功 {processed_count} 层，失败 {failed_count} 层")
                
        except Exception as e:
            error_msg = f"批量处理异常: {str(e)}\n{traceback.format_exc()}"
            self.batch_finished.emit(False, error_msg)
            
    def stop(self):
        """停止批量处理"""
        self.should_stop = True

# ==================== 后台自动处理线程 ====================

class AutoProcessingThread(QThread):
    """后台自动处理线程，监听PLC信号自动处理"""
    auto_processing_status = pyqtSignal(str)  # 状态信息
    layer_auto_completed = pyqtSignal(int, dict)  # 自动完成的层
    
    def __init__(self, layers: Dict[int, LayerInfo], controller: AlignController):
        super().__init__()
        self.layers = layers
        self.controller = controller
        self.enabled = False
        self.current_bias = None
        
    def enable_auto_processing(self, enabled: bool):
        """启用/禁用自动处理"""
        self.enabled = enabled
        self.auto_processing_status.emit(f"自动处理: {'启用' if enabled else '禁用'}")
        
    def set_current_bias(self, bias_data: Dict):
        """设置当前偏差补偿数据"""
        self.current_bias = bias_data
        
    def process_layer_auto(self, layer_id: int):
        """自动处理指定层"""
        if not self.enabled or layer_id not in self.layers:
            return
            
        try:
            self.auto_processing_status.emit(f"自动处理第{layer_id}层...")
            
            layer_info = self.layers[layer_id]
            processor = LayerProcessingThread(layer_info, self.controller, self.current_bias)
            
            # 处理完成处理
            def on_auto_finished(lid, result):
                self.layer_auto_completed.emit(lid, result)
                if 'bias_comp_data' in result:
                    self.current_bias = result['bias_comp_data']
                self.auto_processing_status.emit(f"第{lid}层自动处理完成")
                
            def on_auto_failed(lid, error):
                self.auto_processing_status.emit(f"第{lid}层自动处理失败: {error}")
                
            processor.processing_finished.connect(on_auto_finished)
            processor.processing_failed.connect(on_auto_failed)
            
            processor.start()
            processor.wait()  # 等待完成
            
        except Exception as e:
            self.auto_processing_status.emit(f"自动处理异常: {e}")
            
    def run(self):
        """运行自动处理监听"""
        while True:
            self.msleep(100)  # 100ms间隔
            if not self.enabled:
                self.msleep(1000)  # 禁用时降低检查频率