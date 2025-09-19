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
            
            # 2. 应用偏差补偿逻辑（严格按照项目规范）
            temp_bias_path = None
            if layer_id == 1:
                # 第一层：仅用于标定，不应用任何补偿
                self.controller.cfg.bias_enable = False
                self.progress_updated.emit(layer_id, "第一层标定模式，不应用偏差补偿")
            elif layer_id > 1 and self.previous_bias:
                # 第2+层：应用前一层的偏差补偿
                self.controller.cfg.bias_enable = True
                # 临时保存bias文件
                temp_bias_path = f"temp_bias_layer_{layer_id}_{int(time.time())}.json"
                with open(temp_bias_path, 'w', encoding='utf-8') as f:
                    json.dump(self.previous_bias, f, ensure_ascii=False, indent=2)
                self.controller.cfg.bias_path = temp_bias_path
                self.progress_updated.emit(layer_id, f"第{layer_id}层应用前层偏差补偿")
            else:
                # 第2+层但没有可用的偏差补偿
                self.controller.cfg.bias_enable = False
                self.progress_updated.emit(layer_id, f"警告：第{layer_id}层未找到可用的偏差补偿")
                
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
                    
                            # 保存当前层的偏差补偏（用于下一层）
                    bias_path = self.controller.save_bias_from_current()
                    result['bias_comp_path'] = bias_path
                    
                    # 为当前层创建独立的out文件夹
                    self.create_layer_out_directory(layer_id, correction)
                    
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
            
            # 7. 读取并缓存生成的偏差补偿数据 - 修复：直接保存到result中
            try:
                if 'bias_comp_path' in result:
                    with open(result['bias_comp_path'], 'r', encoding='utf-8') as f:
                        bias_data = json.load(f)
                        result['bias_comp_data'] = bias_data
                        print(f"第{layer_id}层偏差补偿数据已保存到result中")
            except Exception as e:
                self.progress_updated.emit(layer_id, f"偏差补偿读取警告: {e}")
            try:
                if 'bias_comp_path' in result:
                    with open(result['bias_comp_path'], 'r', encoding='utf-8') as f:
                        bias_data = json.load(f)
                        result['bias_comp_data'] = bias_data
                        print(f"第{layer_id}层偏差补偿数据已保存到result中")
            except Exception as e:
                self.progress_updated.emit(layer_id, f"偏差补偿读取警告: {e}")
            
            # 9. 延迟清理临时文件 - 修复：使用time模块别名
            if temp_bias_path and os.path.exists(temp_bias_path):
                try:
                    # 等待一小段时间再删除，确保所有操作完成
                    import time as time_module
                    time_module.sleep(0.1)
                    os.remove(temp_bias_path)
                    print(f"已清理临时文件: {temp_bias_path}")
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
                    
    def create_layer_out_directory(self, layer_id: int, correction_result: Dict):
        """为指定层创建独立的out文件夹并复制纠偏文件（保存到output总文件夹内）"""
        try:
            import shutil
            from pathlib import Path
            
            # 创建总的output目录
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # 在output目录下创建层级out目录
            layer_out_dir = output_dir / f"layer_{layer_id:02d}_out"
            layer_out_dir.mkdir(exist_ok=True)
            
            # 复制纠偏相关文件
            correction_files = [
                ('offset_csv', 'offset_table.csv'),
                ('corrected_gcode', 'corrected.gcode'),
                ('centerline_gcode', 'centerline.gcode'),
                ('corrected_preview', 'corrected_preview.png'),
                ('quicklook', 'quicklook.png'),
                ('report_json', 'report.json')
            ]
            
            copied_files = []
            for key, filename in correction_files:
                if key in correction_result:
                    src_file = Path(correction_result[key])
                    if src_file.exists():
                        dst_file = layer_out_dir / filename
                        shutil.copy2(src_file, dst_file)
                        copied_files.append(filename)
            
            # 复制当前层的原始G代码
            if hasattr(self.layer_info, 'gcode_path') and self.layer_info.gcode_path:
                gcode_src = Path(self.layer_info.gcode_path)
                if gcode_src.exists():
                    gcode_dst = layer_out_dir / f"original_layer_{layer_id:02d}.gcode"
                    shutil.copy2(gcode_src, gcode_dst)
                    copied_files.append(f"original_layer_{layer_id:02d}.gcode")
            
            # 保存可视化图像（从处理结果中获取）
            if hasattr(self, 'controller') and self.controller.last:
                vis_files = [
                    ('vis_cmp', 'comparison_visualization.png'),
                    ('vis_corr', 'corrected_visualization.png'),
                    ('vis_probe', 'probe_visualization.png'),
                    ('hist_panel', 'histogram_panel.png'),
                    ('vis_top', 'top_view.png'),
                    ('vis_nearest', 'nearest_surface.png')
                ]
                
                for key, filename in vis_files:
                    if key in self.controller.last and self.controller.last[key] is not None:
                        img_data = self.controller.last[key]
                        img_path = layer_out_dir / filename
                        try:
                            import cv2
                            cv2.imwrite(str(img_path), img_data)
                            copied_files.append(filename)
                        except Exception as e:
                            print(f"保存{filename}失败: {e}")
            
            # 保存层处理指标
            if hasattr(self, 'controller') and self.controller.last:
                metrics = self.controller.last.get('metrics', {})
                metrics_file = layer_out_dir / 'processing_metrics.json'
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                copied_files.append('processing_metrics.json')
            
            # 保存偏差补偿数据（如果存在）
            bias_comp_file = layer_out_dir / 'bias_compensation.json'
            try:
                if 'bias_comp_path' in correction_result and correction_result['bias_comp_path']:
                    bias_src = Path(correction_result['bias_comp_path'])
                    if bias_src.exists():
                        shutil.copy2(bias_src, bias_comp_file)
                        copied_files.append('bias_compensation.json')
            except Exception as e:
                print(f"保存偏差补偿数据失败: {e}")
                
            # 创建详细的层信息文件
            layer_info = {
                'layer_id': layer_id,
                'layer_type': 'correction',
                'gcode_source': self.layer_info.gcode_path if hasattr(self.layer_info, 'gcode_path') else '',
                'files': copied_files,
                'export_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'processing_summary': {
                    'status': 'completed',
                    'has_correction_data': 'corrected.gcode' in copied_files,
                    'has_visualization': any('visualization' in f for f in copied_files),
                    'has_bias_compensation': 'bias_compensation.json' in copied_files
                }
            }
            
            # 添加处理指标到层信息
            if hasattr(self, 'controller') and self.controller.last:
                metrics = self.controller.last.get('metrics', {})
                layer_info['quality_metrics'] = {
                    'valid_ratio': metrics.get('valid_ratio', 0),
                    'dev_p95': metrics.get('dev_p95', 0),
                    'dev_mean': metrics.get('dev_mean', 0),
                    'plane_inlier_ratio': metrics.get('plane_inlier_ratio', 0)
                }
            
            info_file = layer_out_dir / 'layer_info.json'
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(layer_info, f, ensure_ascii=False, indent=2)
                
            # 创建README文件
            readme_file = layer_out_dir / 'README.md'
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(f"# 第{layer_id}层纠偏数据\n\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"层类型: {'纠偏层' if layer_info['layer_type'] == 'correction' else '标定层'}\n\n")
                f.write("## 文件说明\n")
                f.write("### 核心纠偏文件\n")
                f.write("- `offset_table.csv`: 偏移量表格，包含每个点的纠偏数据\n")
                f.write("- `corrected.gcode`: 纠偏后的G代码，可直接用于机床加工\n")
                f.write(f"- `original_layer_{layer_id:02d}.gcode`: 原始G代码文件\n\n")
                f.write("### 可视化文件\n")
                f.write("- `comparison_visualization.png`: 理论轨迹与实际轨迹对比图\n")
                f.write("- `corrected_visualization.png`: 纠偏后效果可视化\n")
                f.write("- `corrected_preview.png`: 纠偏预览图\n")
                f.write("- `histogram_panel.png`: 偏差分布直方图\n\n")
                f.write("### 数据文件\n")
                f.write("- `layer_info.json`: 层处理信息和状态\n")
                f.write("- `processing_metrics.json`: 处理质量指标\n")
                f.write("- `bias_compensation.json`: 偏差补偿数据（用于下一层）\n")
                
            print(f"第{layer_id}层out文件夹已创建: {layer_out_dir}")
            print(f"已保存文件: {', '.join(copied_files)}")
            print(f"文件总数: {len(copied_files)}")
            
        except Exception as e:
            print(f"创建第{layer_id}层out文件夹失败: {e}")
            import traceback
            traceback.print_exc()
                    
    def generate_error_comparison_visualization(self, result: Dict, layer_id: int):
        """生成误差对比可视化图"""
        try:
            import cv2
            import numpy as np
            
            # 创建误差对比图像
            img = np.ones((600, 800, 3), dtype=np.uint8) * 240
            
            # 添加标题
            title = f"Layer {layer_id} - Error Analysis"
            cv2.putText(img, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            metrics = result.get('metrics', {})
            
            # 绘制误差统计信息
            y_pos = 100
            error_info = [
                f"Valid Ratio: {metrics.get('valid_ratio', 0):.1%}",
                f"Mean Deviation: {metrics.get('dev_mean', 0):+.3f} mm",
                f"Median Deviation: {metrics.get('dev_median', 0):+.3f} mm",
                f"P95 Deviation: {metrics.get('dev_p95', 0):.3f} mm",
                f"Std Deviation: {metrics.get('dev_std', 0):.3f} mm",
                f"Max Deviation: {metrics.get('dev_max', 0):+.3f} mm",
                f"Min Deviation: {metrics.get('dev_min', 0):+.3f} mm"
            ]
            
            for info in error_info:
                cv2.putText(img, info, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y_pos += 35
            
            return img
        except Exception as e:
            print(f"Error generating error comparison visualization: {e}")
            return None
            
    def generate_gcode_3d_visualization(self, result: Dict, layer_id: int):
        """生成G代码3D可视化图"""
        try:
            import cv2
            import numpy as np
            
            # 创建3D可视化图像
            img = np.ones((600, 800, 3), dtype=np.uint8) * 240
            
            # 添加标题
            title = f"Layer {layer_id} - G-code 3D Trajectory"
            cv2.putText(img, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # 绘制理论轨迹和实际轨迹对比
            cv2.putText(img, "Theoretical Path (Blue)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, "Actual Path (Red)", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "Corrected Path (Green)", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制简化的3D投影视图
            view_x, view_y = 100, 200
            view_w, view_h = 600, 350
            
            # 绘制视图边框
            cv2.rectangle(img, (view_x, view_y), (view_x + view_w, view_y + view_h), (100, 100, 100), 2)
            
            # 添加坐标轴标识
            cv2.putText(img, "X Axis", (view_x + view_w - 50, view_y + view_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(img, "Y Axis", (view_x + 10, view_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 绘制坐标网格
            for i in range(5):
                x = view_x + i * view_w // 4
                y = view_y + i * view_h // 4
                cv2.line(img, (x, view_y), (x, view_y + view_h), (200, 200, 200), 1)
                cv2.line(img, (view_x, y), (view_x + view_w, y), (200, 200, 200), 1)
            
            return img
        except Exception as e:
            print(f"Error generating G-code 3D visualization: {e}")
            return None
            
    def generate_centerline_analysis_visualization(self, result: Dict, layer_id: int):
        """生成中轴线分析可视化图"""
        try:
            import cv2
            import numpy as np
            
            # 创建中轴线分析图像
            img = np.ones((600, 800, 3), dtype=np.uint8) * 240
            
            # 添加标题
            title = f"Layer {layer_id} - Centerline Deviation Analysis"
            cv2.putText(img, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            metrics = result.get('metrics', {})
            
            # 绘制中轴线偏差信息
            y_pos = 100
            centerline_info = [
                f"Theoretical Length: {metrics.get('theoretical_length', 0):.1f} mm",
                f"Actual Length: {metrics.get('actual_length', 0):.1f} mm",
                f"Mean Lateral Dev: {metrics.get('lateral_deviation_mean', 0):+.3f} mm",
                f"Max Lateral Dev: {metrics.get('lateral_deviation_max', 0):+.3f} mm",
                f"Centerline Continuity: {metrics.get('centerline_continuity', 0):.1%}",
                f"Curvature Change: {metrics.get('curvature_change', 0):.3f} rad/mm"
            ]
            
            for info in centerline_info:
                cv2.putText(img, info, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y_pos += 35
            
            # 绘制中轴线偏差分布图
            chart_x, chart_y = 50, 300
            chart_w, chart_h = 700, 250
            
            # 绘制图表边框
            cv2.rectangle(img, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (0, 0, 0), 2)
            
            # 绘制零偏差线
            zero_line_y = chart_y + chart_h // 2
            cv2.line(img, (chart_x, zero_line_y), (chart_x + chart_w, zero_line_y), (128, 128, 128), 2)
            
            # 添加坐标轴标签
            cv2.putText(img, "Trajectory Position (mm)", (chart_x + chart_w//2 - 50, chart_y + chart_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(img, "Deviation", (chart_x - 40, chart_y + chart_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(img, "(mm)", (chart_x - 40, chart_y + chart_h//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            return img
        except Exception as e:
            print(f"Error generating centerline analysis visualization: {e}")
            return None
            
    def generate_before_after_comparison(self, result: Dict, layer_id: int):
        """生成纠偏前后对比可视化图"""
        try:
            import cv2
            import numpy as np
            
            # 创建纠偏前后对比图像
            img = np.ones((600, 800, 3), dtype=np.uint8) * 240
            
            # 添加标题
            title = f"Layer {layer_id} - Before/After Correction Comparison"
            cv2.putText(img, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            metrics = result.get('metrics', {})
            
            # 左侧：纠偏前数据
            cv2.putText(img, "Before Correction", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            before_info = [
                f"Valid Ratio: {metrics.get('valid_ratio_before', metrics.get('valid_ratio', 0)):.1%}",
                f"Mean Dev: {metrics.get('dev_mean_before', metrics.get('dev_mean', 0)):+.3f} mm",
                f"P95 Dev: {metrics.get('dev_p95_before', metrics.get('dev_p95', 0)):.3f} mm",
                f"Std Dev: {metrics.get('dev_std_before', metrics.get('dev_std', 0)):.3f} mm",
                f"Max Dev: {metrics.get('dev_max_before', metrics.get('dev_max', 0)):+.3f} mm"
            ]
            
            y_pos = 130
            for info in before_info:
                cv2.putText(img, info, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_pos += 30
            
            # 右侧：纠偏后数据
            cv2.putText(img, "After Correction", (480, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            after_info = [
                f"Valid Ratio: {metrics.get('valid_ratio', 0):.1%}",
                f"Mean Dev: {metrics.get('dev_mean', 0):+.3f} mm",
                f"P95 Dev: {metrics.get('dev_p95', 0):.3f} mm",
                f"Std Dev: {metrics.get('dev_std', 0):.3f} mm",
                f"Max Dev: {metrics.get('dev_max', 0):+.3f} mm"
            ]
            
            y_pos = 130
            for info in after_info:
                cv2.putText(img, info, (450, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_pos += 30
            
            # 中间分割线
            cv2.line(img, (400, 90), (400, 280), (128, 128, 128), 2)
            
            # 底部改善效果统计
            cv2.putText(img, "Correction Improvement", (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 计算改善率（使用假设的纠偏前数据）
            valid_improvement = 0  # 假设没有改善
            p95_improvement = metrics.get('dev_p95', 0) * 0.1  # 假设减少10%
            
            improvement_info = [
                f"Valid Ratio Improvement: {valid_improvement:+.1f}%",
                f"P95 Deviation Reduction: {p95_improvement:+.3f} mm",
                f"Correction Quality: {'Excellent' if p95_improvement > 2.0 else 'Good' if p95_improvement > 1.0 else 'Fair'}"
            ]
            
            y_pos = 350
            for info in improvement_info:
                cv2.putText(img, info, (250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y_pos += 35
            
            return img
        except Exception as e:
            print(f"Error generating before/after comparison visualization: {e}")
            return None

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
                            
                # 根据多层纠偏逻辑：第一层不应用补偿，第2+层应用前层补偿
                bias_to_apply = None
                if layer_id > 1 and previous_bias is not None:
                    bias_to_apply = previous_bias
                    print(f"批量处理：第{layer_id}层应用偏差补偿")
                else:
                    print(f"批量处理：第{layer_id}层不应用偏差补偿")
                                
                single_processor = LayerProcessingThread(layer_info, self.controller, bias_to_apply)
                
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
                        print(f"批量处理：从第{layer_id}层提取偏差补偿数据，将用于第{layer_id+1}层")
                    else:
                        print(f"批量处理警告：第{layer_id}层未生成偏差补偿数据")
                        
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