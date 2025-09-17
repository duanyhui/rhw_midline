# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - 配置管理模块
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import asdict

from multilayer_data import ProjectConfig, LayerInfo

# ==================== 配置管理器 ====================

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.default_config_file = self.config_dir / "default.json"
        self.recent_projects_file = self.config_dir / "recent_projects.json"
        
    def save_project_config(self, project_config: ProjectConfig, file_path: str):
        """保存项目配置"""
        try:
            config_data = project_config.to_dict()
            config_data['save_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            config_data['version'] = "1.0"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
                
            # 更新最近项目列表
            self._update_recent_projects(file_path, project_config.project_name)
            
            return True
        except Exception as e:
            print(f"保存项目配置失败: {e}")
            return False
            
    def load_project_config(self, file_path: str) -> Optional[ProjectConfig]:
        """加载项目配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # 版本兼容性检查
            version = config_data.get('version', '1.0')
            if version != '1.0':
                print(f"配置文件版本不匹配: {version}")
                
            return ProjectConfig.from_dict(config_data)
        except Exception as e:
            print(f"加载项目配置失败: {e}")
            return None
            
    def save_default_config(self, project_config: ProjectConfig):
        """保存默认配置"""
        try:
            config_data = project_config.to_dict()
            config_data['is_default'] = True
            config_data['save_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.default_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            print(f"保存默认配置失败: {e}")
            return False
            
    def load_default_config(self) -> ProjectConfig:
        """加载默认配置"""
        if self.default_config_file.exists():
            try:
                with open(self.default_config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return ProjectConfig.from_dict(config_data)
            except Exception as e:
                print(f"加载默认配置失败: {e}")
                
        # 返回默认配置
        return ProjectConfig()
        
    def get_recent_projects(self) -> list:
        """获取最近项目列表"""
        try:
            if self.recent_projects_file.exists():
                with open(self.recent_projects_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"读取最近项目失败: {e}")
            
        return []
        
    def _update_recent_projects(self, file_path: str, project_name: str):
        """更新最近项目列表"""
        try:
            recent_projects = self.get_recent_projects()
            
            # 移除已存在的项目
            recent_projects = [p for p in recent_projects if p['path'] != file_path]
            
            # 添加到列表开头
            recent_projects.insert(0, {
                'path': file_path,
                'name': project_name,
                'last_opened': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 保持最多10个最近项目
            recent_projects = recent_projects[:10]
            
            with open(self.recent_projects_file, 'w', encoding='utf-8') as f:
                json.dump(recent_projects, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"更新最近项目失败: {e}")

# ==================== 项目管理器 ====================

class ProjectManager:
    """项目管理器"""
    
    def __init__(self, projects_dir: str = "projects"):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(exist_ok=True)
        self.config_manager = ConfigManager()
        
    def create_project(self, project_config: ProjectConfig) -> str:
        """创建新项目"""
        # 创建项目目录
        project_name = project_config.project_name
        safe_name = self._sanitize_filename(project_name)
        
        project_dir = self.projects_dir / safe_name
        project_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (project_dir / "layers").mkdir(exist_ok=True)
        (project_dir / "results").mkdir(exist_ok=True)
        (project_dir / "exports").mkdir(exist_ok=True)
        (project_dir / "images").mkdir(exist_ok=True)
        
        # 保存项目配置
        config_file = project_dir / "project.json"
        self.config_manager.save_project_config(project_config, str(config_file))
        
        return str(project_dir)
        
    def save_project_data(self, project_dir: str, layers: Dict[int, LayerInfo], 
                         current_layer: int = 0) -> bool:
        """保存项目数据"""
        try:
            project_path = Path(project_dir)
            
            # 保存层信息
            layers_data = {
                str(layer_id): layer_info.to_dict()
                for layer_id, layer_info in layers.items()
            }
            
            project_data = {
                'layers': layers_data,
                'current_layer': current_layer,
                'save_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'layer_count': len(layers)
            }
            
            data_file = project_path / "project_data.json"
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            print(f"保存项目数据失败: {e}")
            return False
            
    def load_project_data(self, project_dir: str) -> tuple:
        """加载项目数据"""
        try:
            project_path = Path(project_dir)
            
            # 加载项目配置
            config_file = project_path / "project.json"
            project_config = self.config_manager.load_project_config(str(config_file))
            
            # 加载项目数据
            data_file = project_path / "project_data.json"
            layers = {}
            current_layer = 0
            
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)
                    
                current_layer = project_data.get('current_layer', 0)
                layers_data = project_data.get('layers', {})
                
                for layer_id_str, layer_data in layers_data.items():
                    layer_id = int(layer_id_str)
                    layer_info = LayerInfo.from_dict(layer_data)
                    layers[layer_id] = layer_info
                    
            return project_config, layers, current_layer
            
        except Exception as e:
            print(f"加载项目数据失败: {e}")
            return None, {}, 0
            
    def export_project_results(self, project_dir: str, layers: Dict[int, LayerInfo]) -> str:
        """导出项目结果"""
        try:
            project_path = Path(project_dir)
            export_dir = project_path / "exports" / f"export_{int(time.time())}"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # 导出摘要
            summary = self._generate_project_summary(layers)
            summary_file = export_dir / "project_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            # 导出各层结果
            results_dir = export_dir / "layer_results"
            results_dir.mkdir(exist_ok=True)
            
            for layer_id, layer_info in layers.items():
                if layer_info.processing_result:
                    layer_dir = results_dir / f"layer_{layer_id:02d}"
                    layer_dir.mkdir(exist_ok=True)
                    
                    # 导出层信息
                    layer_data = layer_info.to_dict()
                    layer_file = layer_dir / "layer_info.json"
                    with open(layer_file, 'w', encoding='utf-8') as f:
                        json.dump(layer_data, f, ensure_ascii=False, indent=2)
                        
            return str(export_dir)
            
        except Exception as e:
            print(f"导出项目结果失败: {e}")
            return ""
            
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        import re
        # 移除非法字符
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 限制长度
        if len(filename) > 100:
            filename = filename[:100]
        return filename
        
    def _generate_project_summary(self, layers: Dict[int, LayerInfo]) -> Dict:
        """生成项目摘要"""
        summary = {
            'total_layers': len(layers),
            'completed_layers': 0,
            'failed_layers': 0,
            'average_processing_time': 0.0,
            'average_valid_ratio': 0.0,
            'average_dev_p95': 0.0,
            'generation_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        processing_times = []
        valid_ratios = []
        dev_p95s = []
        
        for layer_info in layers.values():
            if layer_info.status == 'completed':
                summary['completed_layers'] += 1
                
                if layer_info.processing_result:
                    # 提取指标
                    result = layer_info.processing_result
                    if 'processing_time' in result:
                        processing_times.append(result['processing_time'])
                        
                    metrics = result.get('metrics', {})
                    if 'valid_ratio' in metrics:
                        valid_ratios.append(metrics['valid_ratio'])
                    if 'dev_p95' in metrics:
                        dev_p95s.append(metrics['dev_p95'])
                        
            elif layer_info.status == 'error':
                summary['failed_layers'] += 1
                
        # 计算平均值
        if processing_times:
            summary['average_processing_time'] = sum(processing_times) / len(processing_times)
        if valid_ratios:
            summary['average_valid_ratio'] = sum(valid_ratios) / len(valid_ratios)
        if dev_p95s:
            summary['average_dev_p95'] = sum(dev_p95s) / len(dev_p95s)
            
        return summary

# ==================== 备份管理器 ====================

class BackupManager:
    """备份管理器"""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_backup(self, project_dir: str, backup_name: Optional[str] = None) -> str:
        """创建项目备份"""
        try:
            import shutil
            
            project_path = Path(project_dir)
            if not project_path.exists():
                return ""
                
            # 生成备份名称
            if not backup_name:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_name = f"{project_path.name}_{timestamp}"
                
            backup_path = self.backup_dir / backup_name
            
            # 复制项目目录
            shutil.copytree(project_path, backup_path)
            
            # 添加备份信息
            backup_info = {
                'original_path': str(project_path),
                'backup_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'backup_name': backup_name
            }
            
            info_file = backup_path / "backup_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, ensure_ascii=False, indent=2)
                
            return str(backup_path)
            
        except Exception as e:
            print(f"创建备份失败: {e}")
            return ""
            
    def restore_backup(self, backup_path: str, restore_dir: str) -> bool:
        """恢复备份"""
        try:
            import shutil
            
            backup_p = Path(backup_path)
            restore_p = Path(restore_dir)
            
            if not backup_p.exists():
                return False
                
            # 清空目标目录
            if restore_p.exists():
                shutil.rmtree(restore_p)
                
            # 复制备份内容
            shutil.copytree(backup_p, restore_p)
            
            # 移除备份信息文件
            backup_info_file = restore_p / "backup_info.json"
            if backup_info_file.exists():
                backup_info_file.unlink()
                
            return True
            
        except Exception as e:
            print(f"恢复备份失败: {e}")
            return False
            
    def list_backups(self) -> list:
        """列出所有备份"""
        backups = []
        
        try:
            for backup_path in self.backup_dir.iterdir():
                if backup_path.is_dir():
                    info_file = backup_path / "backup_info.json"
                    
                    backup_info = {
                        'path': str(backup_path),
                        'name': backup_path.name,
                        'backup_time': 'Unknown'
                    }
                    
                    if info_file.exists():
                        try:
                            with open(info_file, 'r', encoding='utf-8') as f:
                                file_info = json.load(f)
                                backup_info.update(file_info)
                        except:
                            pass
                            
                    backups.append(backup_info)
                    
            # 按时间排序
            backups.sort(key=lambda x: x.get('backup_time', ''), reverse=True)
            
        except Exception as e:
            print(f"列出备份失败: {e}")
            
        return backups
        
    def cleanup_old_backups(self, max_backups: int = 10):
        """清理旧备份"""
        try:
            backups = self.list_backups()
            
            if len(backups) > max_backups:
                import shutil
                
                for backup in backups[max_backups:]:
                    backup_path = Path(backup['path'])
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                        print(f"已删除旧备份: {backup['name']}")
                        
        except Exception as e:
            print(f"清理备份失败: {e}")