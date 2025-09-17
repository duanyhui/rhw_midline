# 多层加工纠偏系统 - 安装指南

## 系统要求

- **操作系统**: Windows 10/11 (64位)
- **Python版本**: Python 3.7+ 
- **内存**: 建议8GB以上
- **硬盘空间**: 至少2GB可用空间

## 安装步骤

### 1. 安装Python依赖包

```bash
# 必需的基础包
pip install numpy opencv-python PyQt5

# 可选包（根据需要安装）
pip install python-snap7  # S7协议通信支持
```

### 2. 检查文件结构

确保以下文件存在于程序目录中：
```
gui/
├── multilayer_main.py              # 主程序
├── multilayer_data.py              # 数据结构
├── multilayer_plc.py               # PLC通信
├── multilayer_processor.py         # 处理线程
├── multilayer_visualizer.py        # 可视化组件
├── multilayer_config.py            # 配置管理
├── run_multilayer_system.py        # 启动脚本
├── test_multilayer_system.py       # 测试脚本
├── controller.py                   # 原有控制器
├── align_centerline_to_gcode_pro_edit_max.py  # 原有算法
├── T_cam2machine.npy              # 相机外参文件
├── README.md                      # 说明文档
└── INSTALL.md                     # 本安装指南
```

### 3. 相机配置

如果使用真实相机，需要：
1. 安装Percipio相机驱动和SDK
2. 确保`T_cam2machine.npy`外参文件存在
3. 验证相机连接正常

### 4. PLC配置（可选）

#### TCP通信
- 确保PLC支持TCP Socket通信
- 配置PLC的IP地址和端口
- 实现约定的JSON数据格式

#### S7通信  
- 安装python-snap7库：`pip install python-snap7`
- 下载并安装Snap7库（Windows DLL）
- 配置S7 PLC的IP和机架/槽位号

## 运行方式

### 1. 正常启动
```bash
python run_multilayer_system.py
```

### 2. 测试模式（无需硬件）
```bash
python test_multilayer_system.py
```

### 3. 直接启动主程序
```bash
python multilayer_main.py
```

## 配置说明

### 首次运行配置
1. 启动程序后，在"项目配置"中设置：
   - 项目名称
   - 总层数
   - 层厚度
   
2. 在"PLC通信"中配置（如果使用）：
   - 启用PLC通信
   - 选择通信类型（TCP/S7/Mock）
   - 设置IP地址和端口
   
3. 在"G代码管理"中：
   - 选择包含各层G代码的目录
   - 确认文件按层序命名

### 算法参数调整
在`controller.py`中的`GUIConfig`类可以调整：
- 引导步长 (guide_step_mm)
- 搜索半宽 (guide_halfwidth_mm) 
- 平滑窗口 (guide_smooth_win)
- 平面拟合参数等

## 故障排除

### 常见错误及解决方案

#### 1. ImportError: No module named 'PyQt5'
```bash
pip install PyQt5
```

#### 2. ImportError: No module named 'cv2'
```bash
pip install opencv-python
```

#### 3. 相机连接失败
- 检查相机USB连接
- 确认pcammls库安装正确
- 验证T_cam2machine.npy文件存在

#### 4. PLC通信失败
- 检查网络连接
- 确认IP地址和端口设置
- 测试PLC程序是否正常运行

#### 5. G代码加载失败
- 确认G代码文件格式正确
- 检查文件权限
- 验证文件路径中没有特殊字符

### 日志调试
程序运行时会在控制台输出调试信息，可以根据错误信息进行问题定位。

### 测试模式验证
如果怀疑环境配置问题，可以运行测试模式：
```bash
python test_multilayer_system.py
```
测试模式会创建模拟数据，验证程序各模块是否正常工作。

## 性能优化

### 1. 相机优化
- 调整相机曝光时间和增益
- 优化ROI区域设置
- 调整像素分辨率

### 2. 算法优化  
- 根据实际精度要求调整搜索范围
- 优化平滑参数减少噪声
- 调整平面拟合参数

### 3. 系统优化
- 确保足够的内存和CPU资源
- 关闭不必要的后台程序
- 使用SSD硬盘提高I/O性能

## 升级说明

### 备份重要数据
升级前请备份：
- 项目配置文件
- 标定数据文件
- 处理结果数据

### 版本兼容性
- 配置文件格式向下兼容
- 偏差补偿数据格式保持兼容
- G代码格式标准兼容

## 技术支持

如遇到安装或使用问题：
1. 查看控制台错误信息
2. 检查配置文件格式
3. 运行测试模式验证环境
4. 联系技术支持并提供：
   - 系统版本信息
   - 详细错误日志
   - 配置文件内容

---

**多层加工纠偏系统 v1.0**
安装指南 - 最后更新：2024年1月