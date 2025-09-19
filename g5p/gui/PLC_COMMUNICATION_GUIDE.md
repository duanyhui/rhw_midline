# 多层加工纠偏系统 - PLC通信模块使用说明

## 系统概述

多层加工纠偏系统现已支持与西门子PLC的S7协议通信，以及本地模拟测试功能。系统提供了两套完整的解决方案：

1. **模拟测试环境** - 用于本地开发和测试
2. **真实PLC通信** - 用于实际生产环境

## 快速开始

### 1. 启动交互式测试环境（推荐）

**方法一：使用交互式PLC（手动控制）**
```bash
# 终端1：启动交互式模拟PLC
python start_interactive_plc.py

# 终端2：启动主系统
python run_multilayer_system.py
```

**方法二：一键启动（自动使用交互式PLC）**
```bash
python test_full_system.py
```

### 2. 交互式PLC控制流程

启动后，您将看到PLC命令提示符：
```
[第1层-idle] > 
```

**典型的一层加工流程：**
1. `start` - 开始第1层加工
2. `complete` - 完成第1层加工（状态变为waiting）
3. 系统自动开始处理纠偏数据
4. 系统完成后，使用 `next` 进入第2层
5. 重复步骤1-4

**常用命令：**
- `start` - 开始当前层加工
- `complete` - 完成当前层加工
- `next` - 进入下一层
- `status` - 查看当前状态
- `stats` - 查看详细统计
- `help` - 显示所有命令
- `quit` - 退出PLC服务器

### 3. 传统自动模拟PLC

```bash
python mock_plc_server.py --host 127.0.0.1 --port 502 --layers 6 --time 8.0
```

## PLC通信协议

### 数据交换格式

使用JSON格式进行数据交换，支持以下命令类型：

1. **read_current_layer** - 读取当前层号
2. **read_machine_status** - 读取机床状态
3. **write_layer_complete** - 写入层完成信号
4. **send_correction_data** - 发送纠偏数据
5. **alert_deviation_error** - 偏差过大警告

### 机床状态定义

- `idle` - 空闲状态
- `processing` - 加工中
- `waiting` - 等待纠偏数据
- `error` - 错误状态

### 数据交换流程

1. **机床→系统**: 发送当前层号和状态
2. **系统→机床**: 确认接收并开始处理
3. **系统内部**: 相机采图、算法处理、生成纠偏数据
4. **系统→机床**: 发送纠偏数据（offset_table.csv + corrected.gcode）
5. **机床→系统**: 确认接收纠偏数据
6. **机床→系统**: 发送加工完成信号
7. **循环**: 自动进入下一层

## S7协议配置 (真实PLC)

### 地址配置示例

```python
# 项目配置中的PLC地址设置
plc_config = {
    "plc_type": "s7",
    "plc_ip": "192.168.1.100",
    "plc_port": 102,
    "current_layer_address": "DB1.DBD0",     # 当前层号
    "start_signal_address": "DB1.DBX4.0",   # 开始信号
    
    # 以下地址需要PLC工程师配置
    "status_address": "DB1.DBW10",          # 机床状态
    "completion_address": "DB1.DBX5.0",     # 完成标志
    "processing_time_address": "DB1.DBD20", # 处理时间
    "correction_valid_address": "DB2.DBX4.0", # 纠偏数据有效标志
    "alert_address": "DB3.DBX0.0",          # 警告标志
}
```

### PLC工程师集成指南

1. **安装python-snap7库**:
   ```bash
   pip install python-snap7
   ```

2. **修改地址配置**: 在 `multilayer_plc.py` 中的 `S7PLCCommunicator` 类中，根据实际PLC程序修改数据块地址。

3. **数据格式约定**:
   - 层号: 32位无符号整数 (DBD)
   - 状态: 16位无符号整数 (DBW)
   - 标志位: 布尔值 (DBX)
   - 时间: 32位无符号整数，毫秒 (DBD)
   - 偏差值: 32位浮点数 (DBD)

4. **扩展通信协议**: 在 `send_correction_data` 方法中，根据需要添加更多纠偏数据的传输逻辑。

## 安全机制

### 偏差检查

系统内置多重安全检查：

1. **最大偏移量检查**: 默认20mm
2. **梯度变化检查**: 防止突变
3. **数据有效性检查**: 确保文件完整性

### 警告处理

当检测到以下情况时，系统会：
- 发送警告信号到PLC
- 跳过当前层的纠偏
- 使用原始G代码继续加工
- 记录警告日志

```python
# 安全阈值设置（可在代码中调整）
MAX_OFFSET_MM = 20.0     # 最大允许偏移量
MAX_GRADIENT = 0.5       # 最大允许梯度
```

## 测试验证

### 交互式测试流程（推荐）

**步骤1：启动交互式模拟PLC**
```bash
python start_interactive_plc.py
```

**步骤2：启动主系统并配置**
1. 在另一个终端运行: `python run_multilayer_system.py`
2. 在主界面中：
   - 选择测试G代码目录（test_gcode）
   - 设置PLC连接 (TCP, 127.0.0.1:502)
   - 启用自动处理模式
3. 点击"连接PLC"

**步骤3：手动控制加工流程**

在PLC控制台中执行以下序列：

```bash
# 第1层（标定层）
[第1层-idle] > start          # 开始第1层加工
[第1层-processing] > complete # 完成第1层加工
# 系统自动开始处理（仅标定，不纠偏）
# 等待系统完成...
[第1层-idle] > next           # 进入第2层

# 第2层（首次纠偏层）
[第2层-idle] > start          # 开始第2层加工
[第2层-processing] > complete # 完成第2层加工
# 系统自动开始处理（生成纠偏数据）
# 等待系统完成...
[第2层-idle] > next           # 进入第3层

# 继续后续层...
```

**关键观察点：**
1. ✅ PLC连接状态正常显示
2. ✅ 层号自动同步
3. ✅ 状态转换 (idle → processing → waiting → idle)
4. ✅ 第1层仅标定，第2+层开始纠偏
5. ✅ 每层独立的out文件夹生成
6. ✅ 偏差补偿数据传递

### 自动化测试流程

使用传统的自动模拟PLC：

1. 启动完整测试环境
2. 在主界面中：
   - 选择测试G代码目录
   - 设置PLC连接 (TCP, 127.0.0.1:502)
   - 启用自动处理模式
3. 观察自动化流程：
   - 模拟PLC开始第1层加工
   - 系统自动处理并发送纠偏数据
   - 模拟PLC进入下一层
   - 重复直到所有层完成

### 验证要点

- [x] PLC连接状态正常
- [x] 层号自动切换
- [x] 状态同步 (processing → waiting → idle)
- [x] 纠偏数据传输
- [x] 安全检查机制
- [x] 错误处理和恢复
- [x] 完整的多层流程

## 故障排除

### 常见问题

1. **PLC连接失败**
   - 检查IP地址和端口
   - 确认防火墙设置
   - 验证PLC通信参数

2. **S7协议错误**
   - 安装python-snap7: `pip install python-snap7`
   - 检查S7地址格式
   - 确认PLC数据块配置

3. **数据传输失败**
   - 检查文件路径
   - 验证CSV格式
   - 查看错误日志

4. **偏差过大警告**
   - 检查标定精度
   - 调整安全阈值
   - 验证相机标定

### 调试工具

- 模拟PLC服务器控制台输出
- 主系统状态显示
- 详细的错误日志
- 可视化界面监控

## 下一步开发

1. **增强S7通信**: 支持更复杂的数据结构
2. **数据压缩**: 大型纠偏表的高效传输
3. **实时监控**: PLC状态的图形化显示
4. **历史记录**: 完整的操作日志系统
5. **远程诊断**: 网络化的故障诊断

## 技术支持

如需技术支持或定制开发，请联系开发团队。