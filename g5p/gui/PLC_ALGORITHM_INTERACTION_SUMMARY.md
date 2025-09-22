# PLC与算法程序数据交互总结

## PLC工程师对接指南

### 🎯 整体数据交换逻辑概述

**整体数据交换逻辑是**：程序连接上PLC之后，程序加载每一层的G代码，由机床发送信号告诉程序当前执行到第几层，当程序完成当前层作业后，发送作业完成信号，程序收到信号后经过指定长度时间的延时后，采集图像数据进行算法流程产出纠偏数据，第一层只标定偏差，后续层使用经偏差纠正后产生的纠偏数据（每一层out文件夹内的offset_table.csv和corrected.gcode），然后将纠偏数据发送给机床。在这个过程中，如果纠偏数据内出现了偏差过大的值，要设置警告并丢弃偏差数据转而使用没有偏差的纠偏数据，防止造成危险。

### 🔄 分层数据处理逻辑详解

#### 核心工作流程时序
```
系统初始化 → 连接建立 → G代码加载 → 层号同步 → 作业监控 → 完成信号 → 延时采集 → 算法处理 → 安全检查 → 纠偏数据传输
```

#### 第一层处理逻辑（标定基准层）

**处理目标**：建立加工基准坐标系，记录系统性偏差
- **输入数据**：原始G代码文件 + 实时图像采集数据
- **处理方式**：
  ```
  机床完成信号 → 延时采集 → 图像处理 → 特征检测 → 偏差计算 → 基准建立
  ```
- **输出内容**：
  - `layer_01_out/` 标定数据文件夹
  - 偏差基准参数（不直接用于纠偏）
  - 系统性偏差记录数据
- **数据传输**：**不发送纠偏数据给PLC**（仅标定，不纠偏）
- **PLC状态**：接收标定完成信号，准备进入第二层
- **安全特性**：仅标定不纠偏，确保首层加工安全

#### 后续层处理逻辑（纠偏应用层）

**处理目标**：基于前层累积偏差数据进行实时纠偏补偿
- **输入数据**：
  - 前一层的偏差补偿数据
  - 当前层原始G代码文件
  - 实时图像采集数据
- **处理方式**：
  ```
  机床完成信号 → 延时采集 → 图像处理 → 偏差检测 → 坐标变换 → G代码重生成 → 偏移表更新
  ```
- **核心输出文件**：
  - `layer_XX_out/corrected.gcode` - **纠偏后的G代码文件（核心）**
  - `layer_XX_out/offset_table.csv` - **偏移补偿表（核心）**
- **辅助输出文件**：
  - `processing_metrics.json` - 处理质量指标
  - `bias_compensation.json` - 偏差补偿数据
  - `layer_info.json` - 层信息元数据
  - `visualization/` - 可视化图像集
- **数据传输**：**发送完整纠偏数据包给PLC**
- **PLC应用**：使用纠偏数据替换原始加工路径
- **累积补偿**：每层纠偏数据会考虑前层的累积偏差

#### 每层数据流向详解

**阶段1：层序同步**
```
PLC → 算法程序: 当前层号 (层N)
算法程序: 加载layer_N的原始G代码
```

**阶段2：作业完成检测**
```
机床: 完成第N层加工
PLC → 算法程序: 作业完成信号
算法程序: 状态变化检测 (processing → waiting)
```

**阶段3：延时与采集**
```
算法程序: 延时等待 (可配置0.5-2.0秒)
算法程序: 启动图像采集
算法程序: 执行偏差检测算法
```

**阶段4：纠偏数据生成**
```
第1层: 仅生成标定数据 → layer_01_out/基准数据
第N层: 应用(N-1)层补偿 → layer_XX_out/corrected.gcode + offset_table.csv
```

**阶段5：安全检查与传输**
```
安全检查: 偏移量 ≤ 20mm ?
  ↓ 安全                    ↓ 危险
传输纠偏数据              发送警告信号
  ↓                        ↓
PLC接收并应用             PLC丢弃，使用原始路径
```

### 🛡️ 安全控制机制详解

#### 偏差安全阈值系统
- **硬安全阈值**：单轴最大偏移量 ≤ 20mm
- **梯度安全检查**：相邻点偏移变化 < 0.5mm
- **数据完整性验证**：文件存在性和格式正确性
- **检查触发时机**：每次生成纠偏数据后立即执行

#### 危险偏差处理流程
```
检测到危险偏差 → 系统警告 → 数据丢弃 → 安全数据生成 → PLC通知 → 原始路径使用
```

**具体执行步骤**：
1. **即时警告**：向PLC发送警告信号，GUI显示安全警报
2. **数据处理**：立即丢弃危险偏差数据，生成零偏移量的安全替代数据
3. **流程控制**：暂停自动处理流程，等待操作员确认或干预
4. **日志记录**：记录详细异常信息，包括偏差值和触发时间
5. **恢复机制**：提供手动处理选项或自动使用原始G代码

#### 安全响应数据格式
```json
{
  "type": "send_correction_data",
  "layer": N,
  "correction_status": "warning",
  "data": {
    "alert_message": "偏差超过安全阈值20mm",
    "deviation_value": 25.6,
    "max_offset_detected": [12.3, 15.8, 8.9],
    "safety_action": "auto_discard",
    "fallback_action": "use_original_gcode"
  }
}
```
---

## 📡 技术规范与PLC对接要求

#### 📡 通信协议详解

**1. 物理连接参数**
```
协议类型: TCP/IP
端口号: 502 (标准PLC通信端口)
IP地址: 可配置 (默认127.0.0.1本地测试)
超时设置: 5秒连接超时
数据格式: JSON结构化数据
编码方式: UTF-8
```

**2. 数据交换命令集**
```python
# PLC状态查询命令
CMD_READ_LAYER = "read_current_layer"      # 读取当前层号
CMD_READ_STATUS = "read_machine_status"    # 读取机床状态

# 数据传输命令  
CMD_SEND_CORRECTION = "send_correction_data" # 发送纠偏数据
CMD_WRITE_COMPLETE = "write_layer_complete"  # 写入层完成信号

# 安全控制命令
CMD_ALERT_ERROR = "alert_deviation_error"   # 偏差过大警告
```

**3. 机床状态枚举定义**
```python
STATUS_IDLE = "idle"           # 空闲状态 - 等待开始下一层
STATUS_PROCESSING = "processing" # 加工中 - 机床正在执行加工
STATUS_WAITING = "waiting"     # 等待纠偏数据 - 加工完成，等待算法处理
STATUS_ERROR = "error"         # 错误状态 - 系统异常或安全停机
```

#### 📄 数据交换格式规范

**1. 状态查询请求格式**
```json
// 算法程序 -> PLC: 查询当前层号
{
  "type": "read_current_layer",
  "timestamp": 1635728400.123
}

// PLC -> 算法程序: 返回层号
{
  "success": true,
  "layer": 3,
  "timestamp": 1635728400.124
}
```

```json
// 算法程序 -> PLC: 查询机床状态
{
  "type": "read_machine_status",
  "timestamp": 1635728400.125
}

// PLC -> 算法程序: 返回状态
{
  "success": true,
  "status": "waiting",
  "current_layer": 3,
  "timestamp": 1635728400.126
}
```

**2. 纠偏数据发送格式**
```json
// 算法程序 -> PLC: 发送纠偏数据包
{
  "type": "send_correction_data",
  "layer": 2,
  "correction_status": "valid",  // valid/warning/skip
  "timestamp": 1635728400.200,
  "data": {
    // 核心文件路径
    "corrected_gcode_path": "output/layer_02_out/corrected.gcode",
    "offset_table_path": "output/layer_02_out/offset_table.csv",
    "output_directory": "output/layer_02_out",
    "available_files": ["corrected.gcode", "offset_table.csv", "processing_metrics.json"],
    
    // G代码调整详情
    "gcode_adjustments": [
      {
        "line_number": 10,
        "original_line": "G1 X10.000 Y20.000 Z5.000",
        "corrected_line": "G1 X10.123 Y19.876 Z5.000",
        "offset_mm": [0.123, -0.124, 0.000]
      }
    ],
    
    // 纠偏统计信息
    "correction_summary": {
      "avg_correction_mm": 0.087,
      "max_correction_mm": 0.175,
      "affected_gcode_lines": 156,
      "total_corrections": 128
    },
    
    // 处理质量信息
    "processing_info": {
      "total_correction_points": 1024,
      "valid_ratio": 0.892,
      "processing_time": 2.34
    }
  }
}

// PLC -> 算法程序: 确认接收
{
  "success": true,
  "message": "纠偏数据已接收",
  "timestamp": 1635728400.201
}
```

**3. 安全警告数据格式**
```json
// 算法程序 -> PLC: 安全警告 (偏差过大)
{
  "type": "send_correction_data",
  "layer": 4,
  "correction_status": "warning",
  "timestamp": 1635728400.300,
  "data": {
    "alert_message": "偏差超过安全阈值20mm",
    "deviation_value": 25.6,
    "max_offset_detected": [12.3, 15.8, 8.9],
    "safety_action": "auto_discard"
  }
}

// PLC -> 算法程序: 安全确认
{
  "success": true,
  "message": "安全警告已接收，使用原始路径",
  "timestamp": 1635728400.301
}
```

### 🔄 完整交互流程时序

#### 阶段1: 系统初始化和连接
```
1. PLC系统启动，绑定端口502
2. 算法程序启动，加载项目配置和G代码文件
3. 算法程序发起TCP连接请求
4. PLC确认连接，建立通信链路
5. 算法程序开始1秒间隔的状态轮询
```

#### 阶段2: 单层处理循环 (每层重复)

**步骤1: 机床开始加工**
```
PLC操作: 机床开始第N层加工
PLC状态: idle -> processing
算法程序: 轮询检测到processing状态
```

**步骤2: 机床完成加工**
```
PLC操作: 机床完成第N层加工
PLC状态: processing -> waiting
PLC信号: 发送作业完成信号
```

**步骤3: 算法程序检测并响应**
```
算法程序: 轮询发现waiting状态
算法程序: 发出correction_request内部信号
算法程序: 延时0.5秒后开始处理
```

**步骤4: 图像采集和算法处理**
```
算法程序: 相机采集当前层表面图像
算法程序: 计算偏差并生成纠偏数据
算法程序: 生成output/layer_XX_out/文件集
```

**步骤5: 安全检查与决策**
```
算法程序: 检查偏移量是否超过20mm阈值

如果安全:
  算法程序: 发送完整纠偏数据包到PLC
  PLC: 接收并显示纠偏数据详情
  PLC: 准备应用纠偏数据

如果危险:
  算法程序: 发送安全警告到PLC
  PLC: 显示警告信息
  PLC: 自动丢弃危险数据，使用原始G代码
```

**步骤6: 准备下一层**
```
PLC操作: 手动进入下一层 (或算法程序自动进入)
PLC状态: waiting -> idle
层号: N -> N+1
循环: 重复步骤1-6直到所有层完成
```

### 🔧 PLC端实现要求

#### 必须实现的接口
```
1. TCP服务器功能 (端口502)
2. JSON数据解析和生成
3. 状态管理 (idle/processing/waiting/error)
4. 层号记录和更新
5. 纠偏数据接收和存储
6. 安全警告处理
```

#### 推荐的PLC端数据结构
```
// PLC内部状态变量
INT current_layer := 1;              // 当前层号
STRING machine_status := 'idle';     // 机床状态
BOOL correction_received := FALSE;   // 纠偏数据接收标志
STRING correction_files[10];         // 纠偏文件路径列表
REAL max_deviation_mm := 20.0;       // 安全阈值
```

### 🔄 详细交互流程

#### 阶段1: 系统初始化
1. **启动模拟PLC服务器**
   ```bash
   python start_interactive_plc.py
   ```
   - 绑定TCP端口502
   - 初始化机床状态为idle
   - 启动命令行交互界面

2. **启动算法程序**
   ```bash
   python multilayer_main.py
   ```
   - 加载项目配置和G代码文件
   - 初始化相机和控制器
   - 创建PLC通信器

3. **建立连接**
   - 算法程序连接PLC (TCP socket)
   - 启动状态轮询定时器(1秒间隔)
   - 连接成功后开始监听PLC状态变化

#### 阶段2: 单层处理循环

##### 2.1 PLC控制作业开始
**操作**: 在PLC命令行输入 `start N`
```
PLC状态变化: idle → processing
机床开始第N层加工...
```

##### 2.2 PLC控制作业完成  
**操作**: 在PLC命令行输入 `complete`
```
PLC状态变化: processing → waiting
第N层加工完成，等待纠偏数据...
```

##### 2.3 算法程序检测状态变化
**触发**: PLC状态轮询检测到waiting状态
```python
# 状态轮询逻辑
if current_status == PLCDataProtocol.STATUS_WAITING:
    current_layer = self.read_current_layer()
    self.correction_request.emit(current_layer)
```

##### 2.4 算法程序开始处理
**响应**: `on_correction_request(layer_id)` 方法被调用
```python
def on_correction_request(self, layer_id: int):
    print(f"PLC请求第{layer_id}层纠偏数据，开始处理...")
    self.current_layer = layer_id
    # 延迟后开始处理（模拟采集延迟）
    delay_ms = getattr(self, 'process_delay_sec', 0.5) * 1000
    QTimer.singleShot(int(delay_ms), self.process_current_layer)
```

##### 2.5 纠偏数据计算与文件生成
**处理内容**:
- 相机图像采集
- 偏差检测和计算
- G代码纠偏处理
- 生成输出文件到 `output/layer_XX_out/`：
  - `corrected.gcode` - 纠偏后的G代码
  - `offset_table.csv` - 偏移量表
  - `processing_metrics.json` - 处理指标
  - `bias_compensation.json` - 偏差补偿数据

##### 2.6 发送纠偏数据到PLC
**数据结构**:
```json
{
  "type": "send_correction_data",
  "layer": 2,
  "correction_status": "valid",
  "data": {
    "corrected_gcode_path": "output/layer_02_out/corrected.gcode",
    "offset_table_path": "output/layer_02_out/offset_table.csv",
    "processing_metrics_path": "output/layer_02_out/processing_metrics.json",
    "bias_compensation_path": "output/layer_02_out/bias_compensation.json",
    "output_directory": "output/layer_02_out",
    "available_files": ["corrected.gcode", "offset_table.csv", "..."],
    "gcode_adjustments": [
      {
        "line_number": 10,
        "original_line": "G1 X10.000 Y20.000 Z5.000",
        "corrected_line": "G1 X10.123 Y19.876 Z5.000", 
        "offset_mm": [0.123, -0.124, 0.000]
      }
    ],
    "correction_summary": {
      "avg_correction_mm": 0.087,
      "max_correction_mm": 0.175,
      "affected_gcode_lines": 156
    },
    "processing_info": {
      "total_correction_points": 1024,
      "valid_ratio": 0.892,
      "processing_time": 2.34
    }
  }
}
```

##### 2.7 PLC接收和显示纠偏数据
**PLC响应**:
```
[系统] 接收到纠偏数据: 第2层, 状态: valid
━━━ 第2层纠偏数据详情 ━━━
• 输出目录: output/layer_02_out
• 纠偏后G代码: output/layer_02_out/corrected.gcode (12345 字节)
• 偏移表文件: output/layer_02_out/offset_table.csv (678 字节)
• 相关文件: corrected.gcode, offset_table.csv, processing_metrics.json
• 数据质量: 有效点数 1024, 有效率 89.2%
• 纠偏统计: 平均 0.087mm, 最大 0.175mm
• 影响范围: 156 行G代码被修正
• G代码调整: 128 个调整点
• 机床状态: 纠偏数据已接收，等待应用
• 文件状态: corrected.gcode 和 offset_table.csv 已接收
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

##### 2.8 进入下一层
**操作**: 在PLC命令行输入 `next`
```
PLC状态变化: waiting → idle
当前层: 2→3
准备下一层加工...
```

#### 阶段3: 安全控制机制

##### 3.1 偏差安全检查
**检查项目**:
- 最大偏移量：20mm阈值
- 梯度变化：0.5mm/点阈值  
- 数据完整性验证

##### 3.2 危险数据处理
**触发条件**: 偏移量 > 20mm
```json
{
  "type": "send_correction_data",
  "layer": 4,
  "correction_status": "warning",
  "data": {
    "alert_message": "偏差超过安全阈值20mm",
    "deviation_value": 25.6,
    "max_offset_detected": [12.3, 15.8, 8.9]
  }
}
```

**PLC响应**:
```
[警告] 第4层纠偏数据超出安全范围，已自动丢弃
[安全] 机床使用原始加工路径，不应用纠偏
```

### 🎛️ 控制接口总结

#### PLC命令行接口
```bash
start <层号>    # 开始指定层加工
complete        # 完成当前层加工
next           # 进入下一层
status         # 显示当前状态  
correction     # 显示纠偏数据应用情况
simulate <层号> # 模拟应用纠偏数据
deviation      # 显示偏差统计信息
path           # 显示加工路径信息
reset          # 重置到第1层
quit           # 退出服务器
```

#### 算法程序配置
- **处理延迟**: 可在高级参数中配置(默认0.5秒)
- **自动下一层**: 可选择手动或自动模式
- **PLC通信类型**: TCP/S7/Mock三种模式
- **安全阈值**: 可配置最大偏移量和梯度限制

### 📊 数据流向分析

#### 上行数据 (算法程序 → PLC)
1. **纠偏数据包**: 完整的纠偏信息和文件路径
2. **安全警告**: 偏差过大时的警告信息
3. **状态查询**: 定期查询PLC状态和层号

#### 下行数据 (PLC → 算法程序)  
1. **状态信息**: 机床当前状态(idle/processing/waiting/error)
2. **层号信息**: 当前加工层号
3. **完成信号**: 层加工完成的确认信号

#### 文件系统交互
1. **生成路径**: `output/layer_XX_out/`
2. **核心文件**: 
   - `corrected.gcode` (纠偏后G代码)
   - `offset_table.csv` (偏移量表)
3. **辅助文件**:
   - `processing_metrics.json` (处理指标)
   - `bias_compensation.json` (偏差补偿)
   - 可视化图像文件

### 🔧 技术实现要点

#### 1. 异步通信机制
- QTimer定时器实现状态轮询
- 信号槽机制实现事件驱动
- TCP非阻塞socket通信

#### 2. 数据安全保障
- JSON数据格式校验
- 文件存在性检查
- 偏移量安全阈值控制

#### 3. 错误处理机制
- 连接异常自动重连
- 数据传输失败重试
- 危险数据自动丢弃

#### 4. 多层偏差补偿逻辑
- 第1层：仅标定，不应用纠偏
- 第2+层：应用前一层偏差补偿
- 累积误差防止机制