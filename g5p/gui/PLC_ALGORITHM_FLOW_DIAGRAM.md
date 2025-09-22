## 模拟PLC与算法程序数据交互流程图

### 📋 整体数据交换逻辑概述

整个多层加工纠偏系统的数据交换逻辑遵循以下核心流程：

#### 🔗 系统连接与初始化
- **程序连接PLC**：建立TCP通信连接，开始状态轮询监听
- **G代码加载**：程序预先加载每一层的G代码文件到系统中
- **层号同步**：机床通过PLC发送信号告知程序当前执行到第几层

#### ⚙️ 单层处理核心逻辑

**1. 作业完成信号接收**
```
机床完成当前层作业 → 发送作业完成信号 → 程序接收信号
```

**2. 延时与数据采集**
```
程序收到完成信号 → 延时等待（可配置时长）→ 触发图像采集
```

**3. 算法处理与纠偏数据生成**
```
图像数据采集 → 算法分析处理 → 生成纠偏数据文件
```

**4. 分层处理策略**
- **第一层（标定层）**：
  - 仅进行偏差标定，记录基准偏差值
  - 不应用纠偏，建立后续层的补偿基准
  - 生成`layer_01_out/`目录和基准数据

- **后续层（纠偏层）**：
  - 应用前一层的偏差补偿数据进行预纠偏
  - 使用经偏差纠正后的纠偏数据：
    - `offset_table.csv` - 偏移量补偿表
    - `corrected.gcode` - 纠偏后的G代码
  - 每层生成独立的`layer_XX_out/`输出目录

**5. 纠偏数据传输**
```
纠偏数据生成完成 → 安全检查 → 数据发送到机床PLC
```

#### 🛡️ 安全控制机制

**偏差安全检查流程**：
```
纠偏数据生成 → 偏差值检查 → 判断是否超出安全阈值(20mm)
    ↓                    ↓
安全范围内          偏差过大
    ↓                    ↓
正常发送数据      触发安全警告
    ↓                    ↓
机床接收纠偏      丢弃危险数据
                        ↓
                   使用原始路径
```

**安全保护措施**：
- **阈值控制**：设置20mm最大偏移量限制
- **自动丢弃**：超出阈值的纠偏数据自动丢弃
- **警告机制**：发出安全警告信息
- **降级保护**：转为使用无纠偏的原始数据，防止机床危险操作

#### 🔄 循环处理机制

```
第1层：标定基准 → 第2层：应用纠偏 → 第3层：累积纠偏 → ... → 第N层：完成
   ↓               ↓               ↓                    ↓
建立基准        前层补偿        逐层优化           全部完成
```

**数据传承链**：
- 每层的纠偏结果作为下一层的输入基准
- 累积补偿机制确保逐层精度提升
- 防止误差叠加的补偿值校正机制

---

### 📋 详细交互流程图

```mermaid
graph TB
    subgraph "系统初始化"
        A[启动模拟PLC服务器] --> B[启动算法程序]
        B --> C[建立TCP连接]
        C --> D[加载每层G代码文件]
        D --> E[开始状态轮询<br/>1秒间隔]
    end
    
    subgraph "单层处理循环"
        E --> F{PLC手动输入<br/>start N}
        F --> G[PLC状态: idle → processing<br/>开始第N层加工]
        G --> H{机床作业完成<br/>PLC输入: complete}
        H --> I[PLC状态: processing → waiting<br/>发送作业完成信号]
        
        I --> J[算法程序检测状态变化<br/>轮询发现waiting状态]
        J --> K[发出correction_request信号<br/>请求第N层纠偏数据]
        
        K --> L[开始纠偏处理<br/>延时0.5秒后启动]
        L --> M[相机图像采集]
        M --> N[偏差计算与分析]
        
        N --> O{判断层类型<br/>第1层 vs 后续层}
        O -->|第1层| P1[仅标定偏差<br/>建立基准数据]
        O -->|后续层| P2[应用前层补偿<br/>使用纠偏数据]
        
        P1 --> Q1[G代码基准处理<br/>不应用纠偏]
        P2 --> Q2[G代码纠偏处理<br/>应用offset_table.csv]
        
        Q1 --> R[生成输出文件<br/>output/layer_XX_out/]
        Q2 --> R
        
        R --> S{安全检查<br/>偏移量 < 20mm?}
        S -->|是| T[准备纠偏数据包<br/>包含文件路径]
        S -->|否| U[触发安全警告<br/>丢弃危险数据]
        
        T --> V[发送纠偏数据到PLC<br/>JSON格式+文件路径]
        V --> W[PLC接收并显示<br/>纠偏数据详情]
        
        U --> X[PLC显示安全警告<br/>使用原始路径]
        W --> Y{所有层完成?}
        X --> Y
        
        Y -->|否| Z{PLC手动输入<br/>next}
        Z --> AA[进入下一层<br/>layer++]
        AA --> F
        
        Y -->|是| BB[处理完成<br/>系统结束]
    end
    
    subgraph "数据结构详解"
        V --> CC[纠偏数据包结构:<br/>- corrected_gcode_path<br/>- offset_table_path<br/>- original_layer_XX.gcode<br/>- processing_metrics.json<br/>- bias_compensation.json<br/>- 可视化文件集]
        W --> DD[PLC显示内容:<br/>- 文件路径和大小<br/>- 纠偏统计信息<br/>- G代码调整详情<br/>- 数据质量指标<br/>- 安全检查结果]
    end
    
    subgraph "异常处理"
        EE[连接异常] --> FF[自动重连]
        GG[数据传输失败] --> HH[重试机制]
        II[偏差过大] --> JJ[自动丢弃+警告]
    end
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style F fill:#fff3e0
    style H fill:#fff3e0
    style Z fill:#fff3e0
    style S fill:#ffebee
    style U fill:#ffebee
    style O fill:#f3e5f5
    style P1 fill:#f3e5f5
    style P2 fill:#f3e5f5
    style V fill:#e8f5e8
    style W fill:#e8f5e8
```

## 交互时序图

```mermaid
sequenceDiagram
    participant PLC as 模拟PLC程序
    participant Algo as 算法程序
    participant FS as 文件系统
    
    Note over PLC,Algo: 系统初始化阶段
    PLC->>PLC: 启动服务器绑定端口502
    Algo->>PLC: TCP连接请求
    PLC-->>Algo: 连接确认
    Algo->>Algo: 启动状态轮询(1秒间隔)
    
    Note over PLC,Algo: 第N层处理循环
    PLC->>PLC: 手动输入"start N"
    PLC->>PLC: 状态改为processing
    
    loop 状态轮询
        Algo->>PLC: 查询机床状态
        PLC-->>Algo: 返回processing
    end
    
    PLC->>PLC: 手动输入"complete"  
    PLC->>PLC: 状态改为waiting
    
    Algo->>PLC: 查询机床状态
    PLC-->>Algo: 返回waiting
    Algo->>Algo: 检测状态变化，发出correction_request
    
    Note over Algo: 纠偏处理阶段
    Algo->>Algo: 延迟0.5秒后开始处理
    Algo->>Algo: 相机采集图像
    Algo->>Algo: 计算偏差和纠偏
    Algo->>FS: 生成corrected.gcode
    Algo->>FS: 生成offset_table.csv
    Algo->>FS: 生成其他辅助文件
    
    Note over Algo: 安全检查
    Algo->>Algo: 检查偏移量(<20mm)
    
    alt 数据安全
        Algo->>PLC: 发送纠偏数据包(含文件路径)
        PLC->>PLC: 接收并解析数据
        PLC->>FS: 验证文件存在性
        PLC->>PLC: 显示纠偏数据详情
        Note over PLC: 显示文件路径、大小、统计信息
    else 数据危险  
        Algo->>PLC: 发送安全警告
        PLC->>PLC: 显示警告信息
        Note over PLC: 自动丢弃危险数据
    end
    
    Note over PLC,Algo: 进入下一层
    PLC->>PLC: 手动输入"next"
    PLC->>PLC: 层号递增，状态改为idle
    
    Note over PLC,Algo: 重复循环直到所有层完成
```

## 数据包结构图

```mermaid
graph LR
    subgraph "算法程序输出"
        A[纠偏数据包] --> B[文件路径信息]
        A --> C[G代码调整信息] 
        A --> D[纠偏统计信息]
        A --> E[处理质量信息]
        
        B --> B1[corrected_gcode_path]
        B --> B2[offset_table_path]
        B --> B3[output_directory]
        B --> B4[available_files]
        
        C --> C1[line_number]
        C --> C2[original_line]
        C --> C3[corrected_line]
        C --> C4[offset_mm]
        
        D --> D1[avg_correction_mm]
        D --> D2[max_correction_mm]
        D --> D3[affected_gcode_lines]
        
        E --> E1[total_correction_points]
        E --> E2[valid_ratio]
        E --> E3[processing_time]
    end
    
    subgraph "PLC接收显示"
        F[接收确认] --> G[文件验证]
        F --> H[数据解析]
        F --> I[安全检查]
        
        G --> G1[文件大小检查]
        G --> G2[文件存在性验证]
        
        H --> H1[纠偏统计显示]
        H --> H2[G代码调整显示]
        H --> H3[质量指标显示]
        
        I --> I1[偏移量阈值检查]
        I --> I2[梯度变化检查]
    end
    
    A --> F
```