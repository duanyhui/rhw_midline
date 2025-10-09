# 多层加工纠偏系统 — 数据流与执行逻辑说明

本文面向新手，帮助快速理清 GUI 下从 PLC 通信 → 相机/算法处理图像 → 产出纠偏结果 → 回写 PLC 的完整数据流，以及“逐层”数据是如何组织与流转的。文末附上关键代码入口与文件行号，便于对照源码。

---

## 一、总体架构与数据流

- GUI 主窗体接入 PLC 信号，按机床状态与层号驱动算法处理；算法用相机采点云并与 G 代码对齐，输出纠偏数据（offset 表与 corrected.gcode），随后按 PLC 协议分批下发。
- 简化时序（自动模式）：

```
PLC(状态轮询/层号/请求) ──> MultilayerMainWindow ──(启动处理)──> LayerProcessingThread
                               │                                   │
                               │                                   ├─> AlignController.process_single_frame()
                               │                                   │     - 相机抓帧/投影/最近面/引导中心线/质量指标
                               │                                   ├─> 导出纠偏 (offset_table.csv, corrected.gcode)
                               │                                   └─> 生成每层 output/layer_XX_out 结构
                               └─(处理完成)──> PLC 通信器 (S7/TCP) ──> 分批写入 PLC (控制块+偏移块)
```

---

## 二、关键模块与职责

- GUI/调度
  - MultilayerMainWindow：界面、G 代码加载、层表、状态、启动/响应 PLC 请求、贯通处理与可视化
    - multilayer_main.py:47
- PLC 通信
  - PLCCommunicator 基类：轮询状态、读取层号、发纠偏/完成信号（TCP、S7、模拟）
    - multilayer_plc.py:39
  - TCPPLCCommunicator：JSON/TCP 通信，含健壮收发与重连
    - multilayer_plc.py:135
  - S7PLCEnhanced：基于 DB9044/9045–9047 的分批传输、心跳与数据锁
    - s7_plc_enhanced.py:21
  - PLCMonitorThread：后台轮询，触发层号变化与纠偏请求信号
    - multilayer_plc.py:719
- 图像/算法
  - AlignController：相机驱动、单帧流水线、导出纠偏、保存 bias_comp
    - controller.py:275
- 处理线程
  - LayerProcessingThread：单层处理、质量指标、导出 out 目录、写入结果
    - multilayer_processor.py:17
- 可视化
  - LayerVisualizationWidget：分视图显示、统计面板、导出视图
    - multilayer_visualizer.py:487

---

## 三、PLC → GUI 信号流与触发逻辑

1) 轮询与请求
- PLCMonitorThread 周期读取机床状态与层号；当状态变为 `waiting`（机床等待纠偏）时，发出 `correction_request(layer_id)`。
  - multilayer_plc.py:719

2) GUI 响应
- MultilayerMainWindow 订阅通信器信号：
  - 层号变化：`on_layer_changed_from_plc(layer_id)`，保持 UI 同步
    - multilayer_main.py:477
  - 机床状态：`on_machine_status_changed(status)`，在 `waiting` 下准备处理
    - multilayer_main.py:520
  - 纠偏请求：`on_correction_request(layer_id)`，校验层/G 代码后，排程开始处理（支持可配置延迟）
    - multilayer_main.py:564

3) 连接方式
- `tcp | s7 | s7_sim | mock` 四种；S7 增强版附带进度/心跳/安全警告信号：
  - s7_plc_enhanced.py:21

---

## 四、单层处理流水线（算法侧）

发起入口：`process_current_layer()`
- multilayer_main.py:416

处理线程：`LayerProcessingThread.run()` 执行流程：
- 1) G 代码设置：把当前层 G 代码路径写入控制器配置
- 2) 偏差补偿（bias）：
  - 第 1 层为“标定层”，不应用偏差，仅生成 `bias_comp.json`
  - 第 N>1 层，优先应用最近一层的 `bias_comp_data` 进行补偿
- 3) 相机采图 + 核心流水线：`AlignController.process_single_frame()`
  - 点云采集 → 机床坐标变换 → ROI → 顶视投影 → 平面展平（可配）→ 最近表面掩码 → 引导中心线（依据 G 代码切线/法线）→ 指标（轨迹距离分布）
  - controller.py:309
- 4) 二次质量指标：`calculate_corrected_metrics()` 基于“已补偿”的 Δn 重算轨迹精度（mean/median/p95/max consistency 等）
  - multilayer_processor.py:194
- 5) 结果与导出：
  - 标定层：保存 `bias_comp.json`，不导出纠偏 G 代码
  - 纠偏层：`export_corrected()` 导出 `offset_table.csv` 与 `corrected.gcode`，并生成 `output/layer_XX_out` 结构（见下一节）
  - controller.py:680, multilayer_processor.py:312

异常/完成：
- 线程完成信号 → `on_processing_finished(layer_id, result)`，写入 UI/表格、质量提示，并调用 PLC 下发纠偏（非第 1 层）
  - multilayer_main.py:707

---

## 五、每层数据的组织与落盘

- 层对象：`LayerInfo` 保存层号、G 代码路径、状态、处理/纠偏结果、bias_comp 等
  - multilayer_data.py:9
- 导出目录：每个纠偏层在 `output/layer_{layer_id:02d}_out/` 下存放：
  - `offset_table.csv` 偏移表（dx_mm, dy_mm）
  - `corrected.gcode` 纠偏后 G 代码
  - `centerline.gcode` 可选导出中心线
  - `corrected_preview.png`、`quicklook.png`、`report.json` 等
  - `layer_info.json`、`processing_metrics.json`、`bias_compensation.json`、`README.md`
  - multilayer_processor.py:312

- 可视化：GUI 右侧多视图（原始vs理论、纠偏后vs理论、误差分布、顶视高度、最近表面等），底部实时统计
  - multilayer_visualizer.py:487

---

## 六、纠偏数据下发（PLC 侧）

A) S7 增强协议（推荐）
- 控制块 DB9044 字段（机器/程序状态、总批次、数据锁、心跳、层类型等），偏移数据块 DB9045–9047 每块 128 点（dx,dy 各 2 字节，有符号大端）。
  - plc_data_structures.py:Control/MachineStatus/ProgramStatus 等
- 发送流程：
  1) 从 `correction` 中加载 offset 表（或直接点集）；调用 `OffsetDataLoader/Handler` 做安全验证与过滤（最大偏移、最大梯度、数据一致性）
     - offset_data_handler.py:1
  2) `create_batch_transmission_plan()` 规划分批（一次最多 384 点），逐批：等待解锁 → 写控制块统计 → 写 3 个偏移块 → 更新心跳/时间戳 → 解锁
     - s7_plc_enhanced.py:582
  3) 批间短暂延时，最后置 `COMPLETED`，并通过 `data_transmission_progress` 回报进度
- 异常与安全：超阈值触发 `safety_alert(layer_id, msg, value)`；重试与错误码落控制块
  - s7_plc_enhanced.py:532

B) TCP JSON 模式（兼容/调试）
- 通过 `_prepare_correction_data()` 附带 `corrected.gcode`、`offset_table.csv` 路径、指标等，以长度前缀 JSON 进行可靠收发，内置重连与超时处理
  - multilayer_plc.py:135

---

## 七、质量评估与流程控制

- GUI 质量提示：`check_processing_quality()` 依据覆盖率与轨迹 P95 距离给出“优秀/良好/可接受/需注意”
  - multilayer_main.py:739
- 线程二次指标：`calculate_corrected_metrics()` 优先展示轨迹精度（纠偏后）
  - multilayer_processor.py:194
- 偏移安全：`OffsetDataHandler` 最大偏移/梯度/过滤与插值；失败会触发安全警告或跳过下发
  - offset_data_handler.py:1
- 错误处理/重试：`handle_plc_data_error()` 与 `retry_data_transmission()`
  - multilayer_main.py:869, multilayer_main.py:928

---

## 八、逐层策略与偏差累积

- 第 1 层：标定，不下发纠偏，只生成 `bias_comp.json`。
- 第 N>1 层：查找最近一层的 `bias_comp`，在处理前应用（偏差累积/滚动补偿）；处理完成后保存本层 `bias_comp` 供下一层使用。
  - multilayer_main.py:443（查找上一层 bias），controller.py:795（保存 bias）

---

## 九、新手上手（最短路径）

1) 运行入口：`python run_multilayer_system.py`（会检查依赖/目录）；或 `python multilayer_main.py`
2) 左侧“G 代码管理”选择目录，自动按文件名顺序映射为层
3) “PLC 通信”选择类型与 IP/端口，点击“连接PLC”
4) 机床每层加工完成后进入 `waiting`，PLCMonitorThread 会触发纠偏请求 → 程序自动处理并下发纠偏
5) 右侧标签切换不同视图查看质量与可视化；必要时打开“高级参数调节”

---

## 十、源码索引（便于定位）

- multilayer_main.py:47        MultilayerMainWindow 主窗体
- multilayer_main.py:351       connect_plc 连接PLC并注册信号（含增强 S7）
- multilayer_main.py:416       process_current_layer 启动单层处理
- multilayer_main.py:477       on_layer_changed_from_plc 同步层号
- multilayer_main.py:520       on_machine_status_changed 机床状态
- multilayer_main.py:564       on_correction_request 处理PLC纠偏请求
- multilayer_main.py:707       on_processing_finished 处理完成与下发纠偏

- multilayer_plc.py:16         PLCDataProtocol 协议常量
- multilayer_plc.py:39         PLCCommunicator 抽象基类
- multilayer_plc.py:135        TCPPLCCommunicator JSON/TCP 实现
- multilayer_plc.py:719        PLCMonitorThread 轮询与信号

- s7_plc_enhanced.py:21        S7PLCEnhanced 增强通信器
- s7_plc_enhanced.py:532       send_correction_data 入口（含安全）
- s7_plc_enhanced.py:582       _transmit_offset_data_in_batches 分批写入 DB
- s7_plc_enhanced.py:461       read_current_layer 读当前层
- s7_plc_enhanced.py:476       read_machine_status 读机床状态

- controller.py:275            AlignController 控制器
- controller.py:309            process_single_frame 单帧处理流水线
- controller.py:680            export_corrected 导出纠偏
- controller.py:795            save_bias_from_current 保存 bias_comp.json

- multilayer_processor.py:17    LayerProcessingThread 单层线程
- multilayer_processor.py:194   calculate_corrected_metrics 纠偏后指标
- multilayer_processor.py:312   create_layer_out_directory 层输出结构

- multilayer_data.py:9         LayerInfo 层数据模型
- multilayer_data.py:43        ProjectConfig 项目/PLC/算法配置
- offset_data_handler.py:1     偏移数据读/验/滤/分批规划

---

若需要，我可以补充一页“PLC 数据块(DB9044/45/46/47)字段对照表”与“常见异常与排查 Checklist”。

