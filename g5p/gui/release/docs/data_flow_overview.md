# 多层加工纠偏系统数据流说明

> 目标：帮助新同事快速理解 GUI 环境下，从 PLC 通信触发，到算法处理图像并生成纠偏结果，再到分层数据管理的全链路执行逻辑。

## 1. 顶层结构概览

| 模块 | 位置 | 核心职责 |
| --- | --- | --- |
| `run_multilayer_system.py` | 根目录 | 启动依赖检查→创建 PyQt 应用→载入 `multilayer_main` 窗口 |
| `multilayer_main.py` | 根目录 | GUI 主窗体，负责 UI、项目状态管理、PLC 信号接入、处理线程编排 |
| `multilayer_plc.py` / `s7_plc_enhanced.py` | 根目录 | 各类 PLC 通信实现（TCP、S7、本地模拟），对上用 Qt 信号通知 GUI |
| `multilayer_processor.py` | 根目录 | `LayerProcessingThread`：单层加工的拍照、算法运行、纠偏导出和质量评估 |
| `controller.py` | 根目录 | 封装相机与 `align_centerline_to_gcode_pro_edit_max.py` 中的算法流程，并负责纠偏结果导出 |
| `multilayer_visualizer.py` | 根目录 | 右侧可视化与高级参数面板 |
| `multilayer_data.py` | 根目录 | 基础数据结构（`LayerInfo`、`ProjectConfig` 等） |

系统运行时的核心控制流如下：

```
+-------------------+       +-------------------------+
| PLCCommunicator   |  -->  | MultilayerMainWindow    |
|  (轮询/事件)      |  Qt   |  on_correction_request  |
+-------------------+ signal+------------+------------+
                                         |
                                         v
                                +----------------------+
                                | LayerProcessingThread|
                                |  (单层算法流水线)    |
                                +------+---------------+
                                       |
                                       v
                 +----------------------+---------------------+
                 | AlignController.process_single_frame()     |
                 |  - 相机采集 & 点云转换                     |
                 |  - ROI/展平/最近表面                        |
                 |  - 中心线偏差 & 指标计算                     |
                 +----------------------+---------------------+
                                       |
                   +-------------------+-------------------+
                   | 输出：纠偏文件 + 偏差补偿 + 指标 + 可视化 |
                   +-------------------+-------------------+
                                       |
           +---------------------------+-------------------------+
           |                              |                       |
           v                              v                       v
  更新 GUI 可视化                存储到 LayerInfo           发送到 PLC
```

## 2. PLC 通信到 GUI 的事件链

1. **配置来源**：
   - `ProjectConfig`（`multilayer_data.py`）保存 PLC 类型、IP、端口和数据地址，UI 中的“项目配置”面板直接绑定这些属性。
   - `MultilayerMainWindow.connect_plc()` 根据配置实例化对应的 `PLCCommunicator` 子类：
     - `TCPPLCCommunicator`：JSON over TCP。
     - `S7PLCCommunicator`（基础版）或 `S7PLCEnhanced`（带数据块管理、心跳与安全检查）。
     - `MockPLCCommunicator` / `S7SimulatorPLCCommunicator`：本地调试。

2. **连接与轮询**：
   - 成功 `connect()` 后：
     - 调用 `start_polling()` 启动 `QTimer`，定时触发 `_poll_plc_status()`。
     - 同时启动 `PLCMonitorThread`（500 ms 周期），持续读取层号、机床状态、开始信号。二者结合保证实时性。

3. **状态信号**：所有通信类继承自 `QObject`，统一发射 Qt 信号：
   - `connection_status(bool, str)` → `MultilayerMainWindow.on_plc_status_changed()` 更新状态栏。
   - `layer_changed(int)` → `on_layer_changed_from_plc()`，同步当前层号并刷新表格。
   - `machine_status_changed(str)` → `on_machine_status_changed()`，决定 GUI 按钮是否可操作。
   - `correction_request(int)`：当 PLC 状态变为 `waiting` 时触发，是算法处理的真正入口。
   - `S7PLCEnhanced` 还会发射 `data_transmission_progress`、`safety_alert`、`heartbeat_updated`，用于进度条和安全提示。

4. **纠偏请求处理**（`on_correction_request`）：
   - 校验层号与 G 代码文件是否存在。
   - 若当前仍有线程在跑，使用定时器排队等待。
   - 把目标层状态改为 `processing`，并按 `process_delay_sec` 预设延迟后调用 `process_layer_with_validation()` → `process_current_layer()`。

## 3. `LayerProcessingThread` 的单层流水线

`multilayer_processor.py` 中的 `LayerProcessingThread` 封装了完整的一层加工流程，并通过 Qt 信号回传进度与结果：

1. **准备阶段**：
   - 将当前层的 G 代码路径写入 `controller.cfg.gcode_path`。
   - 处理偏差补偿：
     - 第 1 层仅做标定，关闭 bias；
     - 第 2 层起，查找最近有 `bias_comp` 的层，将 JSON 临时存成文件，打开 `bias_enable`，后续导入。

2. **采集 + 算法**：
   - `controller.process_single_frame(for_export=False)`
     - 采集相机点云：`core.PCamMLSStream`。
     - 读取外参矩阵并转到机床坐标；
     - 依据 `roi_mode`、G 代码边界裁剪 ROI；
     - 计算顶视投影、遮挡、平面展平；
     - 最近表面提取、生成引导中心线；
     - 计算法向偏差 `delta_n`、忽略拐角、生成可视化和指标；
     - 结果缓存到 `controller.last`，供后续导出使用。

3. **指标修正**：
   - `calculate_corrected_metrics()` 根据导出 bias 后的轨迹重算精度（平均/中位/P95/最大距离）并写回 `result['metrics']`。

4. **导出与缓存**：
   - 第 1 层：保存 `bias_comp` 文件，标记为 `calibration`。
   - 其它层：
     - `controller.export_corrected()` 生成 `offset_table.csv`、`corrected.gcode`、可选 `centerline.gcode`、可视化 PNG、统计 JSON 等；
     - `create_layer_out_directory()` 把所有结果复制到 `output/layer_{XX}_out/` 下，包含原始 G 代码、纠偏文件、可视化图表、偏差补偿和指标。
   - 所有层都调用 `controller.save_bias_from_current()`（在 `controller` 内实现，保存当前帧的 bias）。

5. **结束**：
   - 记录处理时间、时间戳、层类型等元数据。
   - 若启用了 PLC 且层号 > 1，`MultilayerMainWindow.on_processing_finished()` 会调用 `plc_communicator.send_correction_data()`，把 JSON 中含有的 `corrected.gcode_path` 与 `offset_table.csv` 等路径推送给 PLC。

## 4. `AlignController` 算法管线细化

`controller.py` 的 `process_single_frame()` 是算法核心，内部大量调用 `align_centerline_to_gcode_pro_edit_max.py` 中的函数，可概括为：

1. **输入聚合**：
   - 载入外参 `T_cam2machine.npy`，解析当前层 G 代码、重新采样等长轨迹。
   - 针对 G 代码计算切向与法向向量、弧长、拐角忽略掩码。

2. **点云到栅格**：
   - 相机点云 → 机床坐标（`transform_cam_to_machine_grid`）。
   - 基于配置和 G 代码边界裁剪 ROI。
   - 计算视场像素尺寸。

3. **二维高度图**：
   - 顶视投影 `project_topdown_from_grid`；
   - 遮挡区填充；
   - RANSAC 平面拟合 + 展平（可选）。

4. **最近表面提取**：生成 `nearest_mask`、参考高度等，并获得可用于中心线搜索的高度场。

5. **中心线与偏差**：
   - 搜索沿 G 代码法向的偏移，得到 `delta_n`；
   - 计算有效掩码、守卫条件（有效率、偏差阈值、缺失段长度等）；
   - 生产可视化图（比较视图、分布图、探测可视化等）。

6. **缓存结构 (`self.last`)**：
   - 包含 `cfg`（当前参数）、`g_xy`、`N_ref`、`delta_n`、`valid_mask`、多种可视化图像、指标、偏差直方图等，为导出与后续层提供完整上下文。

7. **导出 (`export_corrected`)**：
   - 应用上一层 bias（如启用）；
   - 根据曲率自适应平滑、限幅以及梯度约束重新生成偏移；
   - 写 `offset_csv`、`corrected_gcode`、`quicklook.png`、`report.json` 等；
   - 绘制叠加了修正轨迹的预览图。

## 5. 分层数据管理

- `LayerInfo`：GUI 中“层管理”表每行数据的基石，字段包括 `gcode_path`、`status`、`bias_comp`、`processing_result` 等。
- `MultilayerMainWindow.layers` 是一个 `Dict[int, LayerInfo]`，每次加载 G 代码目录时重建。
- 处理完成后：
  - 更新 `layer_info.processing_result`，并将 `bias_comp_data` 缓存到内存中，供下一层直接使用；
  - `LayerVisualizationWidget.add_layer_data()` 把结果推送到右侧可视化面板；
  - `overall_stats_text` 汇总有效率、偏差、耗时等信息。

## 6. 典型运行场景

1. **手动模式**：
   - 操作员在左侧面板点击“处理当前层” → 直接进入 `process_current_layer()`。
   - 线程结束后，界面更新并提示是否自动处理下一层（由 `auto_next_check` 决定）。

2. **PLC 自动模式**：
   - PLC 完成某层加工 → 把状态改为 `waiting` → `PLCMonitorThread` 捕获并触发 `correction_request` → 系统自动处理并把纠偏文件回写给 PLC。
   - 若启用了 `S7PLCEnhanced`，`send_correction_data()` 会分批传输偏移表，带安全检查（最大偏移、最大梯度）并在异常时调用 `send_deviation_alert()`。

3. **异常与保护**：
   - 若有效率过低或轨迹精度不达标，`LayerProcessingThread` 会在日志与 UI 状态中给出警告。
   - 发送 PLC 前还有一层 `_check_correction_safety()`（`multilayer_plc.py`）确保偏移值安全。

## 7. 对新同事的建议

- **快速定位入口**：从 `run_multilayer_system.py → multilayer_main.py → process_current_layer()` 顺着看，能迅速抓住主流程。
- **调试技巧**：
  - 使用 `MockPLCCommunicator` 或 `S7SimulatorPLCCommunicator`，不依赖真实 PLC 即可演练完整流程。
  - 若要单步调试算法，可直接实例化 `AlignController` 并调用 `process_single_frame()`。
- **扩展算法**：
  - 在 `controller.py` 中添加新的可视化或指标时，记得更新 `self.last` 缓存结构以及 `LayerProcessingThread.create_layer_out_directory()` 的导出部分。
- **安全第一**：任何改动都要关注 `send_correction_data()` 的安全检查逻辑，避免把不合理偏移推到机床侧。

---
如需更细的函数调用链，可在 IDE 中按上述模块顺序设置断点或开启日志（`print` 在全流程中已有大量埋点，可辅助理解现场行为）。
