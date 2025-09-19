# 多层加工纠偏系统 - Output文件夹增强功能

## 🔧 **功能增强概述**

根据您的需求，系统已升级为将每层out文件夹保存在总的`output`文件夹内，并大幅丰富了每层out文件夹的内容。

## 📂 **新的目录结构**

### 处理过程中的文件组织
```
项目根目录/
├── output/                    # 🆕 总的输出文件夹
│   ├── layer_01_out/         # 第1层数据（标定层，无纠偏文件）
│   ├── layer_02_out/         # 第2层数据
│   │   ├── 🔧 核心纠偏文件
│   │   ├── offset_table.csv           # 偏移量表格
│   │   ├── corrected.gcode            # 纠偏后G代码
│   │   ├── original_layer_02.gcode    # 🆕 原始G代码
│   │   ├── 📊 可视化文件
│   │   ├── comparison_visualization.png    # 🆕 理论vs实际对比图
│   │   ├── corrected_visualization.png     # 🆕 纠偏后可视化
│   │   ├── corrected_preview.png           # 纠偏预览图
│   │   ├── probe_visualization.png         # 🆕 探针可视化
│   │   ├── histogram_panel.png             # 🆕 偏差直方图
│   │   ├── top_view.png                    # 🆕 顶视图
│   │   ├── nearest_surface.png             # 🆕 最近表面图
│   │   ├── 📋 数据文件
│   │   ├── processing_metrics.json         # 🆕 处理质量指标
│   │   ├── bias_compensation.json          # 🆕 偏差补偿数据
│   │   ├── layer_info.json                 # 🆕 层详细信息
│   │   └── README.md                       # 🆕 层级说明文档
│   └── layer_03_out/         # 第3层数据...
└── 其他项目文件...
```

### 导出时的文件组织
```
project_export/
├── machine_data/              # 机床纠偏数据（从output复制）
│   ├── layer_02_out/         # 完整的层数据
│   ├── layer_03_out/
│   ├── overall_statistics.json    # 🆕 总体质量统计
│   └── README.md
├── visualization/             # 可视化结果
│   ├── layer_01/
│   └── layer_02/
├── project_summary.json       # 项目摘要
└── README.md                  # 🆕 导出说明
```

## 🎯 **增强功能详解**

### 1. **丰富的文件内容**

每个`layer_XX_out`文件夹现在包含：

#### 核心纠偏文件
- `offset_table.csv` - 偏移量表格，机床直接使用
- `corrected.gcode` - 纠偏后的G代码文件
- `original_layer_XX.gcode` - **新增**：原始G代码文件备份

#### 完整可视化套件
- `comparison_visualization.png` - **新增**：理论轨迹vs实际轨迹对比
- `corrected_visualization.png` - **新增**：纠偏后效果可视化
- `corrected_preview.png` - 纠偏预览图
- `probe_visualization.png` - **新增**：探针法线可视化
- `histogram_panel.png` - **新增**：偏差分布直方图
- `top_view.png` - **新增**：顶视高度图
- `nearest_surface.png` - **新增**：最近表面分析图

#### 数据与元信息
- `processing_metrics.json` - **新增**：处理质量指标详细数据
- `bias_compensation.json` - **新增**：当前层的偏差补偿数据
- `layer_info.json` - **新增**：层处理信息和状态
- `README.md` - **新增**：每层的详细说明文档

### 2. **统一的Output管理**

- **集中存储**：所有层数据统一保存在`output/`文件夹下
- **标准命名**：采用`layer_{layer_id:02d}_out`格式
- **即时创建**：每层处理完成后立即创建对应目录
- **完整备份**：包含原始和处理后的所有相关文件

### 3. **增强的导出功能**

#### 机床数据导出
- 从`output/`文件夹复制所有层数据到`machine_data/`
- 生成总体质量统计文件
- 创建详细的使用说明文档

#### 质量分析增强
- **层级质量对比**：最佳层vs最差层分析
- **平均指标计算**：有效率、P95偏差等
- **趋势分析数据**：支持质量改进决策

## 🚀 **使用方式**

### 1. **实时使用**
```python
# 每层处理完成后，系统自动在 output/ 下创建层目录
# 包含该层的所有数据文件
```

### 2. **机床集成**
```python
# 机床可按以下方式访问数据：
layer_path = f"output/layer_{current_layer:02d}_out/"
gcode_file = layer_path + "corrected.gcode"
offset_file = layer_path + "offset_table.csv"
```

### 3. **批量导出**
```python
# 使用"导出结果"功能
# 自动整理所有数据到 machine_data/ 文件夹
# 生成完整的项目报告和统计
```

## 📊 **质量监控**

### 新增质量指标追踪
- **实时质量监控**：每层处理完成后立即生成质量报告
- **历史趋势分析**：支持多层质量对比
- **异常层识别**：自动标识质量异常的层

### 质量数据格式
```json
{
    "layer_id": 2,
    "valid_ratio": 0.856,
    "dev_p95": 2.145,
    "dev_mean": 0.023,
    "plane_inlier_ratio": 0.892
}
```

## 🔧 **技术特点**

### 1. **数据完整性**
- 原始G代码完整保留
- 处理过程数据全程记录
- 可视化结果多维度覆盖

### 2. **便于机床集成**
- 标准化文件命名和结构
- JSON格式的机器可读元数据
- 详细的人类可读说明文档

### 3. **质量可追溯**
- 每层处理时间记录
- 质量指标完整保存
- 偏差补偿数据链追踪

## 💡 **优势总结**

1. **🎯 便于机床调用**：统一的output文件夹，标准化访问路径
2. **📊 数据丰富完整**：原始+处理+可视化+元数据一应俱全
3. **🔍 质量可追溯**：每层完整的处理记录和质量分析
4. **📁 文件组织清晰**：层级目录结构，便于管理和维护
5. **🚀 即用即得**：处理完成即可直接使用，无需额外整理

现在您的多层加工纠偏系统具备了完整的数据管理和导出功能，每层都有独立且丰富的out文件夹，保存在统一的output目录中，便于机床按层号调用和质量追溯！