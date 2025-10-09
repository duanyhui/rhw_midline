# Centerline ↔ G-code (Guided Fit) — STRICT One-to-One Edition 说明文档

## 概述

### 功能简介
该算法将相机点云投影为顶视高度图，在G-code引导下沿法线匹配实际中心线，严格一一对应地计算法向偏移并安全导出纠偏G-code。

### 适用场景
- **目标**：激光切割、3D打印等精密加工的轨迹纠偏
- **输入**：相机点云数据 + 理论G-code文件 + 相机标定外参
- **输出**：偏移表CSV + 纠偏后G-code + 质量报告

### 非目标场景
- 不适用于多层复杂几何体
- 不处理Z轴高度变化的3D轨迹
- 不支持实时在线纠偏

### 坐标系约定
- **右手系**：+X向右，+Y向上（机床坐标系）
- **像素系**：行向下，列向右（OpenCV约定）
- **法向偏移`delta_n`**：正值表示实际中心线偏向法向量正方向

### 与旧版差异速览
- **[CHG] 严格一一对应**：以G代码等弧长采样点为唯一主序列，逐点沿法线搜索，导出不再使用KDTree聚合
- **[CHG] 缺失策略**：仅对短缺口（≤`max_gap_pts`）做局部插值，长段缺失直接判FAIL
- **[NEW] 梯度门限**：`|Δδ/Δs| ≤ guide_max_grad_mm_per_mm`，抑制抖动
- **[NEW] 曲率自适应平滑**：高曲率区域缩短平滑窗口
- **[NEW] 拐角忽略**：拐角附近忽略取点，避免跨段传播
- **[NEW] 遮挡处理**：固定设备遮挡区域的多边形标定与环带合成

## 快速开始

### 依赖环境
```bash
# 必需
pip install opencv-python numpy
# Percipio SDK（相机驱动）
pip install pcammls  # 或按厂商说明安装

# 可选（用于高级形态学操作）
pip install scikit-image scipy
```

### 最小运行步骤
1. **准备文件**：
   - 相机外参：`T_cam2machine.npy`
   - 理论G-code：`args/example.gcode`

2. **运行脚本**：
```bash
python align_centerline_to_gcode_pro_edit_max.py
```

3. **交互操作**：
   - `q`：退出程序
   - `s`：保存当前可视化截图
   - `c`：导出纠偏结果（默认按键，可配置）
   - `+/-`：调整像素分辨率

4. **查看结果**：
   - `out/offset_table.csv`：偏移表
   - `out/corrected.gcode`：纠偏后G-code
   - `out/report.json`：质量报告

## 参数总览

### 文件与路径
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `T_path` | `'T_cam2machine.npy'` | str | 相机到机床外参文件，包含R,t,T矩阵 |
| `gcode_path` | `'args/example.gcode'` | str | 理论G-code文件路径 |
| `out_dir` | `'out'` | str | 导出目录 |
| `offset_csv` | `'out/offset_table.csv'` | str | 偏移表CSV路径 |
| `corrected_gcode` | `'out/corrected.gcode'` | str | 纠偏后G-code路径 |
| `centerline_gcode` | `'out/centerline.gcode'` | str | 测得中心线G-code（仅对比用） |

### 顶视投影与ROI
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `pixel_size_mm` | `0.8` | float | 顶视像素分辨率(mm/pixel)，越小越精细但占用内存更多 |
| `bounds_qlo` | `1.0` | float | ROI边界百分位下限，排除极值点 |
| `bounds_qhi` | `99.0` | float | ROI边界百分位上限 |
| `bounds_margin_mm` | `20.0` | float | ROI边界扩展裕量(mm) |
| `max_grid_pixels` | `1200000` | int | 网格最大像素数，超过则自动调整分辨率 |

### 最近表层提取
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `z_select` | `'auto'` | str | 层选择模式：'auto'自动/'manual'手动 |
| `depth_margin_mm` | `1.2` | float | 深度选择裕量(mm) |
| `nearest_use_percentile` | `True` | bool | 是否使用百分位法选层 |
| `nearest_qlo` | `15.0` | float | 最近表层百分位下限 |
| `nearest_qhi` | `85.0` | float | 最近表层百分位上限 |
| `morph_open` | `3` | int | 形态学开运算核大小，去噪 |
| `morph_close` | `5` | int | 形态学闭运算核大小，填洞 |
| `min_component_area_px` | `500` | int | 最小连通域面积(像素)，过滤小碎片 |

### 平面拟合/展平
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `plane_sample_cap` | `120000` | int | 平面拟合最大采样点数 |
| `plane_min_inlier_ratio` | `0.55` | float | 平面内点率最低要求，低于此值Guard失败 |

### 引导中轴线（核心）
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `guide_enable` | `True` | bool | 是否启用G-code引导模式 |
| `guide_step_mm` | `1.0` | float | 等弧长重采样步长(mm)，影响匹配精度 |
| `guide_halfwidth_mm` | `6.0` | float | 法向扫描半宽(±mm)，应大于预期偏差 |
| `guide_use_dt` | `True` | bool | 是否使用距离变换优化 |
| `guide_min_on_count` | `3` | int | 等距带最小有效像素数 |
| `guide_smooth_win` | `7` | int | 基础平滑窗口大小（奇数） |
| `guide_max_offset_mm` | `8.0` | float | 法向偏移幅值上限(mm) |
| `guide_max_grad_mm_per_mm` | `0.08` | float | **[NEW]** 梯度限制(mm/mm)，抑制跳变 |
| `curvature_adaptive` | `True` | bool | **[NEW]** 是否启用曲率自适应平滑 |
| `curvature_gamma` | `35.0` | float | **[NEW]** 曲率敏感度参数 |
| `min_smooth_win` | `3` | int | **[NEW]** 自适应平滑最小窗口 |
| `guide_min_valid_ratio` | `0.60` | float | 最低命中率要求 |

### 拐角忽略
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `corner_ignore_enable` | `False` | bool | **[NEW]** 是否启用拐角忽略 |
| `corner_angle_thr_deg` | `35.0` | float | 拐角角度阈值(度)，≥此角度判为拐角 |
| `corner_ignore_span_mm` | `2.0` | float | 拐角两侧忽略的弧长半径(mm) |

### 遮挡处理
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `occlusion.enable` | `True` | bool | **[NEW]** 是否启用遮挡处理 |
| `occlusion.polys` | `[...]` | list | 遮挡多边形列表（机床XY坐标，mm） |
| `occlusion.dilate_mm` | `3.0` | float | 安全扩张距离(mm) |
| `occlusion.synthesize_band` | `True` | bool | 是否在遮挡区合成环带掩码 |
| `occlusion.band_halfwidth_mm` | `None` | float/None | 环带半宽(mm)，None=自动估计 |

### 缺失处理与梯度限幅
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `strict_one_to_one` | `True` | bool | **[NEW]** 强制一一对应，仅基于法向单点决策 |
| `max_gap_pts` | `5` | int | **[NEW]** 允许插补的最大缺口点数 |
| `long_missing_max_mm` | `20.0` | float | **[NEW]** 长段缺失长度阈值(mm) |
| `long_missing_max_ratio` | `0.08` | float | **[NEW]** 长段缺失比例阈值 |

### 偏差补偿
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `bias_comp.enable` | `True` | bool | 是否启用偏差补偿 |
| `bias_comp.path` | `'bias_comp.json'` | str | 补偿标定文件路径 |
| `offset_apply_mode` | `'invert'` | str | 纠偏模式：'invert'朝理论轨迹/'follow'跟随实际 |
| `auto_flip_offset` | `True` | bool | 自动检测偏移符号并翻转 |

### 导出控制
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `export_on_key` | `'c'` | str | 导出触发按键 |
| `export_centerline` | `False` | bool | 是否导出测得中心线G-code |
| `preview_corrected` | `True` | bool | 是否预览纠偏轨迹叠加 |
| `save_corrected_preview` | `True` | bool | 是否保存纠偏预览图 |

### Guard质量门槛
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `Guard.enable` | `True` | bool | 是否启用质量门槛检查 |
| `Guard.min_valid_ratio` | `0.60` | float | 最低命中率 |
| `Guard.max_abs_p95_mm` | `8.80` | float | 偏移绝对值P95上限(mm) |
| `Guard.min_plane_inlier_ratio` | `0.25` | float | 平面内点率下限 |
| `Guard.long_missing_max_mm` | `20.0` | float | **[NEW]** 长段缺失长度上限(mm) |
| `Guard.grad_max_mm_per_mm` | `0.08` | float | **[NEW]** 梯度P98上限 |

### 可视化与调试
| 参数 | 默认值 | 类型 | 作用与调整建议 |
|------|--------|------|----------------|
| `colormap` | `cv2.COLORMAP_TURBO` | int | 偏差可视化色彩映射 |
| `arrow_stride` | `12` | int | 偏差箭头显示间隔 |
| `debug_normals_window` | `True` | bool | **[NEW]** 法线探针调试窗口 |
| `debug_normals_stride` | `25` | int | 法线显示间隔 |
| `debug_normals_max` | `40` | int | 最大法线显示数量 |
| `debug_normals_text` | `True` | bool | 是否标注delta_n数值 |
| `dump_quicklook` | `True` | bool | 生成快速预览图 |
| `dump_report` | `True` | bool | 生成详细报告JSON |

## 主流程工作流

### 整体流程图
```
[1] 加载外参 & G-code
         ↓
[2] 获取相机点云数据
         ↓  
[3] 顶视投影 & ROI确定
         ↓
[4] 平面展平(可选)
         ↓
[5] 最近表层提取
         ↓
[6] [OCCLUSION] 遮挡区处理
         ↓
[7] G-code等弧长重采样
         ↓
[8] 法向扫描匹配(严格一一对应)
         ↓
[9] 短缺口插补 & 平滑 & 限幅
         ↓
[10] Guard质量检查
         ↓
[11] 偏差补偿 & 导出
```

### 关键步骤详解

#### [1] 初始化
- 加载外参矩阵：`T_cam2machine.npy`
- 解析G-code：提取XY坐标序列，支持G0/G1/G2/G3
- 建立相机流：初始化Percipio SDK

#### [2-3] 顶视投影
- 点云变换：相机坐标→机床坐标→顶视XY网格
- ROI确定：基于G-code边界box + 裕量
- 高度图生成：Z向最大值投影

#### [4] 平面展平（可选）
- RANSAC平面拟合：从高度图中提取主平面
- 残差计算：`height_flat = height - plane_fit`
- 内点率检查：低于阈值触发Guard失败

#### [5] 最近表层提取
- 层选择：auto模式下使用百分位法
- 形态学处理：开运算去噪 + 闭运算填洞
- 连通域过滤：移除小于`min_component_area_px`的碎片

#### [6] 遮挡区处理 **[NEW]**
- 多边形光栅化：机床XY → 像素掩码
- 可见区半宽估计：基于距离变换
- 环带合成：在遮挡区按G-code路径合成等宽带掩码

#### [7] G-code重采样
- 等弧长插值：按`guide_step_mm`重新采样
- 法向量计算：基于切向量的90°旋转
- 拐角检测：相邻段夹角≥阈值判为拐角

#### [8] 法向扫描匹配 **[严格一一对应]**
```python
for i in range(len(gcode_points)):
    if corner_ignore_mask[i]:  # 拐角忽略
        continue
    
    # 沿法向量扫描 ±halfwidth_mm
    intersections = find_contour_intersections(point_i, normal_i)
    
    # 在等距带 d_out - d_in ≈ 0 处取最佳交点
    best_point = select_best_intersection(intersections)
    
    if best_point:
        delta_n[i] = distance_along_normal(point_i, best_point)
        valid_mask[i] = True
```

#### [9] 后处理流水线
```
原始delta_n → 短缺口插补 → 曲率自适应平滑 → 幅值限幅 → 梯度限幅 → delta_n_final
```

#### [10-11] Guard检查与导出
- 统计指标：命中率、P95偏差、长段缺失等
- Guard判定：所有指标通过才允许导出
- 偏差补偿：基于`bias_comp.json`修正系统性偏差
- 文件导出：CSV偏移表 + 纠偏G-code + 质量报告

### 关键中间变量
| 变量 | 含义 | 形状 |
|------|------|------|
| `height` | 顶视高度图 | (H, W) |
| `height_flat` | 平面展平后残差 | (H, W) |
| `nearest_mask` | 最近表层掩码 | (H, W) bool |
| `g_xy` | G-code重采样坐标 | (N, 2) |
| `N_ref` | 法向量序列 | (N, 2) |
| `delta_n` | 法向偏移(原始) | (N,) |
| `valid_mask` | 有效匹配标记 | (N,) bool |
| `centerline_xy` | 实际中心线坐标 | (N, 2) |

## 关键算法细节

### 法向扫描与等距带
**核心思想**：在G-code理论点沿法向量方向扫描，寻找实际轮廓与"等距带"(`d_out - d_in ≈ 0`)的交点。

```python
# 等距带条件
phi = d_out - d_in  # 外距离变换 - 内距离变换
target_phi = 0.0    # 理想中轴线位置

# 沿法向量扫描
for offset in range(-halfwidth_px, +halfwidth_px):
    sample_point = base_point + offset * normal_px
    phi_value = bilinear_sample(phi, sample_point)
    
    # 寻找φ≈0的交点
    if abs(phi_value) < threshold:
        intersections.append((sample_point, phi_value))
```

**退化处理**：当等距带信号微弱时，回退到距离变换最大值位置。

### 短缺口插补与长段缺失 **[NEW]**
**短缺口插补**（≤ `max_gap_pts`）：
```python
def local_interpolate_short_gaps(data, valid_mask, max_gap_pts):
    gaps = find_continuous_nan_segments(data)
    for gap_start, gap_end in gaps:
        gap_length = gap_end - gap_start + 1
        if gap_length <= max_gap_pts:
            # 线性插值
            data[gap_start:gap_end+1] = interpolate(
                data[gap_start-1], data[gap_end+1], gap_length
            )
```

**长段缺失判定**：
- 绝对长度：连续缺失弧长 > `long_missing_max_mm`
- 相对比例：缺失点数/总点数 > `long_missing_max_ratio`
- 触发Guard失败，拒绝导出

### 曲率自适应平滑 **[NEW]**
**动机**：直线段可用长窗口平滑去噪，弯道处应缩短窗口避免拖尾。

```python
def adaptive_smooth_window(base_win, kappa, gamma, win_min):
    # 曲率越大，窗口越小
    adaptive_factor = np.exp(-gamma * kappa)
    actual_win = base_win * adaptive_factor
    return np.clip(actual_win, win_min, base_win)
```

### 梯度限幅 **[NEW]**
**目的**：抑制相邻点间的跳变，确保物理合理性。

```python
def gradient_clamp(delta, arc_s, gmax):
    """限制 |Δδ/Δs| ≤ gmax"""
    for i in range(1, len(delta)):
        ds = max(1e-9, arc_s[i] - arc_s[i-1])
        max_change = gmax * ds
        
        actual_change = delta[i] - delta[i-1]
        if abs(actual_change) > max_change:
            delta[i] = delta[i-1] + np.sign(actual_change) * max_change
```

### 拐角忽略策略 **[NEW]**
**检测拐角**：相邻切向量夹角 ≥ `corner_angle_thr_deg`
**忽略范围**：拐角顶点 ± `corner_ignore_span_mm` 弧长
**导出策略**：忽略区域强制`delta_n = 0`，保持理论G-code坐标不变

### 遮挡处理详解 **[NEW]**

#### 多边形标定
```python
# 示例：左下角62x52mm矩形遮挡（相对机床原点）
occlusion_polys = [
    [(-50, -50), (30, -30), (30, 70), (-50, 70)]  # 顺/逆时针均可
]
```

#### 可见区半宽估计
1. 在非遮挡区域计算距离变换
2. 沿G-code路径采样距离值
3. 取中位数作为可见半宽估计

#### 环带掩码合成
在遮挡区域内，基于G-code路径生成等宽"虚拟环带"：
```python
# 遮挡区内的G-code段
occluded_segments = intersect(gcode_path, occlusion_polygons)

# 为每段生成±halfwidth的带状掩码
for segment in occluded_segments:
    band_mask = dilate_polyline(segment, halfwidth=estimated_width)
    synthesized_mask |= band_mask
```

### 偏差补偿机制

#### 两种模式
1. **per-index表模式**：逐点补偿表，精确但需预标定
2. **向量模式**：`bias = v·N + b`，参数化模型

#### 步长对齐
```python
# 检查补偿表与当前G-code的步长一致性
step_tolerance = 1e-9
step_ok = abs(bias_file['guide_step_mm'] - current_step_mm) < step_tolerance

if not step_ok:
    # 基于拐角锁定的分段重采样
    resampled_bias = resample_bias_piecewise(
        bias_table, current_length, corner_knots
    )
```

#### 应用策略
```python
# 仅对有效测量点应用补偿，NaN点保持不变
valid_indices = np.isfinite(delta_n_measured)
delta_n_measured[valid_indices] -= bias[valid_indices]
```

## Guard质量门槛与导出策略

### Guard检查项目

| 指标 | 阈值参数 | 失败原因 | 排查建议 |
|------|----------|----------|----------|
| 命中率 | `min_valid_ratio=0.60` | 法向扫描失败过多 | 增大`guide_halfwidth_mm`，检查最近表层质量 |
| P95偏差 | `max_abs_p95_mm=8.80` | 存在异常大偏差 | 检查ROI边界，排除干扰轮廓 |
| 平面内点率 | `min_plane_inlier_ratio=0.25` | 表面非平面或噪声过大 | 调整相机曝光，改善点云质量 |
| 长段缺失(长度) | `long_missing_max_mm=20.0` | 连续缺失段过长 | 检查遮挡标定，调整扫描参数 |
| 长段缺失(比例) | `long_missing_max_ratio=0.08` | 总缺失比例过高 | 同上 |
| 梯度P98 | `grad_max_mm_per_mm=0.08` | 存在剧烈跳变 | 增大平滑窗口，检查数据质量 |

### 导出决策流程
```
Guard检查 → PASS → 偏差补偿 → 符号判定 → 生成导出文件
           ↓
         FAIL → 拒绝导出，显示失败原因
```

### 导出文件清单

#### 1. 偏移表CSV (`offset_table.csv`)
```csv
arc_length_mm,x_theory_mm,y_theory_mm,delta_n_mm,x_actual_mm,y_actual_mm
0.000,10.000,20.000,0.123,10.045,20.089
1.000,11.000,20.000,-0.067,10.982,19.933
...
```

#### 2. 纠偏G-code (`corrected.gcode`)
```gcode
; Original: G1 X10.000 Y20.000
G1 X10.045 Y20.089
; Original: G1 X11.000 Y20.000  
G1 X10.982 Y19.933
...
```

#### 3. 质量报告JSON (`report.json`)
```json
{
  "timestamp": "2024-01-01 12:00:00",
  "guard_status": "PASS",
  "statistics": {
    "valid_ratio": 0.87,
    "delta_n_p95_mm": 2.45,
    "plane_inlier_ratio": 0.78,
    "longest_missing_mm": 8.5
  },
  "bias_compensation": {
    "enabled": true,
    "mode": "per_index",
    "mean_correction_mm": -0.034
  }
}
```

#### 4. 可视化图片
- `out/quicklook.png`：偏差热力图概览
- `out/corrected_preview.png`：纠偏前后轨迹对比
- `out/bias_histogram.png`：偏差分布直方图

## 可视化与排障手册

### 关键可视化窗口

#### 1. 主窗口：偏差热力图叠加
**观察要点**：
- 蓝色=负偏差（向左），红色=正偏差（向右）
- 绿色箭头=偏差方向和大小
- 白色骨架线=骨架提取结果（仅参考）

#### 2. 法线探针窗口 **[NEW]** (`debug_normals_window=True`)
**观察要点**：
- 白色圆点=理论G-code采样点
- 浅蓝线段=法向量方向
- 品红圆点=实际匹配交点
- 数字标注=该点的`delta_n`值(mm)

**异常信号**：
- 大量缺失交点 → `guide_halfwidth_mm`太小或表层质量差
- 交点位置异常 → 轮廓提取或等距带计算有误

#### 3. 偏差补偿对比窗口 (`bias_comp.enable=True`)
**观察要点**：
- 蓝色直方图=原始偏差分布
- 红色直方图=补偿后偏差分布
- 理想情况：补偿后分布更集中于0附近

#### 4. 骨架提取窗口
**观察要点**：
- 彩色=最近表层掩码
- 白色线条=提取的骨架
- **注意**：骨架仅用于可视化，不参与最终导出

### 常见问题排障

#### 问题1：Guard失败 - 命中率过低
**现象**：`valid_ratio < min_valid_ratio`
**可能原因**：
- `guide_halfwidth_mm`设置过小
- 最近表层掩码质量差
- ROI边界截断了有效区域

**排查步骤**：
1. 观察法线探针窗口，查看缺失交点的分布
2. 检查最近表层掩码是否连续
3. 逐步增大`guide_halfwidth_mm`从6.0到12.0mm
4. 调整`bounds_margin_mm`确保ROI完整覆盖

#### 问题2：Guard失败 - P95偏差过大
**现象**：`delta_n_p95_mm > max_abs_p95_mm`
**可能原因**：
- 存在干扰轮廓或错误匹配
- ROI包含了非目标物体
- 相机标定不准确

**排查步骤**：
1. 检查主窗口偏差热力图，定位异常大偏差区域
2. 确认ROI边界是否包含干扰物
3. 验证相机外参`T_cam2machine.npy`的准确性
4. 考虑启用`corner_ignore_enable=True`排除拐角干扰

#### 问题3：Guard失败 - 长段缺失
**现象**：`longest_missing_mm > long_missing_max_mm`
**可能原因**：
- 遮挡区域未正确标定
- 表层质量在某段轨迹上极差
- `guide_halfwidth_mm`局部不足

**排查步骤**：
1. 检查`occlusion.polys`是否正确标定固定遮挡
2. 观察缺失段在哪些区域集中出现
3. 增大`guide_halfwidth_mm`或启用`occlusion.synthesize_band=True`
4. 调整最近表层提取参数改善掩码质量

#### 问题4：偏差补偿不生效
**现象**：启用`bias_comp.enable=True`但补偿前后无差异
**可能原因**：
- `bias_comp.json`文件不存在或格式错误
- 步长不匹配：`guide_step_mm`与补偿表不一致
- 拐角忽略导致补偿点被跳过

**排查步骤**：
1. 检查终端日志是否有`[BIAS] failed:`错误信息
2. 验证`bias_comp.json`的`guide_step_mm`字段
3. 查看偏差补偿对比窗口确认直方图变化
4. 检查`corner_ignore_enable`设置与补偿表的兼容性

### 调试检查清单

**启动前检查**：
- [ ] `T_cam2machine.npy`文件存在且外参矩阵合理
- [ ] G-code文件路径正确，包含有效的G0/G1/G2/G3指令
- [ ] 相机连接正常，pcammls SDK可用
- [ ] 输出目录`out_dir`可写

**运行中观察**：
- [ ] ROI覆盖完整，无关键区域被截断
- [ ] 最近表层掩码连续，无大片空洞
- [ ] 法线探针窗口显示合理的交点分布
- [ ] 主窗口偏差分布符合预期（无异常大值）

**导出前验证**：
- [ ] Guard所有指标通过
- [ ] 偏差补偿效果（如启用）符合预期
- [ ] 预览图显示纠偏轨迹合理
- [ ] 安全阈值检查：最大偏移<20mm（依据记忆中的安全控制要求）

## 示例与最佳实践

### 最小配置示例
```python
# 基础配置（安全保守设置）
PARAMS_MINIMAL = dict(
    # 文件
    T_path='T_cam2machine.npy',
    gcode_path='args/example.gcode',
    
    # 核心参数
    pixel_size_mm=1.0,              # 较粗分辨率，节省内存
    guide_step_mm=1.0,              # 标准采样精度
    guide_halfwidth_mm=8.0,         # 较宽扫描，提高命中率
    guide_smooth_win=5,             # 保守平滑
    
    # 安全门槛
    guide_max_offset_mm=6.0,        # 较严格的偏移限制
    Guard=dict(
        enable=True,
        min_valid_ratio=0.50,       # 降低命中率要求
        max_abs_p95_mm=6.0,         # 更严格的P95限制
    ),
    
    # 禁用复杂功能
    corner_ignore_enable=False,
    occlusion=dict(enable=False),
    bias_comp=dict(enable=False),
)
```

### 典型运行日志
```
[INIT] 加载外参: T_cam2machine.npy
[GCODE] 解析G-code: 1247 points, 1246.8mm total length
[CAM] 初始化相机: Percipio SDK v2.1
[ROI] 确定区域: X[10.2, 89.7]mm, Y[15.1, 76.3]mm
[PROJ] 顶视投影: 1024x768 grid, 0.8mm/pixel
[PLANE] 平面拟合: inlier_ratio=0.82 > 0.25 [PASS]
[LAYER] 最近表层: 87.3% area coverage
[GUIDE] G-code重采样: 1247→1248 points (1.0mm step)
[MATCH] 法向扫描: valid_ratio=0.74 > 0.60 [PASS]
[GUARD] 质量检查:
  ✓ valid_ratio: 0.74 ≥ 0.60
  ✓ delta_n_p95: 2.31mm ≤ 8.80mm
  ✓ plane_inlier: 0.82 ≥ 0.25
  ✓ longest_missing: 8.5mm ≤ 20.0mm
  → Guard: PASS
[BIAS] 偏差补偿: 应用per_index表，平均纠偏-0.034mm
[EXPORT] 导出完成:
  - out/offset_table.csv: 1248 rows
  - out/corrected.gcode: 1247 lines
  - out/report.json: Guard通过
```

## 维护与扩展

### 版本变更记录位点
源码顶部注释明确标注了主要变更：
- **[CHG] 严格一一对应**：不再使用KDTree聚合，纯粹基于法向匹配
- **[NEW] 梯度门限**：`guide_max_grad_mm_per_mm`参数抑制跳变
- **[NEW] 曲率自适应**：`curvature_adaptive`高曲率区域缩短平滑窗口
- **[NEW] 拐角忽略**：`corner_ignore_enable`避免拐角附近的跨段传播
- **[NEW] 遮挡处理**：`occlusion.polys`多边形标定与环带合成

### 常见扩展需求

#### 新增设备遮挡
1. 在机床坐标系下测量遮挡物边界
2. 更新`PARAMS['occlusion']['polys']`添加新多边形
3. 调整`dilate_mm`确保安全裕量
4. 验证`synthesize_band`效果

#### 迁移旧版偏差表
1. 检查旧表的步长与当前`guide_step_mm`兼容性
2. 如不兼容，使用`_resample_per_index_bias_piecewise`重采样
3. 转换为新的JSON格式，包含`mode`、`guide_step_mm`等元数据
4. 验证拐角锁定的一致性

#### 新增Guard指标
1. 在`run()`主循环中计算新统计量
2. 添加对应的`Guard.*`参数到`PARAMS`
3. 在Guard检查逻辑中增加新判定条件
4. 更新失败原因的文本描述

### 术语与符号约定

- **坐标系**：右手系，+X右，+Y上（机床）；像素系Y向下
- **量纲**：所有空间距离以mm为单位，角度以度为单位
- **统计量**：P95=95%分位数，median=中位数，mean=平均值
- **delta_n**：法向偏移，正值=实际中心线偏向法向量正方向
- **有效测量**：`valid_mask[i]=True`且`isfinite(delta_n[i])`
- **长段缺失**：连续缺失弧长>`long_missing_max_mm`的区间

---

**完整文档总结**：本文档涵盖了从参数配置到故障排除的完整流程，新接手的同事可按照"快速开始"部分立即运行，根据"参数总览"调整配置，利用"排障手册"解决问题，并通过"Guard门槛"确保导出安全性。所有核心算法改动已标注[NEW]/[CHG]，便于理解新版本的技术优势与向后兼容性考量。