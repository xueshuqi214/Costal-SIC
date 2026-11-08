# Costal-SIC
基于已有机器学习模型封装的 Windows 桌面预测应用。用户导入多波段遥感特征影像，选择 Landsat 或 Sentinel 数据类型，软件自动调用对应模型完成土壤无机碳（SIC）逐像元预测，并导出带完整空间参考的 GeoTIFF。
=======
# Coastal SIC — 滨海湿地土壤无机碳遥感估算系统

基于已有机器学习模型封装的 Windows 桌面预测应用。用户导入多波段遥感特征影像，
选择 Landsat 或 Sentinel 数据类型，软件自动调用对应模型完成土壤无机碳（SIC）逐像元
预测，并导出带完整空间参考的 GeoTIFF。

本软件**不训练模型**、**不修改模型参数**，只做推理与结果导出。

---

## 一、快速开始

### 1. 环境要求

- Windows 10 / 11 x64
- Python 3.10（推荐 3.10.2，与模型训练环境一致）

### 2. 安装依赖

```bat
cd E:\NewGame\软著所需代码\Coastal_SIC
python -m pip install -r requirements.txt
```

> **重要**：`scikit-learn==1.6.1` 与 `xgboost==3.0.0` 是模型文件导出时使用的版本。
> 升级主版本可能导致模型反序列化失败或预测数值改变，请勿随意升级。

### 3. 启动软件

```bat
cd E:\NewGame\软著所需代码\Coastal_SIC
python main.py
```

### 4. 命令行（用于验证与批量处理）

在 `Coastal_SIC` 目录下：

```bat
cd E:\NewGame\软著所需代码\Coastal_SIC
python cli.py inspect  <影像路径>
python cli.py validate landsat <影像路径>
python cli.py predict  landsat <影像路径> <输出目录> --name 结果名
python cli.py compare  <结果A.tif> <结果B.tif>
python cli.py stats    <栅格路径>
```

或者在上级目录用模块方式：

```bat
cd E:\NewGame\软著所需代码
python -m Coastal_SIC.cli inspect <影像路径>
```

---

## 二、模型事实核验结果

开发前对提供的模型文件做了逐项反查，以下是**以模型文件为准**的结论。

### Landsat 模块

| 项目 | 结论 |
|---|---|
| 模型类型 | `RandomForestRegressor` |
| 模型文件结构 | joblib 直接存 estimator，**无 scaler** |
| `n_features_in_` | 10 |
| `random_state` | 1711 |
| 特征顺序 | `Lon, SI, DI, NDVI, NIR, Lat, NDMI, CCI, Green, SWIR2` |

特征顺序已由三方面交叉确认：训练脚本 `Landsat-RF.py` 的 `FEATURE_SETS`、
原始预测脚本 `Landsat含量估算.py` 的 `EXPECTED_FEATURES`、
以及输入影像 `JZ_SIC_Landsat_Features_*.tif` 的 10 个波段描述。三者完全一致。

### Sentinel 模块

| 项目 | 结论 |
|---|---|
| 模型类型 | `XGBRegressor` |
| 模型文件结构 | joblib 存 dict：`model` / `scaler` / `features` / `method` / `random_state` |
| `n_features_in_` | 10 |
| `random_state` | 1684 |
| 特征顺序（模型内嵌） | `VV, B7, B6, CCI, Lon, VH, Lat, B12, B11, B4` |
| 标准化 | **必须**使用随模型保存的 `StandardScaler`，禁止重新拟合 |

`scaler.mean_` 与 `scaler.scale_` 均已核实为已拟合的 10 维参数。

---

## 三、必须了解的数据问题

### Sentinel 输入影像与模型特征不匹配

现有 Sentinel 特征影像 `JZ_SIC_Features_2020.tif`（及 2024 版）的波段描述为：

```
VH, B7, Lon, Lat, CCI, VV, B5, B4, B11, B8
```

而模型需要：

```
VV, B7, B6, CCI, Lon, VH, Lat, B12, B11, B4
```

差异有三类：

1. **顺序不同**（`VV` 在第 6 位而非第 1 位，`Lon` 在第 3 位而非第 5 位……）
2. **缺少 B6 与 B12**（影像里根本没有这两个波段）
3. **多了 B5 与 B8**

原始预测脚本 `Sentinels含量估算.py` 是**按波段位置盲读**的（`src.read()` 后直接
`reshape` 成特征矩阵），并没有按名称重排。因此它实际喂给模型的是：
第 1 波段 `VH` 被当成 `VV`、第 3 波段 `Lon` 被当成 `B6`……

用原始逻辑复算 2020 年，结果与历史成果 `JZ_SIC_2020_New.tif` **逐像元完全一致**
（1,114,073 个有效像元，max diff = 0.0）。这意味着**历史成果本身可能就是这批
错配输入的产物**，其 SIC 数值的物理意义值得重新审视。

### 本软件的处理策略

按确认的方案，软件采用**严格按波段名匹配**：

- 读取每个波段描述，按模型特征名重排后输入模型
- 缺失任何特征（如 B6、B12）→ 明确报错并列出缺失项，**拒绝预测**
- 不提供位置兜底，避免在用户不知情时产出错误结果

因此当前这份 2020/2024 影像会被软件拦下。要跑通 Sentinel，需要重新从 GEE
导出包含 `B6` 与 `B12` 的影像，并为每个波段写入正确的波段描述。

可参考 `tests/make_sentinel_test_raster.py` 的做法，用现有数据拼一份波段名
正确的测试影像来验证流程（B6 用 B5、B12 用 B11 代填，**仅用于流程验证，
不可用于科研**）。

---

## 四、验证结果

### Landsat 逐像元一致性

| 项目 | 结果 |
|---|---|
| 输入 | `JZ_SIC_Landsat_Features_1985.tif` |
| 历史成果 | `results/JZ_SIC_1985.tif` |
| 有效像元 | 496,098（两边一致） |
| 掩膜差异 | 0 |
| 最大绝对差 | **0.0** |
| 结论 | **逐像元完全相同** |

### Sentinel 流程正确性

用波段名正确的测试影像，与一份**完全独立实现**的预测流程（不复用本软件任何代码，
直接按特征名重排 + `scaler.transform` + `model.predict`）对比：

| 对照精度 | 数值不同像元 | 最大绝对差 | 结论 |
|---|---|---|---|
| float32（与原始脚本一致） | 0 | 0.0 | **逐像元完全相同** |
| float64 | 2,861 (0.26%) | 0.32 | 精度敏感参考 |

float64 下的少量差异来自 XGBoost 决策树的分裂阈值：输入相差约 1e-7 时，
约 0.26% 的像元会落在阈值另一侧，产生约 0.32 的离散跳变。这是树模型的固有特性，
原始脚本同样使用 float32，因此软件与原始脚本位级一致。

### 单元测试

```bat
cd E:\NewGame\软著所需代码
python -m unittest Coastal_SIC.tests.test_core -v
```

18 项全部通过，覆盖：配置正确性、模型加载与特征核验、标准化强制校验、
波段名映射（含乱序与缺失）、导出路径安全检查。

---

## 五、目录结构

```
Coastal_SIC/
├── main.py                  # 程序入口
├── cli.py                   # 命令行工具
├── requirements.txt
├── build_exe.spec           # PyInstaller 打包配置
├── README.md
│
├── models/                  # 模型文件（可替换）
│   ├── landsat_rf.joblib
│   └── sentinel_xgb.joblib
│
├── config/
│   └── model_config.json    # 数据集定义、特征顺序、校验策略
│
├── core/                    # 核心引擎（不依赖 GUI）
│   ├── paths.py             # 路径解析（源码/打包双模式）
│   ├── errors.py            # 异常与中文提示
│   ├── model_loader.py      # 模型加载 + 元数据核验
│   ├── raster_reader.py     # 栅格元信息 + 分块窗口
│   ├── feature_validator.py # 输入影像校验 + 波段映射
│   ├── raster_predictor.py  # 分块预测引擎
│   ├── raster_exporter.py   # 输出路径与空间参考校验
│   ├── statistics.py        # 统计量与预览、栅格对比
│   └── service.py           # 流程编排（GUI 与 CLI 共用）
│
├── gui/
│   ├── main_window.py       # 主窗口 + 侧边导航
│   ├── home_page.py         # 首页
│   ├── prediction_page.py   # 预测页
│   ├── result_page.py       # 结果页
│   ├── widgets.py           # 通用组件与配色
│   └── workers.py           # 后台线程
│
├── resources/
│   └── styles/app.qss       # 界面样式
│
├── tests/
│   ├── test_core.py                     # 单元测试
│   ├── make_sentinel_test_raster.py     # 构造 Sentinel 测试影像
│   ├── verify_sentinel_engine.py        # Sentinel 独立实现对照
│   └── diagnose_precision.py            # 精度敏感性诊断
│
└── output/                  # 默认输出目录
```

---

## 六、输入数据要求

| 项目 | 要求 |
|---|---|
| 格式 | GeoTIFF（`.tif` / `.tiff`） |
| 波段数 | Landsat 10 个 / Sentinel 10 个 |
| 波段描述 | **必须**为每个波段写入对应特征名（这是匹配依据） |
| 数据类型 | float32 / float64 最佳；整数类型会转换并给出警告 |
| 坐标系 | EPSG:4326（模型中的 Lon/Lat 以度为单位） |
| NoData | 建议显式设置；未设置时以 NaN/Inf 判定 |

### 为波段写入描述名的方法

**使用 GDAL 命令行：**

```bat
gdal_edit.py -setbanddesc 1 Lon -setbanddesc 2 SI ... input.tif
```

**使用 Python：**

```python
import rasterio
with rasterio.open("input.tif", "r+") as ds:
    for i, name in enumerate(["Lon","SI","DI","NDVI","NIR","Lat","NDMI","CCI","Green","SWIR2"], 1):
        ds.set_band_description(i, name)
```

---

## 七、界面说明

| 页面 | 功能 |
|---|---|
| 首页 | 软件介绍、两个数据类型的模型概览、工作流程、快速入口 |
| SIC 预测 | 选择数据类型与影像 → 自动校验 → 设置输出 → 执行 → 实时日志与进度 |
| 预测结果 | 最小值/最大值/平均值/标准差/有效像元/覆盖率、空间分布图、直方图、导出 |

预测页的校验结果以表格形式列出 11 项检查（尺寸、波段数、波段名、CRS、分辨率、
范围、NoData、类型、压缩、内存、有效像元占比），错误标红、警告标黄，
未通过时「开始预测」按钮保持禁用。

---

## 八、打包为 EXE

```bat
cd E:\NewGame\软著所需代码\Coastal_SIC
pyinstaller build_exe.spec --noconfirm
```

产物：`dist/Coastal_SIC/Coastal_SIC.exe`

打包后的目录结构：

```
dist/Coastal_SIC/
├── Coastal_SIC.exe
├── models/                # 可直接替换模型
├── config/
├── resources/
├── output/
└── _internal/             # Python 运行时与依赖
```

若 `models/` 与 `config/` 只在 `_internal/` 里，把它们复制到 EXE 同级即可
（程序的查找顺序是：EXE 同级 → `_internal`）：

```bat
cd dist\Coastal_SIC
xcopy /E /I /Y "_internal\models" "models"
xcopy /E /I /Y "_internal\config" "config"
xcopy /E /I /Y "_internal\resources" "resources"
mkdir output
```

### 打包后自检

EXE 内置自检开关，用于确认模型与依赖是否完整（不打开界面）：

```bat
Coastal_SIC.exe --selftest
```

正常输出应包含两个模型的加载结果、rasterio/GDAL 版本、sklearn 与 xgboost 版本，
最后一行是 `=== 自检通过 ===`。

**关键点**：`models/` 与 `config/` 位于 EXE 同级，用户无需重新打包即可替换模型
或调整配置。日志写入 `%LOCALAPPDATA%\Coastal_SIC\logs\coastal_sic.log`。

### 打包体积说明

实测约 426 MB。主要来自 scipy / sklearn / xgboost / matplotlib / PySide6 与 GDAL。
`build_exe.spec` 已排除 torch、tensorflow、pandas、tkinter、QtWebEngine 等
无关重型依赖；若首次打包耗时异常长，通常是因为 `collect_submodules` 把
torch/tensorflow 一并扫入，请勿改回该写法。

---

## 九、稳定性设计要点

原始预测脚本存在若干会影响软件稳定性的问题，本实现已逐项处理：

| 原始脚本问题 | 本实现处理 |
|---|---|
| 按位置盲读波段，不校验名称 | 按波段描述名严格匹配，缺失即报错 |
| 只检查波段总数 | 同时校验数量、名称、顺序、CRS、分辨率、范围、dtype、有效像元占比 |
| 直接 `raise ValueError`（英文） | 中文错误提示 + 修复建议 + 问题清单 |
| 硬编码个人电脑绝对路径 | 全部改为相对路径，源码/打包双模式解析 |
| `src.block_windows(1)` 依赖文件分块 | 自建窗口迭代器，对非 tiled 文件同样有效 |
| `data.reshape(n_bands, -1).T` 全块读入 | 只读取模型需要的波段，减少 I/O；块尺寸按内存预算计算 |
| 单块异常导致整幅失败 | 单块失败标记 NoData 并记录，其余块继续 |
| 无进度、无日志 | 分块进度、阶段状态、分级日志、可取消 |
| 输出直接写目标路径 | 先写 `.part` 再原子替换，避免中断产生损坏文件 |
| 不校验输出空间参考 | 写出后回读比对 CRS / transform / 尺寸 |
| 无统计输出 | 流式累积 min/max/mean/std/直方图，无需二次全图读取 |

---

## 十、常见问题

**Q：软件提示"影像缺少模型必需的波段：B6, B12"**
A：这正是严格校验在起作用。你的影像波段描述里没有 B6/B12，请重新导出包含
这两个波段的影像，或用 `set_band_description` 为已有波段写入正确名称。

**Q：预测结果大面积为 NoData**
A：检查影像是否全为 NaN/Inf、NoData 值是否设置正确，以及有效像元占比。
校验页会给出抽样估计的有效像元比例。

**Q：结果数值与历史成果不一致**
A：若是 Landsat，本软件与历史成果逐像元一致；若不一致，通常是波段顺序问题——
请确认波段描述与论文中使用的顺序一致。若是 Sentinel，见第三节说明。

**Q：加载模型报错**
A：多为 scikit-learn / xgboost 版本不匹配。请按 `requirements.txt` 的固定版本
安装，尤其是 `scikit-learn==1.6.1` 和 `xgboost==3.0.0`。

---

## 十一、开发验证清单

- [x] 核实 Landsat 模型真实特征与顺序（三方交叉验证）
- [x] 核实 Sentinel 模型特征、scaler、随机种子
- [x] 确认模型文件加载方式（两种结构分别处理）
- [x] 确认 Sentinel 标准化调用方式（训练期 scaler，不重新拟合）
- [x] 审查原始预测脚本的栅格处理逻辑
- [x] 找出并修复影响稳定性的问题（见第九节）
- [x] 核心预测引擎完整实现（Landsat + Sentinel）
- [x] Landsat 与历史成果逐像元一致（max diff = 0.0）
- [x] Sentinel 与独立实现逐像元一致（float32 路径）
- [x] 18 项单元测试通过
- [x] PySide6 桌面界面
- [x] PyInstaller 打包配置
>>>>>>> master
