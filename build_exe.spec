# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（one-folder 模式）。

为什么用 one-folder 而不是 one-file：
  模型文件（.joblib）和 config 需要能被用户看到与替换，
  one-file 每次启动都会解包到临时目录，既慢又不利于替换模型。

打包命令（在 Coastal_SIC 目录下）：
    pyinstaller build_exe.spec --noconfirm

产物：
    dist/Coastal_SIC/Coastal_SIC.exe
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

PROJECT = Path(SPECPATH).resolve()  # noqa: F821

# ---- 需要随包分发的数据文件 ------------------------------------------
datas = [
    (str(PROJECT / "config" / "model_config.json"), "config"),
    (str(PROJECT / "resources" / "styles"), "resources/styles"),
]
# 模型放在 EXE 同级，便于用户替换
datas += [(str(p), "models") for p in (PROJECT / "models").glob("*.joblib")]

# rasterio / GDAL 的数据文件（proj 定义等），缺少会导致坐标系处理失败
try:
    datas += collect_data_files("rasterio")
except Exception:
    pass

# xgboost 依赖 native 库 lib/xgboost.dll，PyInstaller 不会自动收集，
# 缺少它时导入 XGBoost 会报 "Cannot find XGBoost Library"。
binaries = []
try:
    binaries += collect_dynamic_libs("xgboost")
except Exception:
    pass
try:
    datas += collect_data_files("xgboost")
except Exception:
    pass

# ---- 隐藏导入 --------------------------------------------------------
# 只列出真正需要的子模块。collect_submodules 会连带扫入 torch/tensorflow
# 等无关大包，使打包耗时极长、体积膨胀，因此不使用。
hiddenimports = [
    # sklearn：joblib 反序列化 RandomForest 需要的内部模块
    "sklearn.ensemble._forest",
    "sklearn.ensemble",
    "sklearn.tree._tree",
    "sklearn.tree",
    "sklearn.preprocessing._data",
    "sklearn.preprocessing",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._weight_vector",
    "sklearn.utils._array_api",
    "sklearn.utils._bunch",
    "sklearn.neighbors._partition_nodes",
    "scipy.sparse.csgraph._validation",
    # xgboost
    "xgboost",
    "xgboost.core",
    "xgboost.sklearn",
    # 其他运行期依赖
    "joblib",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
]

# rasterio 有大量 Cython 子模块（serde、_base、_io 等），必须全部收集，
# 否则打包后会报 ModuleNotFoundError: No module named 'rasterio.serde'。
# rasterio 本身不依赖 torch/tensorflow，因此全量收集是安全的。
try:
    hiddenimports += collect_submodules("rasterio")
except Exception:
    pass

# ---- 排除用不到的大体积模块 ------------------------------------------
excludes = [
    # GUI 中未使用
    "tkinter",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebChannel",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQml",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtBluetooth",
    "PySide6.QtPositioning",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvgWidgets",
    # 训练/分析阶段才用到的重型包，预测流程不需要
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "pandas",
    "pyarrow",
    "polars",
    "lxml",
    "IPython",
    "notebook",
    "jupyter",
    "zmq",
    "pyzmq",
    "Pythonwin",
    "win32com",
    "win32api",
    "pythoncom",
    "pywintypes",
    "cryptography",
    "OpenSSL",
    "pytest",
    "_pytest",
    "sphinx",
    "docutils",
    "PIL.ImageTk",
    "matplotlib.tests",
    "matplotlib.testing",
    "numpy.tests",
    "scipy.tests",
]

a = Analysis(  # noqa: F821
    [str(PROJECT / "main.py")],
    pathex=[str(PROJECT.parent)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Coastal_SIC",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Coastal_SIC",
)
