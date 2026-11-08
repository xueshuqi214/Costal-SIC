"""应用级常量与路径解析。

所有对外路径都以"应用根目录"为基准，绝不出现开发者个人电脑的绝对路径。
打包成 EXE 后，资源文件通过 sys._MEIPASS 定位；模型文件放在可写的外部目录，
以便在不重新打包的情况下替换模型。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


APP_NAME = "Coastal SIC"
APP_DIR_NAME = "Coastal_SIC"

# 配置文件相对路径（相对于包根目录）
CONFIG_RELATIVE = Path("config") / "model_config.json"


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包后的环境中。"""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """只读资源根目录。

    - 源码运行：项目根目录
    - 打包运行：PyInstaller 解包目录（sys._MEIPASS）
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    """可写资源根目录（models/、output/、logs/ 位于此处）。

    - 源码运行：项目根目录
    - 打包运行：EXE 所在目录，便于用户替换模型
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    """用户级数据目录（日志、最近使用记录）。"""
    if sys.platform == "win32":
        base = Path.home() / "AppData" / "Local"
    else:
        base = Path.home() / ".local" / "share"
    d = base / APP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_output_dir() -> Path:
    d = runtime_root() / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def config_path() -> Path:
    return _first_existing(
        runtime_root() / CONFIG_RELATIVE,
        bundle_root() / CONFIG_RELATIVE,
    )


def resolve_resource(relative: str) -> Path:
    """解析资源相对路径，优先可写目录，其次打包目录。"""
    rel = Path(relative)
    if rel.is_absolute():
        return rel
    return _first_existing(
        runtime_root() / rel,
        bundle_root() / rel,
    )


class Config:
    """model_config.json 的只读访问封装。"""

    def __init__(self, data: Dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        p = path or config_path()
        if not p.exists():
            raise FileNotFoundError(f"找不到配置文件：{p}")
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data, p)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def app(self) -> Dict[str, Any]:
        return self._data.get("application", {})

    @property
    def raster_io(self) -> Dict[str, Any]:
        return self._data.get("raster_io", {})

    @property
    def validation(self) -> Dict[str, Any]:
        return self._data.get("validation", {})

    @property
    def datasets(self) -> Dict[str, Dict[str, Any]]:
        return self._data.get("datasets", {})

    def dataset(self, key: str) -> Dict[str, Any]:
        datasets = self.datasets
        if key not in datasets:
            raise KeyError(
                f"未知的数据类型 '{key}'，可选值：{', '.join(datasets)}"
            )
        return datasets[key]

    def dataset_names(self) -> list[str]:
        return list(self.datasets.keys())
