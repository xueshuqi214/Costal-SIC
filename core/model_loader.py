"""模型加载与元数据核实。

两个模型文件的结构不同，这一点必须严格区分：

- Landsat：joblib 里直接就是一个 RandomForestRegressor 实例，没有 scaler。
- Sentinel：joblib 里是 dict，含 model / scaler / features / method / random_state。

加载时把"模型真实声明的特征"与"配置文件声明的特征"做交叉核验，
任何不一致都直接报错，绝不在错误状态下继续预测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from .errors import ModelIncompatibleError, ModelLoadError
from .paths import Config, resolve_resource

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """一次加载完成的模型及其全部必要元数据。"""

    dataset_key: str
    display_name: str
    algorithm: str
    estimator: Any
    features: List[str]
    n_features: int
    scaler: Optional[Any]
    requires_scaler: bool
    random_state: Optional[int]
    model_path: str
    warnings: List[str] = field(default_factory=list)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """按模型要求执行标准化后预测。"""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"预测输入必须是二维数组，实际为 {x.ndim} 维")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"预测输入有 {x.shape[1]} 列，但模型需要 {self.n_features} 个特征"
            )

        if self.requires_scaler:
            if self.scaler is None:
                raise ModelIncompatibleError(
                    f"{self.display_name} 模型需要标准化，但模型包中未找到 StandardScaler。"
                )
            x = self.scaler.transform(x)

        return np.asarray(self.estimator.predict(x))


def _get_n_features(estimator: Any) -> Optional[int]:
    n = getattr(estimator, "n_features_in_", None)
    return int(n) if n is not None else None


def _unwrap_model_package(package: Any) -> tuple[Any, Optional[Any], Optional[List[str]], Dict[str, Any]]:
    """兼容两种保存格式，返回 (estimator, scaler, features, extra)。"""
    if isinstance(package, dict):
        if "model" not in package:
            raise ModelLoadError(
                "模型包是字典格式但缺少 'model' 键，无法识别。",
                detail=f"实际键：{sorted(package.keys())}",
            )
        estimator = package["model"]
        scaler = package.get("scaler")
        features = package.get("features")
        extra = {k: v for k, v in package.items() if k not in ("model", "scaler", "features")}
        return estimator, scaler, list(features) if features else None, extra

    return package, None, None, {}


def load_model(
    dataset_key: str,
    config: Config,
    model_path: Optional[str] = None,
) -> LoadedModel:
    """加载指定数据类型的模型，并核验其与配置是否一致。"""
    ds = config.dataset(dataset_key)
    display = ds.get("display_name", dataset_key)

    rel_path = model_path or ds["model_file"]
    path = resolve_resource(rel_path)

    if not path.exists():
        raise ModelLoadError(
            f"找不到 {display} 模型文件：{path}",
            hint="请确认 models 目录中的模型文件是否存在，且未被重命名。",
            context={"model_file": str(path)},
        )

    try:
        package = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            f"{display} 模型文件无法加载：{path.name}",
            detail=f"{type(exc).__name__}: {exc}",
            hint="模型文件可能损坏，或与当前 scikit-learn / xgboost 版本不兼容。",
            context={"model_file": str(path)},
        ) from exc

    estimator, scaler, stored_features, extra = _unwrap_model_package(package)

    config_features: List[str] = list(ds["features"])
    requires_scaler: bool = bool(ds.get("requires_scaler", False))
    warnings: List[str] = []

    # ---- 特征列表核验 -------------------------------------------------
    if stored_features is not None:
        if list(stored_features) != config_features:
            raise ModelIncompatibleError(
                f"{display} 模型文件内部保存的特征顺序与配置文件不一致，已停止预测。",
                detail=(
                    "模型文件 features: " + ", ".join(stored_features) + "\n"
                    "配置文件 features: " + ", ".join(config_features)
                ),
                hint="请以模型文件为准修正 config/model_config.json，或使用匹配的模型文件。",
                context={
                    "stored_features": list(stored_features),
                    "config_features": config_features,
                },
            )
    else:
        warnings.append(
            f"{display} 模型文件未内嵌特征名，已采用配置文件中的特征顺序（"
            + ", ".join(config_features)
            + "）。"
        )

    # ---- 特征数量核验 -------------------------------------------------
    n_features = len(config_features)
    declared_n = _get_n_features(estimator)
    if declared_n is not None and declared_n != n_features:
        raise ModelIncompatibleError(
            f"{display} 模型声明需要 {declared_n} 个输入特征，但配置中列出了 {n_features} 个。",
            hint="请核对配置文件中的 features 列表是否与训练时一致。",
            context={"declared": declared_n, "configured": n_features},
        )

    # ---- scaler 核验 ---------------------------------------------------
    if requires_scaler:
        if scaler is None:
            raise ModelIncompatibleError(
                f"{display} 模型要求标准化，但模型文件中没有 StandardScaler。",
                hint="请使用训练脚本导出的完整模型包（含 model 与 scaler）。",
            )
        scaler_n = getattr(scaler, "n_features_in_", None)
        if scaler_n is not None and int(scaler_n) != n_features:
            raise ModelIncompatibleError(
                f"{display} 的 StandardScaler 期望 {scaler_n} 个特征，但配置为 {n_features} 个。"
            )
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        if mean is None or scale is None:
            raise ModelIncompatibleError(
                f"{display} 的 StandardScaler 尚未拟合（缺少 mean_ / scale_）。"
            )
        mean_arr = np.asarray(mean, dtype=float)
        scale_arr = np.asarray(scale, dtype=float)
        if not np.all(np.isfinite(mean_arr)) or not np.all(np.isfinite(scale_arr)):
            raise ModelIncompatibleError(
                f"{display} 的 StandardScaler 参数包含非有限值，模型文件可能损坏。"
            )
        if np.any(scale_arr == 0):
            raise ModelIncompatibleError(
                f"{display} 的 StandardScaler 存在标准差为 0 的特征，无法标准化。"
            )
    elif scaler is not None:
        warnings.append(
            f"{display} 配置为不需要标准化，但模型包中带有 StandardScaler，已忽略。"
        )

    # ---- 随机种子核验：仅作提示，不阻断 --------------------------------
    cfg_seed = ds.get("random_state")
    stored_seed = extra.get("random_state")
    if stored_seed is None:
        stored_seed = getattr(estimator, "random_state", None)
    if cfg_seed is not None and stored_seed is not None and int(cfg_seed) != int(stored_seed):
        warnings.append(
            f"{display} 配置记录的随机种子为 {cfg_seed}，模型实际为 {stored_seed}。"
            " 这不影响预测，但元数据存在偏差。"
        )

    algorithm = type(estimator).__name__

    logger.info(
        "已加载模型 dataset=%s algorithm=%s features=%d scaler=%s path=%s",
        dataset_key,
        algorithm,
        n_features,
        type(scaler).__name__ if scaler is not None else "None",
        path,
    )

    return LoadedModel(
        dataset_key=dataset_key,
        display_name=display,
        algorithm=algorithm,
        estimator=estimator,
        features=config_features,
        n_features=n_features,
        scaler=scaler,
        requires_scaler=requires_scaler,
        random_state=int(stored_seed) if stored_seed is not None else None,
        model_path=str(path),
        warnings=warnings,
    )


def inspect_model_file(path: str) -> Dict[str, Any]:
    """诊断用：不依赖配置，直接读取模型文件内部信息。"""
    p = resolve_resource(path)
    package = joblib.load(p)
    estimator, scaler, features, extra = _unwrap_model_package(package)
    info: Dict[str, Any] = {
        "path": str(p),
        "package_type": type(package).__name__,
        "estimator": type(estimator).__name__,
        "n_features_in": _get_n_features(estimator),
        "features": features,
        "extra": {k: str(v) for k, v in extra.items()},
    }
    if scaler is not None:
        info["scaler"] = {
            "type": type(scaler).__name__,
            "mean": np.asarray(getattr(scaler, "mean_", [])).round(6).tolist(),
            "scale": np.asarray(getattr(scaler, "scale_", [])).round(6).tolist(),
        }
    return info
