"""异常类型与中文提示。

所有面向用户的错误都携带 code 与中文说明，GUI 层直接展示 message，
日志层可记录 detail 供排查。避免把英文堆栈直接抛给普通用户。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class CoastalSICError(Exception):
    """所有业务异常的基类。"""

    code = "UNKNOWN"
    default_title = "发生错误"

    def __init__(
        self,
        message: str,
        *,
        title: Optional[str] = None,
        detail: Optional[str] = None,
        hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.title = title or self.default_title
        self.detail = detail
        self.hint = hint
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "message": self.message,
            "detail": self.detail,
            "hint": self.hint,
            "context": self.context,
        }


class ConfigError(CoastalSICError):
    code = "CONFIG_ERROR"
    default_title = "配置错误"


class ModelLoadError(CoastalSICError):
    code = "MODEL_LOAD_ERROR"
    default_title = "模型加载失败"


class ModelIncompatibleError(CoastalSICError):
    code = "MODEL_INCOMPATIBLE"
    default_title = "模型与配置不一致"


class RasterReadError(CoastalSICError):
    code = "RASTER_READ_ERROR"
    default_title = "影像读取失败"


class RasterValidationError(CoastalSICError):
    """影像未通过输入校验。issues 保存全部问题，便于一次性告知用户。"""

    code = "RASTER_VALIDATION_ERROR"
    default_title = "输入影像校验未通过"

    def __init__(
        self,
        message: str,
        *,
        issues: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.issues = issues or []

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["issues"] = self.issues
        return d


class PredictionError(CoastalSICError):
    code = "PREDICTION_ERROR"
    default_title = "预测失败"


class ExportError(CoastalSICError):
    code = "EXPORT_ERROR"
    default_title = "结果导出失败"


class CancelledByUser(CoastalSICError):
    code = "CANCELLED"
    default_title = "已取消"

    def __init__(self, message: str = "操作已被用户取消。") -> None:
        super().__init__(message)
