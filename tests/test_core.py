"""核心引擎单元测试。

运行方式（项目根目录）：

    python -m unittest discover -s Coastal_SIC/tests -t . -v

或用 pytest：

    python -m pytest Coastal_SIC/tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Coastal_SIC.core.errors import ModelIncompatibleError, RasterValidationError
from Coastal_SIC.core.feature_validator import resolve_band_mapping, validate_raster
from Coastal_SIC.core.model_loader import load_model
from Coastal_SIC.core.paths import Config
from Coastal_SIC.core.raster_exporter import (
    assert_not_overwriting_input,
    resolve_output_path,
    sanitize_filename,
)


class TestConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.load()

    def test_two_datasets_present(self) -> None:
        names = self.config.dataset_names()
        self.assertIn("landsat", names)
        self.assertIn("sentinel", names)

    def test_landsat_features(self) -> None:
        ds = self.config.dataset("landsat")
        self.assertEqual(
            ds["features"],
            ["Lon", "SI", "DI", "NDVI", "NIR", "Lat", "NDMI", "CCI", "Green", "SWIR2"],
        )
        self.assertFalse(ds["requires_scaler"])
        self.assertEqual(ds["random_state"], 1711)

    def test_sentinel_features(self) -> None:
        ds = self.config.dataset("sentinel")
        self.assertEqual(
            ds["features"],
            ["VV", "B7", "B6", "CCI", "Lon", "VH", "Lat", "B12", "B11", "B4"],
        )
        self.assertTrue(ds["requires_scaler"])
        self.assertEqual(ds["random_state"], 1684)

    def test_unknown_dataset_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.config.dataset("unknown")


class TestModelLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.load()

    def test_landsat_model(self) -> None:
        m = load_model("landsat", self.config)
        self.assertEqual(m.n_features, 10)
        self.assertFalse(m.requires_scaler)
        self.assertIsNone(m.scaler)
        self.assertEqual(m.random_state, 1711)
        self.assertEqual(type(m.estimator).__name__, "RandomForestRegressor")

    def test_sentinel_model(self) -> None:
        m = load_model("sentinel", self.config)
        self.assertEqual(m.n_features, 10)
        self.assertTrue(m.requires_scaler)
        self.assertIsNotNone(m.scaler)
        self.assertEqual(type(m.estimator).__name__, "XGBRegressor")
        # 关键：标准化器必须已拟合，且参数完整
        self.assertTrue(hasattr(m.scaler, "mean_"))
        self.assertEqual(len(m.scaler.mean_), 10)

    def test_sentinel_predict_scales_input(self) -> None:
        """验证 predict() 内部确实调用了 scaler.transform。"""
        m = load_model("sentinel", self.config)
        x = np.zeros((3, 10), dtype=np.float32)
        y = m.predict(x)
        self.assertEqual(y.shape, (3,))

        # 手工复算
        expected = m.estimator.predict(m.scaler.transform(x))
        np.testing.assert_allclose(y, expected, rtol=0, atol=0)

    def test_feature_mismatch_detected(self) -> None:
        """当配置被篡改、与模型内嵌特征不符时必须报错。"""
        cfg = Config.load()
        cfg._data["datasets"]["sentinel"]["features"] = list(
            reversed(cfg.dataset("sentinel")["features"])
        )
        with self.assertRaises(ModelIncompatibleError):
            load_model("sentinel", cfg)


class TestBandMapping(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.load()
        self.ds = self.config.dataset("sentinel")

    def test_exact_name_match(self) -> None:
        names = ["VV", "B7", "B6", "CCI", "Lon", "VH", "Lat", "B12", "B11", "B4"]
        mapping, issues = resolve_band_mapping(
            names, self.ds["features"], self.ds, strict=True
        )
        self.assertEqual(len(issues), 0)
        self.assertTrue(all(m is not None for m in mapping))
        self.assertEqual([m.band_index for m in mapping], list(range(10)))

    def test_reordered_bands_mapped_by_name(self) -> None:
        """波段顺序打乱时，仍应按名称正确映射。"""
        names = ["B4", "B11", "B12", "Lat", "VH", "Lon", "CCI", "B6", "B7", "VV"]
        mapping, issues = resolve_band_mapping(
            names, self.ds["features"], self.ds, strict=True
        )
        self.assertEqual(len(issues), 0)
        by_feature = {m.feature: m.band_index for m in mapping if m}
        self.assertEqual(by_feature["VV"], 9)
        self.assertEqual(by_feature["B4"], 0)
        self.assertEqual(by_feature["B6"], 7)

    def test_missing_band_is_error(self) -> None:
        names = ["VV", "B7", "CCI", "Lon", "VH", "Lat", "B12", "B11", "B4", "B8"]
        mapping, issues = resolve_band_mapping(
            names, self.ds["features"], self.ds, strict=True
        )
        self.assertTrue(any(i.code == "MISSING_FEATURES" for i in issues))
        missing = [f for f, m in zip(self.ds["features"], mapping) if m is None]
        self.assertEqual(missing, ["B6"])

    def test_no_positional_fallback_in_strict_mode(self) -> None:
        names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"]
        mapping, issues = resolve_band_mapping(
            names, self.ds["features"], self.ds, strict=True
        )
        self.assertTrue(all(m is None for m in mapping))
        self.assertTrue(any(i.severity == "error" for i in issues))


class TestValidationError(unittest.TestCase):
    def test_missing_features_blocks_prediction(self) -> None:
        """真实 Sentinel 影像缺 B6/B12，必须被拦截。"""
        config = Config.load()
        path = Path(r"E:\NewGame\训练中心\2实际训练合集\data\JZ_SIC_Features_2020.tif")
        if not path.exists():
            self.skipTest("测试影像不存在")
        result = validate_raster(str(path), "sentinel", config, sample_pixels=False)
        self.assertFalse(result.ok)
        messages = " ".join(i.message for i in result.errors)
        self.assertIn("B6", messages)
        self.assertIn("B12", messages)
        with self.assertRaises(RasterValidationError):
            result.raise_if_invalid()


class TestExportPaths(unittest.TestCase):
    def test_sanitize(self) -> None:
        self.assertEqual(sanitize_filename('a<b>c:d'), "a_b_c_d")
        self.assertEqual(sanitize_filename("   "), "SIC_prediction")
        self.assertEqual(sanitize_filename("CON"), "_CON")

    def test_resolve_output_path_adds_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = resolve_output_path(tmp, "result", allow_overwrite=True)
            self.assertEqual(p.suffix.lower(), ".tif")
            self.assertEqual(p.parent, Path(tmp))

    def test_existing_file_blocked(self) -> None:
        from Coastal_SIC.core.errors import ExportError

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.tif"
            target.write_text("x")
            with self.assertRaises(ExportError):
                resolve_output_path(tmp, "x")

    def test_overwriting_input_blocked(self) -> None:
        from Coastal_SIC.core.errors import ExportError

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "in.tif"
            f.write_text("x")
            with self.assertRaises(ExportError):
                assert_not_overwriting_input(f, str(f))


class TestModelCache(unittest.TestCase):
    def test_service_caches_model(self) -> None:
        from Coastal_SIC.core.service import SICService

        svc = SICService()
        a = svc.get_model("landsat")
        b = svc.get_model("landsat")
        self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
