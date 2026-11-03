"""Cache location is optional and portable; no Kit, network or training starts."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/train_elf3_cached_visuals.py"


def load():
    spec = importlib.util.spec_from_file_location("portable_visual_cache", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CachedVisualPortabilityTests(unittest.TestCase):
    def test_default_uses_current_users_home(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "home", return_value=Path("/example/user")):
            module = load()
        self.assertEqual(module.ARROW, Path("/example/user/.cache/ov/client/http/4a8ab93074b320c51f69135679cb742b1c8d69fd1f1fb4b70816d14f2babe6ea.usd"))

    def test_explicit_cache_override_keeps_resource_identity(self):
        with patch.dict(os.environ, {"ELF3_ARROW_CACHE": "/example/cache/arrow_x.usd"}):
            module = load()
        self.assertEqual(module.ARROW, Path("/example/cache/arrow_x.usd"))
        self.assertEqual(module.ARROW_SHA256, "d321116a8fecfef657ac95a5392115b2bfbe7d5eeae2a74556a7037f15259132")
        self.assertTrue(module.URL_PATH.endswith("/UIElements/arrow_x.usd"))


if __name__ == "__main__":
    unittest.main()
