from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_bench.adapter import load_forward_module


class AdapterImportTest(unittest.TestCase):
    def test_forward_module_can_import_repository_shared_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "experiment_data"
            shared = package / "_shared"
            candidate = package / "candidate"
            shared.mkdir(parents=True)
            candidate.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (shared / "__init__.py").write_text("", encoding="utf-8")
            (shared / "settings.py").write_text("VALUE = 17\n", encoding="utf-8")
            forward = candidate / "forward.py"
            forward.write_text(
                "from experiment_data._shared.settings import VALUE\n",
                encoding="utf-8",
            )

            module = load_forward_module(forward)

        self.assertEqual(module.VALUE, 17)


if __name__ == "__main__":
    unittest.main()
