import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manifest", ROOT / "tools/build_release_manifest.py")
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


class ReleaseCoverageTests(unittest.TestCase):
    def test_generate_check_and_detect_changed_missing_and_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "README.md"
            file.write_text("release")
            manifest.process(root)
            manifest.process(root, check=True)
            file.write_text("modified")
            with self.assertRaises(RuntimeError): manifest.process(root, check=True)
            file.write_text("release")
            file.unlink()
            with self.assertRaises(RuntimeError): manifest.process(root, check=True)
            file.write_text("release")
            extra = root / "new-input.csv"
            extra.write_text("added")
            with self.assertRaises(RuntimeError): manifest.process(root, check=True)


if __name__ == "__main__":
    unittest.main()
