"""Regression checks for release validation; no training or inference."""
import copy
import importlib.util
import json
import hashlib
import shutil
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


class ScientificComparisonTests(unittest.TestCase):
    def setUp(self):
        self.expected = json.loads((ROOT / "expected_outputs/PRIMARY_EXTERNAL_RESULT.json").read_text())

    def test_generated_timestamp_and_endpoint_order_are_ignored(self):
        actual = copy.deepcopy(self.expected)
        actual["created_at_utc"] = "different generated timestamp"
        actual["secondary_family"].reverse()
        reproduce.compare_result(actual, self.expected)

    def test_every_scientific_leaf_is_checked(self):
        def leaves(value, prefix=()):
            if isinstance(value, dict):
                for key, child in value.items():
                    if prefix == () and key in {"created_at_utc", "patient_delta_sha256", "secondary_results_sha256"}:
                        continue
                    yield from leaves(child, prefix + (key,))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    yield from leaves(child, prefix + (index,))
            else:
                yield prefix, value
        for path, value in leaves(self.expected):
            with self.subTest(path=path):
                actual = copy.deepcopy(self.expected)
                target = actual
                for part in path[:-1]:
                    target = target[part]
                changed = (not value if isinstance(value, bool) else
                           value + 1 if isinstance(value, int) else
                           value * 1.1 + 1e-30 if isinstance(value, float) else
                           value + " changed")
                target[path[-1]] = changed
                with self.assertRaises(RuntimeError):
                    reproduce.compare_result(actual, self.expected)

    def test_missing_extra_duplicate_and_nonfinite_values_fail(self):
        mutations = []
        for mode in ["missing", "extra", "duplicate", "nan", "bool_as_int"]:
            actual = copy.deepcopy(self.expected)
            if mode == "missing": del actual["primary"]["sensitivity"]
            elif mode == "extra": actual["primary"]["extra"] = 1
            elif mode == "duplicate": actual["secondary_family"][1] = actual["secondary_family"][0]
            elif mode == "nan": actual["primary"]["mean_delta"] = float("nan")
            else: actual["primary"]["support_rule_met"] = 1
            mutations.append((mode, actual))
        for mode, actual in mutations:
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                reproduce.compare_result(actual, self.expected)

    def test_generated_statistics_hash_and_csv_semantics_are_both_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ["PRIMARY_EXTERNAL_RESULT.json", "PRIMARY_PATIENT_DELTAS.csv", "SECONDARY_FAMILY_RESULTS.csv"]:
                shutil.copy2(ROOT / "expected_outputs" / name, out / name)
            reproduce.compare_generated_statistics(out, self.expected)
            csv = out / "SECONDARY_FAMILY_RESULTS.csv"
            csv.write_text(csv.read_text().replace("0.029924750796828", "0.039924750796828"))
            # First reject byte corruption even when the JSON is untouched.
            with self.assertRaises(RuntimeError):
                reproduce.compare_generated_statistics(out, self.expected)
            # Updating the file hash cannot hide a changed scientific value.
            result = copy.deepcopy(self.expected)
            result["secondary_results_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
            (out / "PRIMARY_EXTERNAL_RESULT.json").write_text(json.dumps(result))
            with self.assertRaises(RuntimeError):
                reproduce.compare_generated_statistics(out, self.expected)

    def test_historical_hashes_pass_and_modified_artifacts_fail(self):
        reproduce.verify_inference_hashes(ROOT / "reproduction_inputs/statistics")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for filename in reproduce.EXPECTED_INFERENCE_HASHES:
                (out / filename).write_bytes(b"changed artifact")
            with self.assertRaises(RuntimeError):
                reproduce.verify_inference_hashes(out)


if __name__ == "__main__":
    unittest.main()
