"""Existing-run uploads preserve iteration history and translate elapsed time only."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("resume_upload", ROOT / "scripts/repair_swanlab_scalars.py")
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class ResumeIndexTests(unittest.TestCase):
    def test_iteration_indices_never_shift(self):
        for tag in ("Train/mean_reward", "Loss/value_function", "Feedback/reporting_error_rms"):
            self.assertEqual(upload.metric_index(tag, 42394, 53174), 42394)

    def test_only_explicit_time_axes_shift(self):
        for tag in upload.TIME_AXIS_TAGS:
            self.assertEqual(upload.metric_index(tag, 493, 53174), 53667)
            self.assertGreater(upload.metric_index(tag, 495, 53174), 53667)
        self.assertEqual(upload.metric_index("Other/time", 493, 53174), 493)

    def test_existing_overlap_is_skipped(self):
        cursor = 42393
        accepted = [step for step in range(42001, 42401)
                    if upload.metric_index("Train/mean_reward", step, 53174) > cursor]
        self.assertEqual(accepted, list(range(42394, 42401)))

    def test_default_mapping_is_unchanged(self):
        for tag in ("Train/mean_reward", *upload.TIME_AXIS_TAGS):
            self.assertEqual(upload.metric_index(tag, 17), 17)


if __name__ == "__main__":
    unittest.main()
