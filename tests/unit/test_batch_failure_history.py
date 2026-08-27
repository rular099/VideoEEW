import csv
from pathlib import Path
import tempfile
import unittest

from scripts.run_pga_feature_batch import (
    _migrate_failure_history,
    _read_failure_rows,
    _write_failure_rows,
)


class BatchFailureHistoryTests(unittest.TestCase):
    def test_legacy_failure_is_preserved_when_current_table_is_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "failed_records.csv"
            history = root / "failure_history.csv"
            with current.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("record_id", "status", "exception_type", "reason"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "record_id": "48",
                        "status": "FAILED",
                        "exception_type": "ModuleNotFoundError",
                        "reason": "imageio",
                    }
                )

            migrated = _migrate_failure_history(current, history)
            _write_failure_rows(current, [])
            self.assertEqual(len(migrated), 1)
            self.assertEqual(_read_failure_rows(current), [])
            preserved = _read_failure_rows(history)
            self.assertEqual(preserved[0]["record_id"], "48")
            self.assertEqual(
                preserved[0]["attempt_utc"], "UNKNOWN_PREVIOUS_ATTEMPT"
            )


if __name__ == "__main__":
    unittest.main()
