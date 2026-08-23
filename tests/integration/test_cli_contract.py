import subprocess
import sys
import unittest


class CliContractTests(unittest.TestCase):
    def test_required_entry_points_expose_help(self) -> None:
        modules = (
            "seismic_motion.cli.run",
            "seismic_motion.cli.realtime",
            "seismic_motion.cli.benchmark_synthetic",
            "seismic_motion.cli.evaluate_pga",
        )
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
