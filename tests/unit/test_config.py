from pathlib import Path
import unittest

from seismic_motion.config import ConfigError, config_sha256, load_config, validate_config


ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_baseline_config_loads_and_hashes_deterministically(self) -> None:
        config = load_config(ROOT / "configs" / "pc_baseline.yaml")
        self.assertEqual(config["tracker"]["window_len"], 16)
        self.assertEqual(config_sha256(config), config_sha256(dict(config)))

    def test_invalid_queue_bound_is_rejected(self) -> None:
        config = load_config(ROOT / "configs" / "pc_baseline.yaml")
        config["runtime"]["max_queue_blocks"] = 0
        with self.assertRaises(ConfigError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()

