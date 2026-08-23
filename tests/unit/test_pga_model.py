from pathlib import Path
import tempfile
import unittest

import numpy as np

from seismic_motion.pga.model import (
    EmpiricalPGAModel,
    evaluate_predictions,
    group_kfold_indices,
)


class PGAModelTests(unittest.TestCase):
    def test_group_split_has_no_leakage(self) -> None:
        groups = [f"event-{index // 3}" for index in range(30)]
        for train, test in group_kfold_indices(groups, folds=5):
            self.assertFalse(set(np.asarray(groups)[train]) & set(np.asarray(groups)[test]))

    def test_ridge_round_trip_and_scale_gate(self) -> None:
        rng = np.random.default_rng(9)
        features = rng.normal(size=(80, 3))
        truth = np.exp(2 + features @ np.asarray([0.4, -0.2, 0.1]))
        model = EmpiricalPGAModel(("a", "b", "c"), algorithm="ridge", alpha=1e-6)
        model.fit(features, truth)
        prediction = model.predict(features)
        self.assertLess(evaluate_predictions(truth, prediction)["log_pga_mae"], 1e-5)
        with self.assertRaises(RuntimeError):
            model.predict(features, scale_valid=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = EmpiricalPGAModel.load(path)
            np.testing.assert_allclose(restored.predict(features), prediction)

    def test_huber_downweights_outlier(self) -> None:
        x = np.linspace(0.1, 2.0, 60)[:, None]
        truth = np.exp(1 + 0.5 * x[:, 0])
        corrupted = truth.copy()
        corrupted[-1] *= 100
        ridge = EmpiricalPGAModel(("x",), algorithm="ridge", alpha=0).fit(x, corrupted)
        huber = EmpiricalPGAModel(("x",), algorithm="huber", alpha=0).fit(x, corrupted)
        ridge_error = evaluate_predictions(truth[:-1], ridge.predict(x[:-1]))["log_pga_mae"]
        huber_error = evaluate_predictions(truth[:-1], huber.predict(x[:-1]))["log_pga_mae"]
        self.assertLess(huber_error, ridge_error)


if __name__ == "__main__":
    unittest.main()

