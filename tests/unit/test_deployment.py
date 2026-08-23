import unittest

import torch

from seismic_motion.deployment.correlation import (
    correlation_batched_matmul,
    correlation_einsum,
)


class DeploymentRewriteTests(unittest.TestCase):
    def test_correlation_matmul_matches_einsum(self) -> None:
        generator = torch.Generator().manual_seed(5)
        features = torch.randn((2, 3, 4, 5, 6, 7), generator=generator)
        support = torch.randn((2, 4, 3, 3, 7), generator=generator)
        expected = correlation_einsum(features, support)
        actual = correlation_batched_matmul(features, support)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(float(torch.max(torch.abs(actual - expected))), 2e-6)


if __name__ == "__main__":
    unittest.main()

