import unittest

import numpy as np
import torch

from evaluation.fold_summary import summarize_folds
from evaluation.metrics import safe_ap, safe_auc
from training.losses import auc_pair_loss, focal_bce_with_mask
from training.objective import combine_training_losses


class LossMetricTests(unittest.TestCase):
    def test_auc_and_ap(self):
        y = [0, 0, 1, 1]
        p = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(safe_auc(y, p), 1.0)
        self.assertAlmostEqual(safe_ap(y, p), 1.0)

    def test_pair_loss_finite(self):
        logits = torch.tensor(
            [-1.0, -0.5, 0.5, 1.0]
        )
        labels = torch.tensor(
            [0.0, 0.0, 1.0, 1.0]
        )
        loss = auc_pair_loss(logits, labels)
        self.assertTrue(torch.isfinite(loss))

    def test_masked_focal(self):
        logits = torch.zeros(2, 3, 1)
        targets = torch.tensor(
            [
                [[0.0], [1.0], [0.0]],
                [[1.0], [0.0], [1.0]],
            ]
        )
        mask = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
        )
        loss = focal_bce_with_mask(
            logits, targets, mask
        )
        self.assertTrue(torch.isfinite(loss))

    def test_objective(self):
        x = torch.tensor(1.0)
        out = combine_training_losses(
            x, x, x, x
        )
        self.assertAlmostEqual(
            out.item(),
            1.0 + 0.75 + 0.03 + 0.10,
            places=6,
        )

    def test_fold_summary_ddof1(self):
        rows = [
            {"auc": 0.7},
            {"auc": 0.8},
            {"auc": 0.9},
        ]
        out = summarize_folds(rows, ["auc"])
        self.assertAlmostEqual(
            out["auc_mean"], 0.8
        )
        self.assertAlmostEqual(
            out["auc_sd"],
            float(np.std([0.7, 0.8, 0.9], ddof=1)),
        )


if __name__ == "__main__":
    unittest.main()
