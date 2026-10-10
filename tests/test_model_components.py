import unittest

import torch

from model.cross_attention import CrossBlock
from model.pooling import masked_max, masked_mean
from model.rna_graph import RNA2DGraphEncoder


class ComponentTests(unittest.TestCase):
    def test_pooling(self):
        x = torch.tensor(
            [[[1.0], [3.0], [100.0]]]
        )
        mask = torch.tensor([[1, 1, 0]])
        self.assertAlmostEqual(
            masked_mean(x, mask).item(),
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            masked_max(x, mask).item(),
            3.0,
            places=6,
        )

    def test_rna_graph_shape(self):
        enc = RNA2DGraphEncoder(
            dim=16,
            layers=2,
            dropout=0.0,
        )
        h = torch.randn(2, 6, 16)
        feat = torch.randn(2, 6, 10)
        adj = torch.eye(6).repeat(2, 1, 1)
        mask = torch.ones(2, 6, dtype=torch.bool)
        out = enc(h, feat, adj, mask)
        self.assertEqual(out.shape, h.shape)

    def test_cross_attention_shape(self):
        block = CrossBlock(
            dim=16,
            heads=4,
            dropout=0.0,
        )
        r = torch.randn(2, 6, 16)
        c = torch.randn(2, 5, 16)
        rm = torch.ones(2, 6, dtype=torch.bool)
        cm = torch.ones(2, 5, dtype=torch.bool)
        ro, co = block(r, c, rm, cm)
        self.assertEqual(ro.shape, r.shape)
        self.assertEqual(co.shape, c.shape)


if __name__ == "__main__":
    unittest.main()
