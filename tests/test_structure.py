import unittest

from data.rna_structure import (
    can_pair_base,
    dotbracket_to_pairs,
    greedy_secondary_pairs,
)


class StructureTests(unittest.TestCase):
    def test_dotbracket(self):
        pairs = dotbracket_to_pairs("((..))")
        self.assertEqual(
            sorted(pairs),
            [(0, 5), (1, 4)],
        )

    def test_pair_rules(self):
        self.assertTrue(can_pair_base("G", "C"))
        self.assertTrue(can_pair_base("G", "U"))
        self.assertFalse(can_pair_base("A", "C"))

    def test_greedy_pairs_bounds(self):
        seq = "GGGAAACCC"
        pairs = greedy_secondary_pairs(
            seq, min_loop=2
        )
        for i, j in pairs:
            self.assertLess(i, j)
            self.assertGreater(j - i, 2)


if __name__ == "__main__":
    unittest.main()
