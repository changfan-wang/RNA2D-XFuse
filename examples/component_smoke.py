import torch

from model.cross_attention import CrossBlock
from model.rna_graph import RNA2DGraphEncoder


def main():
    batch = 2
    rna_len = 7
    compound_len = 5
    dim = 32

    rna = torch.randn(batch, rna_len, dim)
    compound = torch.randn(batch, compound_len, dim)
    rna_mask = torch.ones(batch, rna_len, dtype=torch.bool)
    compound_mask = torch.ones(
        batch, compound_len, dtype=torch.bool
    )

    struct_feat = torch.randn(batch, rna_len, 10)
    struct_adj = torch.eye(rna_len).repeat(batch, 1, 1)

    graph = RNA2DGraphEncoder(
        dim=dim,
        layers=2,
        dropout=0.0,
    )
    cross = CrossBlock(
        dim=dim,
        heads=4,
        dropout=0.0,
    )

    rna = graph(
        rna,
        struct_feat,
        struct_adj,
        rna_mask,
    )
    rna, compound = cross(
        rna,
        compound,
        rna_mask,
        compound_mask,
    )

    print("RNA:", tuple(rna.shape))
    print("Compound:", tuple(compound.shape))


if __name__ == "__main__":
    main()
