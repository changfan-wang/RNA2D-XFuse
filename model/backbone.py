import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .cross_attention import CrossBlock
from .rna_graph import RNA2DGraphEncoder


class RNA2DXFuseBackbone(nn.Module):
    """Public representation backbone.

    This module intentionally stops before the unreleased interaction-scoring
    and adaptive-fusion head.
    """

    def __init__(
        self,
        rna_model,
        compound_model,
        hidden_dim=256,
        struct_layers=2,
        cross_layers=2,
        num_heads=8,
        dropout=0.22,
        freeze_rna=True,
        freeze_compound=True,
    ):
        super().__init__()
        self.rna_embedding = rna_model
        self.compound_embedding = compound_model
        self.freeze_rna = bool(freeze_rna)
        self.freeze_compound = bool(freeze_compound)

        self.rna_proj = nn.Sequential(
            nn.Linear(640, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.comp_proj = nn.Sequential(
            nn.Linear(300, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.struct_encoder = RNA2DGraphEncoder(
            dim=hidden_dim,
            layers=struct_layers,
            dropout=dropout,
        )
        self.cross = nn.ModuleList(
            [
                CrossBlock(hidden_dim, num_heads, dropout)
                for _ in range(cross_layers)
            ]
        )

    def encode_rna(self, rna_input):
        rmask = (rna_input >= 4) & (rna_input <= 7)

        if self.freeze_rna:
            self.rna_embedding.eval()
            with torch.no_grad():
                h = self.rna_embedding(
                    rna_input, repr_layers=[12]
                )["representations"][12]
        else:
            h = self.rna_embedding(
                rna_input, repr_layers=[12]
            )["representations"][12]

        h = self.rna_proj(h)
        return h * rmask.unsqueeze(-1).float(), rmask

    def encode_compound(self, compound_input):
        x, edge_index, edge_attr, batch = compound_input

        if self.freeze_compound:
            self.compound_embedding.eval()
            with torch.no_grad():
                _, node_h = self.compound_embedding(
                    x, edge_index, edge_attr, batch
                )
        else:
            _, node_h = self.compound_embedding(
                x, edge_index, edge_attr, batch
            )

        h = self.comp_proj(node_h)
        h, cmask = to_dense_batch(h, batch)
        return h * cmask.unsqueeze(-1).float(), cmask

    def forward(self, rna_input, compound_input, rna_struct):
        r, rmask = self.encode_rna(rna_input)
        c, cmask = self.encode_compound(compound_input)

        struct_feat, struct_adj = rna_struct
        r = r + self.struct_encoder(
            r, struct_feat, struct_adj, rmask
        )

        for block in self.cross:
            r, c = block(r, c, rmask, cmask)

        return {
            "rna": r,
            "compound": c,
            "rna_mask": rmask,
            "compound_mask": cmask,
        }
