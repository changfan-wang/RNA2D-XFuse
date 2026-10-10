import torch
import torch.nn as nn
import torch.nn.functional as F


class RNA2DGraphEncoder(nn.Module):
    def __init__(self, dim=256, struct_dim=10, layers=2, dropout=0.15):
        super().__init__()
        self.layers = int(layers)
        self.struct_proj = nn.Linear(struct_dim, dim)
        self.msg = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(max(0, layers))]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(dim * 2, dim) for _ in range(max(0, layers))]
        )
        self.norm = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(max(0, layers))]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 2, dim),
                    nn.Dropout(dropout),
                )
                for _ in range(max(0, layers))
            ]
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def normalize_adj(adj, mask):
        adj = adj * mask.unsqueeze(1).float() * mask.unsqueeze(2).float()
        eye = torch.eye(
            adj.size(-1), device=adj.device, dtype=adj.dtype
        ).unsqueeze(0)
        adj = torch.clamp(
            adj + eye * mask.unsqueeze(1).float(), max=1.0
        )
        deg = adj.sum(dim=-1).clamp(min=1.0)
        dinv = deg.pow(-0.5)
        return adj * dinv.unsqueeze(-1) * dinv.unsqueeze(-2)

    def forward(self, rna_h, struct_feat, struct_adj, mask):
        length = rna_h.size(1)

        if struct_feat.size(1) != length:
            if struct_feat.size(1) > length:
                struct_feat = struct_feat[:, :length]
                struct_adj = struct_adj[:, :length, :length]
            else:
                pad_l = length - struct_feat.size(1)
                struct_feat = F.pad(struct_feat, (0, 0, 0, pad_l))
                struct_adj = F.pad(struct_adj, (0, pad_l, 0, pad_l))

        h = rna_h + self.struct_proj(struct_feat)

        if self.layers <= 0:
            return h * mask.unsqueeze(-1).float()

        adj = self.normalize_adj(struct_adj, mask)

        for msg, gate, norm, ffn in zip(
            self.msg, self.gate, self.norm, self.ffn
        ):
            m = msg(torch.bmm(adj, h))
            g = torch.sigmoid(gate(torch.cat([h, m], dim=-1)))
            h = norm(h + self.dropout(g * m))
            h = norm(h + ffn(h))
            h = h * mask.unsqueeze(-1).float()

        return h
