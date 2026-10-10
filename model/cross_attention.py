import torch.nn as nn


class CrossBlock(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.15):
        super().__init__()
        self.r2c = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.c2r = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.rn1 = nn.LayerNorm(dim)
        self.cn1 = nn.LayerNorm(dim)
        self.rn2 = nn.LayerNorm(dim)
        self.cn2 = nn.LayerNorm(dim)
        self.rffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.cffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, r, c, rmask, cmask):
        ro, _ = self.r2c(
            query=self.rn1(r),
            key=c,
            value=c,
            key_padding_mask=~cmask.bool(),
            need_weights=False,
        )
        r = r + ro
        r = r + self.rffn(self.rn2(r))
        r = r * rmask.unsqueeze(-1).float()

        co, _ = self.c2r(
            query=self.cn1(c),
            key=r,
            value=r,
            key_padding_mask=~rmask.bool(),
            need_weights=False,
        )
        c = c + co
        c = c + self.cffn(self.cn2(c))
        c = c * cmask.unsqueeze(-1).float()

        return r, c
