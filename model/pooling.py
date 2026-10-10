import torch


def masked_mean(x, mask, dim=1, eps=1e-6):
    m = mask.float()
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp(min=eps)


def masked_max(x, mask, dim=1):
    m = mask.bool()
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    return x.masked_fill(~m, -1e4).max(dim=dim).values
