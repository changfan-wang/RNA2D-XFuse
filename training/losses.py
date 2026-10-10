import torch
import torch.nn.functional as F


def auc_pair_loss(
    logits,
    labels,
    target_margin=0.0,
    max_pairs=8192,
):
    y = labels.view(-1).float()
    s = logits.view(-1)

    pos = s[y > 0.5]
    neg = s[y <= 0.5]

    if pos.numel() == 0 or neg.numel() == 0:
        return s.new_tensor(0.0)

    diff = pos[:, None] - neg[None, :]

    if diff.numel() > max_pairs:
        flat = diff.reshape(-1)
        index = torch.randperm(
            flat.numel(), device=flat.device
        )[:max_pairs]
        diff = flat[index]

    return F.softplus(target_margin - diff).mean()


def focal_bce_with_mask(
    logits,
    targets,
    mask,
    alpha=0.25,
    gamma=2.0,
    eps=1e-6,
):
    logits = logits.squeeze(-1)

    if targets.dim() == 3:
        targets = targets.squeeze(-1).float()
    else:
        targets = targets.float()

    mask = mask.float()

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    loss = alpha * (1 - pt).pow(gamma) * bce

    return (
        (loss * mask).sum()
        / mask.sum().clamp(min=eps)
    )
