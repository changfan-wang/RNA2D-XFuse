import torch
from torch.utils.data import Dataset


TOKEN_TO_BASE = {4: "A", 5: "C", 6: "G", 7: "U"}


def rna_tokens_to_seq(rna_tokens):
    ids = rna_tokens.detach().cpu().view(-1).tolist()
    return "".join(
        TOKEN_TO_BASE[int(x)]
        for x in ids
        if int(x) in TOKEN_TO_BASE
    )


class RNA2DAugmentedDataset(Dataset):
    """Attach structural features to an existing interaction dataset.

    Parameters
    ----------
    base_dataset:
        Dataset returning the original interaction tuple.
    structure_provider:
        Callable ``provider(sequence, max_token_len)`` returning
        ``(struct_feat, struct_adj)`` tensors.
    """

    def __init__(self, base_dataset, structure_provider):
        self.base = base_dataset
        self.structure_provider = structure_provider

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        graph = item[0]
        rna_tokens = item[1]
        bs_label = item[2]
        rna_seq_len = item[3]
        dti_label = item[4]

        seq = rna_tokens_to_seq(rna_tokens)
        struct_feat, struct_adj = self.structure_provider(
            seq, int(rna_tokens.numel())
        )

        return (
            graph,
            rna_tokens,
            bs_label,
            rna_seq_len,
            dti_label,
            struct_feat,
            struct_adj,
        )


def move_rna2d_batch(
    rna_input,
    compound_input,
    bs_label,
    dti_label,
    struct_feat,
    struct_adj,
    device,
):
    return (
        rna_input.to(device),
        tuple(x.to(device) for x in compound_input),
        bs_label.to(device),
        dti_label.to(device),
        (struct_feat.to(device).float(), struct_adj.to(device).float()),
    )
