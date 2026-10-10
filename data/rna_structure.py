def clean_rna_seq(seq):
    return "".join(
        c
        for c in str(seq).upper().replace("T", "U")
        if c in {"A", "C", "G", "U"}
    )


def dotbracket_to_pairs(dot):
    pairs = []
    closing = {")": "(", "]": "[", "}": "{", ">": "<"}
    stacks = {"(": [], "[": [], "{": [], "<": []}

    for i, ch in enumerate(str(dot).strip()):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in closing:
            op = closing[ch]
            if stacks[op]:
                j = stacks[op].pop()
                pairs.append((j, i))

    return pairs


def can_pair_base(a, b):
    a = str(a).upper().replace("T", "U")
    b = str(b).upper().replace("T", "U")
    return (a, b) in {
        ("A", "U"),
        ("U", "A"),
        ("G", "C"),
        ("C", "G"),
        ("G", "U"),
        ("U", "G"),
    }


def greedy_secondary_pairs(seq, min_loop=3):
    seq = clean_rna_seq(seq)
    n = len(seq)
    pairs = []
    used = [False] * n

    i = 0
    j = n - 1

    while i < j:
        if j - i <= min_loop:
            i += 1
            j = n - 1
            continue

        if (
            not used[i]
            and not used[j]
            and can_pair_base(seq[i], seq[j])
        ):
            pairs.append((i, j))
            used[i] = True
            used[j] = True
            i += 1
            j -= 1
        else:
            j -= 1
            if j <= i + min_loop:
                i += 1
                j = n - 1

    return pairs


def build_simple_adjacency(seq, pairs, add_local_edges=True):
    import torch

    n = len(clean_rna_seq(seq))
    adj = torch.zeros((n, n), dtype=torch.float32)

    if add_local_edges:
        for i in range(n - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0

    for i, j in pairs:
        if 0 <= i < n and 0 <= j < n:
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return adj
