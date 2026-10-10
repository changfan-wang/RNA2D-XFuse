# RNA2D-XFuse

Public research code accompanying **RNA2D-XFuse** for RNA–small-molecule
interaction prediction and nucleotide-level binding-site prediction.

This repository releases the data adapters, RNA secondary-structure utilities,
backbone components, cross-attention blocks, training losses, evaluation
functions, and fold-level statistics used around the model.

## Release scope

The public release is intentionally partial. The following parts of the final
interaction head are not included in this version:

- the complete Top-K residue–atom interaction scorer;
- the final global/local score construction;
- the sample-adaptive dynamic fusion head;
- the complete end-to-end `RNA2D-XFuse` model wrapper and trained checkpoints.

The released modules are therefore useful for inspecting the preprocessing,
RNA-graph encoder, bidirectional cross-attention, loss functions, evaluation
conventions, and backbone data flow, but they are **not an end-to-end
reproduction package for the full model**.

## Repository structure

```text
RNA2D-XFuse/
├── data/
│   ├── aligned_rna2d.py
│   └── rna_structure.py
├── model/
│   ├── backbone.py
│   ├── cross_attention.py
│   ├── pooling.py
│   └── rna_graph.py
├── training/
│   ├── losses.py
│   ├── objective.py
│   └── reproducibility.py
├── evaluation/
│   ├── fold_summary.py
│   └── metrics.py
├── examples/
│   └── component_smoke.py
└── tests/
```

## Core components released

### RNA secondary-structure encoder

`model/rna_graph.py` contains the graph refinement block used to combine
token-level RNA representations with structural node features and adjacency
information.

### Bidirectional cross-attention

`model/cross_attention.py` contains the RNA-to-compound and
compound-to-RNA cross-attention block.

### Public backbone

`model/backbone.py` exposes the public portion of the representation pipeline:
pretrained RNA/compound embeddings, projection layers, RNA graph refinement,
and stacked bidirectional cross-attention blocks. It deliberately stops before
the private interaction-scoring and dynamic-fusion head.

### Training objectives

`training/losses.py` contains the pairwise AUC surrogate and masked focal
binding-site loss used in the experiments. `training/objective.py` documents
the composition of the interaction, ranking, memory, and binding-site terms
without exposing the private model head.

### Evaluation

`evaluation/metrics.py` provides ROC-AUC/AP helpers and binding-site pooling
utilities. `evaluation/fold_summary.py` reports mean and sample standard
deviation across folds (`ddof=1`).

## Dependencies

The released components use Python 3.8+ and expect compatible versions of:

```bash
pip install -r requirements.txt
```

The full model additionally depends on pretrained RNA-FM and Mole-BERT
checkpoints and on the unreleased interaction head.

## Quick check

```bash
python -m unittest discover -s tests -v
python examples/component_smoke.py
```

## Data convention

The experimental benchmark uses the four inherited evaluation settings:

- unseen pair;
- unseen RNA;
- unseen compound;
- unseen both.

A sample contains an RNA token sequence, a molecular graph, an interaction
label, optional nucleotide-level binding-site labels, and RNA structural
features/alignment information.

## Reproducibility note

Model selection in our experiments is performed using validation DTI AUC.
Test metrics are computed after the selected state is restored. Fold-level
results should be summarized using all predefined folds rather than selecting
the best-performing test folds.

## Citation

Citation information will be updated with the final publication record.
