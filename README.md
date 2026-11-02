# RNA2D-XFuse

Official code repository for **RNA2D-XFuse: Structure-Aware Global and Local Evidence Fusion for Predicting RNA–Small Molecule Interactions and Binding Sites**.

This repository provides the implementation of RNA2D-XFuse for RNA–small molecule interaction prediction and compound-specific binding-site prediction.

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

## Core components

### RNA secondary-structure encoder

`model/rna_graph.py` contains the graph refinement block used to combine
token-level RNA representations with structural node features and adjacency
information.

### Bidirectional cross-attention

`model/cross_attention.py` contains the RNA-to-compound and
compound-to-RNA cross-attention block.

### Representation backbone

`model/backbone.py` implements the representation backbone, integrating
pretrained RNA and compound embeddings, projection layers, RNA graph
refinement, and stacked bidirectional cross-attention blocks.

### Training objectives

`training/losses.py` implements the pairwise AUC-ranking loss and masked
focal binding-site loss. `training/objective.py` defines the combined
training objective.

### Evaluation

`evaluation/metrics.py` provides ROC-AUC/AP helpers and binding-site pooling
utilities. `evaluation/fold_summary.py` reports mean and sample standard
deviation across folds (`ddof=1`).

## Dependencies

The code requires Python 3.8+. Install the dependencies with:

```bash
pip install -r requirements.txt
```

Pretrained RNA-FM and Mole-BERT checkpoints are required for feature extraction.
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
