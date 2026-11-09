# Sparse Mixture-of-Experts for Efficient Multilingual Machine Translation

This repository contains the implementation for the paper
**"Sparse Mixture-of-Experts for Efficient Multilingual Machine Translation"**
(EMNLP 2025).

The code includes the sparse top-k router, the shared-expert translation
backbone and the training/evaluation pipeline used in the paper. The released
checkpoint configuration is in `configs/base.yaml`.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m src.moe_mt.train --config configs/base.yaml
```

## Evaluation

```bash
python -m src.moe_mt.train --config configs/base.yaml --eval-only
```

## Citation

```bibtex
@inproceedings{chen2025sparse,
  title={Sparse Mixture-of-Experts for Efficient Multilingual Machine Translation},
  author={Chen, Maya and Morales, Patrick and others},
  booktitle={Proceedings of EMNLP 2025},
  year={2025}
}
```
