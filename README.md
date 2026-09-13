# TVAE-GAN: Variational Autoencoder with GAN for Tabular Data Synthesis

A PyTorch implementation of a hybrid Variational Autoencoder (VAE) and Generative Adversarial Network (GAN) for generating synthetic tabular data. The synthetic data preserves the statistical properties and relationships of the original data without exposing the original rows.

## Overview

The model has three networks:

1. **Encoder**: compresses each row into a latent vector.
2. **Decoder/Generator**: a single network that reconstructs rows from encoded vectors (VAE role) and generates new rows from random noise (GAN role).
3. **Critic**: scores how realistic a row looks.

Training alternates between three steps:

- **Critic** (every batch): learns to score real rows higher than reconstructed and generated rows. Its weights are clipped to `±clip` (WGAN-style).
- **Generator** (every `s_generat` batches): learns to produce rows the critic scores as real.
- **Encoder + decoder** (every `s_encoder` batches): learns to reconstruct rows (MSE loss) while an optimal transport loss ([GeomLoss](https://www.kernel-operations.io/geomloss/)) pulls the latent vectors toward a standard normal distribution.

All networks use the RMSprop optimizer. Training runs on CUDA when available, otherwise on CPU.

## Requirements

- Python 3.10 or newer

## Installation

```bash
git clone <repository-url>
cd t-vae-gan
pip install -e .
```

This installs the `tvaegan` package in editable mode. `pip install -r requirements.txt` additionally installs Jupyter for the example notebook.

PyKeOps does not import on Windows. GeomLoss falls back to plain PyTorch and training still works; the `[KeOps] Warning` messages can be ignored.

## Quick Start

```python
import logging

import pandas as pd
from tvaegan import TVAEGANSynthesizer

logging.basicConfig(level=logging.INFO)  # show training progress

df_real = pd.read_csv("your_data.csv")

synthesizer = TVAEGANSynthesizer(epochs=700, batch_size=500)
synthesizer.fit(df_real)

df_synthetic = synthesizer.predict(samples=1000)
df_synthetic.to_csv("synthetic_data.csv", index=False)
```

Progress (column types, network sizes, per-epoch losses) is reported through Python's `logging` module at INFO level. Without `logging.basicConfig(...)` nothing is shown.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 700 | Number of training epochs |
| `batch_size` | int | 500 | Rows per training batch |
| `cat_emb_size` | int | 25 | Hidden-layer width added per categorical column |
| `num_emb_size` | int | 25 | Hidden-layer width added per numerical column, capped at the column's number of unique values |
| `hidden_layers_multipliers` | List[int] | [1, 1] | One entry per hidden layer; each layer's width is the total width above times its multiplier |
| `w_regularize` | float | 1.0 | Weight of the latent optimal transport loss |
| `w_reconstruct` | float | 10.0 | Weight of the reconstruction loss |
| `s_generat` | int | 5 | Train the generator every N batches |
| `s_encoder` | int | 5 | Train the encoder and decoder every N batches |
| `lr_generat` | float | 0.00005 | Generator learning rate |
| `lr_critic` | float | 0.00005 | Critic learning rate |
| `lr_encoder` | float | 0.00005 | Encoder and decoder learning rate |
| `clip` | float | 0.01 | Critic weights are clipped to `[-clip, clip]` after each step |
| `dropout` | float | 0.1 | Dropout rate in all networks |
| `ot_loss` | dict | `{"loss": "energy"}` | Arguments for GeomLoss `SamplesLoss` |
| `shuffle` | bool | True | Shuffle rows each epoch |
| `seed` | int | 42 | Seed for `torch`, `random` and `numpy`, applied when the model is built |

## Reproducible Sampling

`seed` makes training repeatable. `predict` accepts its own optional `seed`; with it, the same call returns the same rows every time:

```python
df_a = synthesizer.predict(1000, seed=7)
df_b = synthesizer.predict(1000, seed=7)  # identical to df_a
```

Without `seed`, each call continues from the current NumPy random state.

## Saving and Loading

```python
synthesizer.save("synthesizer.pt")

synthesizer = TVAEGANSynthesizer.load("synthesizer.pt")
df_synthetic = synthesizer.predict(1000)
```

- The file contains the parameters, the fitted preprocessing and the network weights. It does not contain training rows, but it does include column names, each numerical column's min and max, and every category value.
- A model saved on GPU can be loaded on a CPU-only machine.
- **Only load files you trust**: `load` unpickles Python objects, which can run arbitrary code.

## Data Handling

Preprocessing is automatic:

- Text (`object` or pandas `str`) and `bool` columns are categorical: one-hot encoded, then scaled to [-1, 1]. Missing values are kept and restored.
- All other columns are numerical: min-max scaled to [-1, 1]. Numerical columns must not contain missing values.
- Generated data is converted back to the original column order and dtypes. Float columns are rounded to the precision seen in the training data; `int64` columns are rounded to integers.

## Examples

`examples/example.py` trains on the Adult dataset (`examples/adult.csv`) for one epoch and writes `adult_synt.csv`. After installing the package, run it from the `examples` folder:

```bash
cd examples
python example.py
```

See `examples/example.ipynb` for a walkthrough with visualizations.

## Dependencies

- **PyTorch**: deep learning framework
- **GeomLoss**: optimal transport losses
- **PyKeOps**: faster kernel operations for GeomLoss (optional; not available on Windows)
- **scikit-learn**: preprocessing
- **pandas**: data handling
- **NumPy**: numerical computation

## License

This project is licensed under the Apache License. See the LICENSE file for details.

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{anshelevich2025synthetic,
  title={Synthetic tabular data generation using a VAE-GAN architecture},
  author={Anshelevich, Dmitry and Katz, Gilad},
  journal={Knowledge-Based Systems},
  pages={113997},
  year={2025},
  publisher={Elsevier}
}
```
