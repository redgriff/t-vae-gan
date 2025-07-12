# TVAE-GAN: Variational Autoencoder with GAN for Tabular Data Synthesis

A Python implementation of a hybrid model combining Variational Autoencoder (VAE) and Generative Adversarial Network (GAN) for generating synthetic tabular data. This project is designed to create realistic synthetic datasets that preserve the statistical properties and relationships of the original data while ensuring privacy.

## Overview

TVAE-GAN combines the strengths of both VAE and GAN architectures:
- **VAE Component**: Learns a compressed representation of the data and ensures reconstruction quality
- **GAN Component**: Uses adversarial training to generate more realistic synthetic data
- **Optimal Transport Loss**: Incorporates geometric loss functions for better distribution matching

## Architecture

The model consists of three main components:

1. **Encoder (VAE)**: Compresses input data into a latent representation
2. **Generator/Decoder**: Reconstructs data from latent representations
3. **Critic/Discriminator**: Evaluates the quality of generated samples

### Key Features

- **Mixed Data Types**: Handles both categorical and numerical data automatically
- **Automatic Preprocessing**: Scales and encodes data appropriately
- **Configurable Architecture**: Adjustable network sizes and training parameters
- **Privacy-Preserving**: Generates synthetic data that maintains statistical properties without exposing original data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd t-vae-gan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.tvaegan_synthesizer import TVAEGANSynthesizer
import pandas as pd

# Load your data
df_real = pd.read_csv('your_data.csv')

# Create and train the synthesizer
synthesizer = TVAEGANSynthesizer(epochs=700, batch_size=500)
synthesizer.fit(df_real)

# Generate synthetic data
df_synthetic = synthesizer.predict(samples=1000)
df_synthetic.to_csv('synthetic_data.csv', index=False)
```

### Example with Custom Parameters

```python
synthesizer = TVAEGANSynthesizer(
    epochs=700,
    batch_size=500,
    cat_emb_size=25,
    num_emb_size=25,
    w_regularize=1.0,
    w_reconstruct=10.0,
    s_generat=5,
    s_encoder=5,
    lr_generat=0.00005,
    lr_critic=0.00005,
    lr_encoder=0.00005,
    clip=0.01,
    dropout=0.1,
    hidden_layers_multipliers=[1, 1]
)
```

## Parameters

### TVAEGANSynthesizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 700 | Number of training epochs |
| `batch_size` | int | 500 | Batch size for training |
| `cat_emb_size` | int | 25 | Embedding size for categorical features |
| `num_emb_size` | int | 25 | Embedding size for numerical features |
| `w_regularize` | float | 1.0 | Weight for regularization loss |
| `w_reconstruct` | float | 10.0 | Weight for reconstruction loss |
| `s_generat` | int | 5 | Generator training frequency |
| `s_encoder` | int | 5 | Encoder training frequency |
| `lr_generat` | float | 0.00005 | Generator learning rate |
| `lr_critic` | float | 0.00005 | Critic learning rate |
| `lr_encoder` | float | 0.00005 | Encoder learning rate |
| `clip` | float | 0.01 | Gradient clipping value |
| `dropout` | float | 0.1 | Dropout rate |
| `ot_loss` | dict | `{"loss": "energy"}` | Optimal transport loss configuration |
| `hidden_layers_multipliers` | List[int] | [1, 1] | Hidden layer size multipliers |
| `shuffle` | bool | True | Whether to shuffle data during training |

## Data Requirements

The synthesizer automatically handles:
- **Categorical Data**: Object and boolean types are treated as categorical
- **Numerical Data**: Integer and float types are treated as numerical
- **Data Scaling**: Categorical data is one-hot encoded and scaled, numerical data is min-max scaled

## Training Process

The training involves three components working together:

1. **Encoder Training**: Learns to compress data into latent space
2. **Generator Training**: Learns to generate realistic data from latent space
3. **Critic Training**: Learns to distinguish real from synthetic data

The model uses:
- **RMSprop** optimizer for all components
- **Optimal Transport Loss** for distribution matching
- **MSE Loss** for reconstruction
- **Gradient Clipping** for critic stability

## Examples

### Adult Dataset Example

```python
# Load the adult dataset
df_real = pd.read_csv('examples/adult.csv')

# Create synthesizer with minimal training for demo
synthesizer = TVAEGANSynthesizer(epochs=10)

# Train the model
synthesizer.fit(df_real)

# Generate synthetic data
df_synthetic = synthesizer.predict(len(df_real))
df_synthetic.to_csv('adult_synthetic.csv', index=False)
```

### Jupyter Notebook

See `examples/example.ipynb` for a detailed walkthrough with visualizations.

## Performance Considerations

- **GPU Support**: Automatically uses CUDA if available
- **Memory Usage**: Batch size affects memory consumption
- **Training Time**: Depends on dataset size and number of epochs
- **Quality**: More epochs generally improve synthetic data quality

## Dependencies

- **PyTorch**: Deep learning framework
- **GeomLoss**: Optimal transport loss functions
- **PyKeOps**: Efficient kernel operations
- **scikit-learn**: Data preprocessing utilities
- **pandas**: Data manipulation
- **numpy**: Numerical computations

## License

This project is licensed under the Apache License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tvaegan2023,
  title={TVAE-GAN: Variational Autoencoder with GAN for Tabular Data Synthesis},
  author={Your Name},
  year={2023},
  publisher={GitHub},
  url={https://github.com/yourusername/t-vae-gan}
}
```