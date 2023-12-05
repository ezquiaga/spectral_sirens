# Spectral sirens

Spectral siren cosmology with the population of compact binary coalescences. This repository contains code to perform hierarchical Bayesian inference, mock data generation and sensitivity estimates.

The inference code is written in JAX and numpyro. The mock data generation and sensitivity estimates are written in python.

## Installation

To install the code, clone the repository and install the dependencies with pip:

```bash
git clone
cd spectral-sirens
pip install .
```

The base code builds on the IGWN conda environment. To install the IGWN conda environment, see [here](https://computing.docs.ligo.org/conda/environments/igwn/) for instructions and details.

To run the inference on GPUs you need jaxlib and numpyro with GPU support. See [JAX docs](https://github.com/google/jax#pip-installation-gpu-cuda) and [numpyro docs](https://num.pyro.ai/en/latest/getting_started.html) for installation instructions. In particular it requires CUDA to be installed first.
