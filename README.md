# Spectral sirens

Spectral siren cosmology with the population of compact binary coalescences. This repository contains code to perform hierarchical Bayesian inference, mock data generation and sensitivity estimates.

The inference code is written in [JAX](https://github.com/google/jax) and [numpyro](https://num.pyro.ai). The mock data generation and sensitivity estimates are written in python.

Notebooks detailing how to run the inference and simulate observations are provided in the [`examples`](/examples/) folder. A set of mock gravitational wave events and sensitivity estimates are computed using this code and stored in [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10655745.svg)](https://doi.org/10.5281/zenodo.10655745)
.

## Installation

To install the code, clone the repository and install the dependencies with pip:

```bash
git clone
cd spectral-sirens
pip install .
```

The base code builds on the IGWN conda environment. To install the IGWN conda environment, see [here](https://computing.docs.ligo.org/conda/environments/igwn/) for instructions and details.

To run the inference on GPUs you need jaxlib and numpyro with GPU support. See [JAX docs](https://github.com/google/jax#pip-installation-gpu-cuda) and [numpyro docs](https://num.pyro.ai/en/latest/getting_started.html) for installation instructions. In particular it requires CUDA to be installed first.

In the folder `envs` you can find the conda environment files used to run the inference on GPUs. To install the conda environment, run

```bash
conda env create -f envs/inference_gpu.yml
```

## Citing spectral sirens

If you use this code, please consider citing the following papers:

```bibtex
@article{Chen:2024gdn,
    author = "Chen, Hsin-Yu and Ezquiaga, Jose Mar\'\i{}a and Gupta, Ish",
    title = "{Cosmography with next-generation gravitational wave detectors}",
    eprint = "2402.03120",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "2",
    year = "2024"
}
```

and
    
```bibtex
@article{Ezquiaga:2022zkx,
    author = "Ezquiaga, Jose Mar\'\i{}a and Holz, Daniel E.",
    title = "{Spectral Sirens: Cosmology from the Full Mass Distribution of Compact Binaries}",
    eprint = "2202.08240",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1103/PhysRevLett.129.061102",
    journal = "Phys. Rev. Lett.",
    volume = "129",
    number = "6",
    pages = "061102",
    year = "2022"
}
```
