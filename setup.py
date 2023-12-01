import setuptools

verstr = "0.0.1"

setuptools.setup(
    name="spectral_sirens",
    version=verstr,
    author="Jose María Ezquiaga",
    author_email="jose.ezquiaga@nbi.ku.dk",
    description="Spectral siren cosmology with the population of compact binary coalescences. This repository contains code to perform hierarchical Bayesian inference, mock data generation and sensitivity estimates.",
    packages=[
        "spectral_sirens",
    ],
    #install_requires=[
    #    "phazap >= 0.3.0",
    #],

    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    python_requires='>=3.7',
)
