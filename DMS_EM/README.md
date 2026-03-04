# The EM Perspective of Directional Mean Shift Algorithm

This repository contains Python implementations and experiments for the paper [The EM Perspective of Directional Mean Shift Algorithm](https://arxiv.org/abs/2101.10058), (2021+).

## Overview

This project provides implementations of the directional kernel density estimation (KDE) and directional mean shift algorithm with the von Mises kernel. The code includes algorithms for computing basins of attraction and applications to real-world astronomical data from the SDSS galaxy survey.

## Requirements

- Python >= 3.6 (earlier versions might be applicable)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit)
- [pandas](https://pandas.pydata.org/)
- [Ray](https://ray.io/) (for parallel computing)
- [SciPy](https://www.scipy.org/) (specifically [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) for the modified Bessel function of the first kind)

## File Descriptions

### Core Modules

| File | Description |
|------|-------------|
| DirMS_fun_Fast.py | Implements the directional KDE and directional mean shift algorithm with the von Mises and other general kernels, optimized for multi-processing iterations. |
| Utility_fun.py | Contains utility functions for data processing, visualization, and statistical computations. |
| Basin_of_Attraction.py | Generates plots of basins of attraction for the directional mean shift algorithm (Figure 2 in the paper). Runtime: 5-7 minutes |
| Basins of Attraction on SDSS Galaxies.ipynb | Demonstrates the application of directional mean shift to detect basins of attraction on the SDSS galaxy dataset. Runtime: 1-2 hours|

## Data

The `Data/` directory contains:

- **SDSSIV_gal_dat.csv**: A snapshot of the real-world astronomical dataset from the Sloan Digital Sky Survey (SDSS) containing galaxy measurements and features for validation experiments.

## Results and Figures

- **Figures/**: Generated figures and visualizations from experiments.
- **Results/**: Computed results and data outputs from SDSS galaxy applications.
