## The EM Perspective of Directional Mean Shift Algorithm

This repository contains code to generate the figures in the paper [The EM Perspective of Directional Mean Shift Algorithm](https://arxiv.org/abs/2101.10058), (2020).

### Requirements

- Python >= 3.6 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (A speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) is used to compute the modified Bessel function of the first kind of real order).

### Description
Some high-level descriptions of each python script are as follows:

- **Basin_of_Attraction.py**: This script generates the plots of basins of attraction for directional mean shift algorithm (Figure 1 in the paper). It may take 5-7 minutes to run.
- **DirMS_fun_Fast.py**: This script implements the directional KDE and directional mean shift algorithm with the von Mises and other general kernels in order to accommodate multi-processing iterations.
- **Utility_fun.py**: This script contains all the utility functions for our experiments.
