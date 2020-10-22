# Directional Mean Shift Algorithm
Implementing the directional mean shift algorithm using Python3

- Paper Reference: 
- We provide a Python3 implementation of our mean shift algorithm with directional data.

### Requirements

- Python >= 3.6 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (A speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) is used to compute the modified Bessel function of the first kind of real order; the function [scipy.stats.entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) is applied to calculate the entropy of the relative frequencies in the blurring version of our directional mean shift algorithm), [collections](https://docs.python.org/3.6/library/collections.html) (The `Counter` function is used).

### Description
Some high-level descriptions of each python script are as follows:

- **Blurring_DirMS.py**: This script runs the repeated experiments of the directional mean shift algorithm and its blurring version on simulated vMF distributed data sets with one, two, and three modes, repectively (Table 1 in the paper). 
- **Circular_Sim.py**: This script generates all the plots of our simulation study in the circular case (Figure 5 in the paper).
- **Craters_on_Mars.py**: This script generates all the plots of real-world applications on crater data on Mars (Figure 7 and 10 in the paper).
- **DirMS_fun.py**: This script implements the main functions for the directional KDE, directional mean shift algorithm, and blurring directional mean shift algorithm.
- **Earthquakes.py**: This script generates all the plots of real-world applications on the earthquake data (Figure 8 in the paper).
- **Spherical_Sim.py**: This script generates all the plots of our simulation studies in the spherical cases (Figure 1, 6, and 9 in the paper).
- **Utility.py**: This script contains all the utility functions for our experiments.
- **vMF_Density_Fig.py**: This script generates the contour plots of a 2-von Mises-Fisher density and a mixture of 2-vMF densities (Figure 2 in the paper).

#### Directional Kernel Density Estimation (KDE)

Given a random directional sample <img src="https://latex.codecogs.com/svg.latex?\Large&space;\left\{\mathbf{X}_1,...,\mathbf{X}_n\right\}" />
