# Directional Mean Shift Algorithm
Implementing the directional mean shift algorithm using Python3

- Paper Reference: 
- We provide a Python3 implementation of our mean shift algorithm with directional data.

### Requirements

- Python >= 3.6 (Earlier version might be applicable.)
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (A speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) is used to compute the modified Bessel function of the first kind of real order; the function [scipy.stats.entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) is applied to calculate the entropy of the relative frequencies in the blurring version of our directional mean shift algorithm), [collections](https://docs.python.org/3.6/library/collections.html) (The `Counter` function is used).
