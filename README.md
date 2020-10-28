# Directional Mean Shift Algorithm
Implementing the directional mean shift algorithm using Python3

- Paper Reference: [Kernel Smoothing, Mean Shift, and Their Learning Theory with Directional Data](https://arxiv.org/abs/2010.13523) (2020)
- We provide a Python3 implementation of our mean shift algorithm with directional data.

## Requirements

- Python >= 3.6 (earlier version might be applicable).
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/) (especially the [Basemap](https://matplotlib.org/basemap/) toolkit), [pandas](https://pandas.pydata.org/), [SciPy](https://www.scipy.org/) (A speical function [scipy.special.iv](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv) is used to compute the modified Bessel function of the first kind of real order; the function [scipy.stats.entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) is applied to calculate the entropy of the relative frequencies in the blurring version of our directional mean shift algorithm), [collections](https://docs.python.org/3.6/library/collections.html) (The `Counter` function is used).

## Description
Some high-level descriptions of each python script are as follows:

- **Blurring_DirMS.py**: This script runs the repeated experiments of the directional mean shift algorithm and its blurring version on simulated vMF distributed data sets with one, two, and three modes, repectively (Table 1 in the paper). 
- **Circular_Sim.py**: This script generates all the plots of our simulation study in the circular case (Figure 5 in the paper).
- **Craters_on_Mars.py**: This script generates all the plots of real-world applications on crater data on Mars (Figure 7 and 10 in the paper).
- **DirMS_fun.py**: This script implements the main functions for the directional KDE, directional mean shift algorithm, and blurring directional mean shift algorithm.
- **Earthquakes.py**: This script generates all the plots of real-world applications on the earthquake data (Figure 8 in the paper).
- **Spherical_Sim.py**: This script generates all the plots of our simulation studies in the spherical cases (Figure 1, 6, and 9 in the paper).
- **Utility.py**: This script contains all the utility functions for our experiments.
- **vMF_Density_Fig.py**: This script generates the contour plots of a 2-von Mises-Fisher density and a mixture of 2-vMF densities (Figure 2 in the paper).

### Directional Kernel Density Estimation (KDE)

Given a random directional sample <img src="https://latex.codecogs.com/svg.latex?\Large&space;\left\{\mathbf{X}_1,...,\mathbf{X}_n\right\}\subset\Omega_q" />, where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\Omega_q=\left\{\mathbf{x}\in\mathbb{R}^{q+1}:||\mathbf{x}||_2=1\right\}\subset\mathbb{R}^{q+1}" /> is the q-dimensional unit sphere, the directional KDE is written as 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{f}_h(\mathbf{x})=\frac{c_{h,q}(L)}{n}\sum_{i=1}^nL\left(\frac{1-\mathbf{x}^T\mathbf{X}_i}{h^2}\right)," />

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;L" /> is a directional kernel function, <img src="https://latex.codecogs.com/svg.latex?\Large&space;h" /> is the smoothing bandwidth parameter, <img src="https://latex.codecogs.com/svg.latex?\Large&space;c_{h,q}\asymp\,h^{-q}" /> is a normalizing constant to ensure that <img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{f}_h" /> is a probability density function. Another equivalent form of the directional KDE is given by

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widetilde{f}_h(\mathbf{x})=\frac{c_{h,q}(L)}{n}\sum_{i=1}^nL\left(\left|\left|\frac{\mathbf{x}-\mathbf{X}_i}{h}\right|\right|_2^2\right)," />

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;||\cdot||_2" /> is the usual Euclidean norm, because <img src="https://latex.codecogs.com/svg.latex?\Large&space;||\mathbf{x}-\mathbf{X}_i||_2^2=2(1-\mathbf{x}^T\mathbf{X}_i)" /> on <img src="https://latex.codecogs.com/svg.latex?\Large&space;\Omega_q" />. The first form of the directional KDE is the commonly used one in the literature, while the second form sheds light on the derivation of our directional mean shift algorithm. 

### Directional Mean Shift Algorithm

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla\widetilde{f}_h(\mathbf{x})=\frac{c_{h,q}(L)}{nh^2}\sum_{i=1}^n(\mathbf{x}-\mathbf{X}_i)\cdot\,L'\left(\frac{1}{2}\left|\left|\frac{\mathbf{x}-\mathbf{X}_i}{h}\right|\right|_2^2\right)\\=\frac{c_{h,q}(L)}{nh^2}\left[\sum_{i=1}^n-L'\left(\frac{1}{2}\left|\left|\frac{\mathbf{x}-\mathbf{X}_i}{h}\right|\right|_2^2\right)\right]\cdot\,\left[\frac{\sum_{i=1}^n\mathbf{X}_i\cdot\,L'\left(\frac{1}{2}\left|\left|\frac{\mathbf{x}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}{\sum_{i=1}^n\,L'\left(\frac{1}{2}\left|\left|\frac{\mathbf{x}-\mathbf{X}_i}{h}\right|\right|_2^2\right)}-\mathbf{x}\right]." />

The first term of the above product can be viewed as a proportional form of the directional density estimate with the "kernel" <img src="https://latex.codecogs.com/svg.latex?\Large&space;G(r)=-L'(r)" />. The second term ofthe above product is indeed the _mean shift_ vector. This shows that the directional mean shift vector aligns with the total gradient of <img src="https://latex.codecogs.com/svg.latex?\Large&space;\widetilde{f}_h" />; thus, moving along the directional mean shift direction yields an ascending path to a local mode of <img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{f}_h=\widetilde{f}_h" />. However, due to the manifold structure of <img src="https://latex.codecogs.com/svg.latex?\Large&space;\Omega_q" />, translating point <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathbf{x}\in\Omega_q" /> in the mean shift direction deviates the point from <img src="https://latex.codecogs.com/svg.latex?\Large&space;\Omega_q" />. We thus need to project the translated point back to <img src="https://latex.codecogs.com/svg.latex?\Large&space;\Omega_q" /> via a simple standardization <img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\mathbf{x}}{||\mathbf{x}||_2}" />.

We also encapsulate the entire directional mean shift algorithm into a single fixed-point iteration

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\widehat{\mathbf{y}}_{s+1}=-\frac{\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\widehat{\mathbf{y}}_s^T\,\mathbf{X}_i}{h^2}\right)}{\left|\left|\sum_{i=1}^n\mathbf{X}_i\,L'\left(\frac{1-\widehat{\mathbf{y}}_s^T\,\mathbf{X}_i}{h^2}\right)\right|\right|_2}=\frac{\nabla\widehat{f}_h(\widehat{\mathbf{y}}_s)}{\left|\left|\nabla\widehat{f}_h(\widehat{\mathbf{y}}_s)\right|\right|_2}," />

where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\left\{\widehat{\mathbf{y}}_s\right\}_{s=0,1,...}\subset\Omega_q" /> denotes the path of successive points defined by our directional mean shift iteration. This surprisingly incorporates the total gradient <img src="https://latex.codecogs.com/svg.latex?\Large&space;\nabla\widehat{f}_h" /> into our proposed algorithm. See Fig 1 below for a graphical illustration.

<p align="center">
<img src="https://github.com/zhangyk8/DirMS/blob/main/Figures/MS_One_Step1.png" style="zoom:20%" />
 <B>Fig 1. </B>Illustration of one-step iteration of our directional mean shift algorithm 
 </p>
 
The implementation of the directional kernel density estimation is encapsulated into the Python function called `DirKDE` in the script **DirMS_fun.py**, while the directional mean shift algorithm is implemented in the Python function `MS_DirKDE` in the same script. The input arguments of the function `MS_DirMS` essentially subsumes the ones of the function ``DirKDE``; thus, we combine the descriptions of their arguments as follows:
`def DirKDE(x, data, h=None)`
 
`def MS_DirKDE(y_0, data, h=None, eps=1e-7, max_iter=1000, diff_method='all')`
- Parameters: 
    - x or y_0: (m,d)-array or (N,d)-array
           ---- The Euclidean coordinates of m or N directional initial points in d-dimensional Euclidean space.
    - data: (n,d)-array
           ---- The Euclidean coordinates of n directional random sample points in d-dimensional Euclidean space.
    - h: float
           ---- The bandwidth parameter. (Default: h=None. Then a rule of thumb for directional KDEs with the von Mises kernel in Garcia-Portugues (2013) is applied.)
    - eps: float
           ---- The precision parameter for stopping the mean shift iteration. (Default: eps=1e-7)
    - max_iter: int
           ---- The maximum number of iterations for the mean shift iteration. (Default: max_iter=1000)
    - diff_method: str ('all'/'mean')
           ---- The method of computing the differences between two consecutive sets of iteration points when they are compared with the precision parameter to stop the algorithm. (When diff_method='all', all the differences between two consecutive sets of iteration points need to be smaller than 'eps' for terminating the algorithm. When diff_method='mean', only the mean difference is compared with 'eps' and stop the algorithm. Default: diff_method='all'.)
    
- Return:
    - MS_path: (N,d,T)-array
           ---- The whole iterative trajectory of every initial point yielded by the mean shift algorithm.
Usage:
 
```bash
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from Utility_fun import sph2cart, cart2sph, vMF_samp, Unique_Modes
from DirMS_fun import DirKDE, MS_DirKDE
 
## Set up the query points for computing the estimated densities
nrows, ncols = (100, 180)
lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), np.linspace(-90, 90, nrows))
xg, yg, zg = sph2cart(lon, lat)
query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                               yg.reshape(nrows*ncols, 1),
                               zg.reshape(nrows*ncols, 1)), axis=1)
   
np.random.seed(123)   ## Set a seed only for reproducibility
## Generate a vMF random sample with one mode
vMF_data1 = vMF_samp(1000, mu=np.array([1,0,0]), kappa=5)
lon1, lat1, R = cart2sph(*vMF_data1.T)
d_hat1 = DirKDE(query_points, vMF_data1).reshape(nrows, ncols)
## Perform the directional mean shift algorithm on the generated random sample
MS_path1 = MS_DirKDE(vMF_data1, vMF_data1, h=None, eps=1e-7, max_iter=1000)
# Compute the mode affiliations
num_ms_m1 = MS_path1.shape[2]-1
uni_ms_m1, uni_ms_m_lab1 = Unique_Modes(can_modes=MS_path1[:,:,num_ms_m1], tol=1e-2)

## Plotting the directional mean shift trajectory on the sphere
plt.figure(figsize=(14,12))
plt.subplot(221)
curr_step = 0
lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
# draw lat/lon grid lines every 30 degrees.
m2.drawmeridians(np.arange(-180, 180, 30))
m2.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = m2(lon, lat)
x4, y4 = m2(lon4, lat4)
# contour data over the map.
cs = m2.contourf(x, y, d_hat1)
cs = m2.scatter(x4, y4, color='red', s=10)
plt.title('Contours of directional KDE and mean shift: Step '+str(curr_step))

plt.subplot(222)
curr_step = (MS_path1.shape[2]-1)//3
lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
# draw lat/lon grid lines every 30 degrees.
m2.drawmeridians(np.arange(-180, 180, 30))
m2.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = m2(lon, lat)
x4, y4 = m2(lon4, lat4)
# contour data over the map.
cs = m2.contourf(x, y, d_hat1)
cs = m2.scatter(x4, y4, color='red', s=10)
plt.title('Contours of directional KDE and mean shift: Step '+str(curr_step))

plt.subplot(223)
curr_step = MS_path1.shape[2]-1
lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
# draw lat/lon grid lines every 30 degrees.
m2.drawmeridians(np.arange(-180, 180, 30))
m2.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = m2(lon, lat)
x4, y4 = m2(lon4, lat4)
# contour data over the map.
cs = m2.contourf(x, y, d_hat1)
cs = m2.scatter(x4, y4, color='red', s=10)
plt.title('Contours of directional KDE and mean shift: Step '+str(curr_step))

plt.subplot(224)
curr_step = 0
lon3, lat3, R = cart2sph(*MS_path1[:,:,curr_step].T)
m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
# draw lat/lon grid lines every 30 degrees.
m2.drawmeridians(np.arange(-180, 180, 30))
m2.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = m2(lon, lat)
x3, y3 = m2(lon3, lat3)
# contour data over the map.
cs = m2.contourf(x, y, d_hat1)
cs = m2.scatter(x3, y3, c=uni_ms_m_lab1, s=10, cmap=cm.cool)
## cmap=cm.cool
plt.title('Contours of directional KDE and affiliations of initial points')
plt.show()
 ```
<p align="center">
<img src="https://github.com/zhangyk8/DirMS/blob/main/Figures/Output.png" style="zoom:60%" />
 <B>Fig 2. </B>An example output of directional KDE and directional mean shift algorithm 
 </p>
 
 ### Blurring Directional Mean Shift Algorithm
 
 As the dimension of data or sample size increase, it is notable that the directional mean shift algorithm is slow. One possible method to accelerate the algorithm is to use a blurring procedure, where the dataset for density estimation is successively updated after each mean shift iteration. We implement this acceleration method with our directional mean shift algorithm in the Python function `MS_Blurring_DirKDE` following the stopping criterion in Carreira-Perpiñán (2006). However, the stopping criterion is designed for the (Gaussian) mean shift algorithm in the Euclidean data setting and may not be optimal to our directional mean shift algorithm. The refinement of the stopping criterion is left as a future direction.
 
 `def MS_Blurring_DirKDE(y_0, data, h=None, tol_1=1e-5, tol_2=1e-7, bins=None, max_iter=1000)`
 - Parameters: 
    - y_0: (m,d)-array or (N,d)-array
           ---- The Euclidean coordinates of m or N directional initial points in d-dimensional Euclidean space.
    - data: (n,d)-array
           ---- The Euclidean coordinates of n directional random sample points in d-dimensional Euclidean space.
    - h: float
           ---- The bandwidth parameter. (Default: h=None. Then a rule of thumb for directional KDEs with the von Mises kernel in Garcia-Portugues (2013) is applied.)
    - tol_1: float
           ---- The precision parameter for the mean location difference between two consecutive iteration sets of points. (Default: tol_1=1e-5)
    - tol_2: float
           ---- The stopping criterion for the entropy differences between two consecutive iteration sets of points. (Default: tol_2=1e-7)
    - bins: int
           ---- The number of bins for computing the entropy. (Default: bins=None. Then 'bins=int(np.floor(0.9*n))', where n is the number of random sample points.)
    - max_iter: int
           ---- The maximum number of iterations for the mean shift iteration. (Default: max_iter=1000)
    
- Return:
    - MS_path: (N,d,T)-array
           ---- The whole iterative trajectory of every initial point yielded by the blurring directional mean shift algorithm.
 
 Usage:
 ```bash
 import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from Utility_fun import sph2cart, cart2sph, vMF_samp, Unique_Modes
from DirMS_fun import DirKDE, MS_Blurring_DirKDE

nrows, ncols = (100, 180)
lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), np.linspace(-90, 90, nrows))
xg, yg, zg = sph2cart(lon, lat)

query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                              yg.reshape(nrows*ncols, 1),
                              zg.reshape(nrows*ncols, 1)), axis=1)

np.random.seed(123)  ## Set an arbitrary seed for reproducibility
vMF_data1 = vMF_samp(1000, mu=np.array([1,0,0]), kappa=5)
lon1, lat1, R = cart2sph(*vMF_data1.T)
d_hat1 = DirKDE(query_points, vMF_data1).reshape(nrows, ncols)

BMS_path1 = MS_Blurring_DirKDE(vMF_data1, vMF_data1, h=None, tol_1=1e-7, tol_2=1e-8, 
                               bins=800, max_iter=1000)
num_bms_m1 = BMS_path1.shape[2]-1
uni_bms_m1, uni_bms_m_lab1 = Unique_Modes(can_modes=BMS_path1[:,:,num_bms_m1], tol=1e-2)

## Plotting
plt.figure(figsize=(14,12))
plt.subplot(221)
curr_step = 0
lon4, lat4, R = cart2sph(*BMS_path1[:,:,curr_step].T)
m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
# draw lat/lon grid lines every 30 degrees.
m2.drawmeridians(np.arange(-180, 180, 30))
m2.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = m2(lon, lat)
x4, y4 = m2(lon4, lat4)
# contour data over the map.
cs = m2.contourf(x, y, d_hat1)
cs = m2.scatter(x4, y4, color='red', s=10)
plt.title('Contours of directional KDE and mean shift: Step '+str(curr_step))

plt.subplot(222)
curr_step = (BMS_path1.shape[2]-1)//3
lon4, lat4, R = cart2sph(*BMS_path1[:,:,curr_step].T)
m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
# draw lat/lon grid lines every 30 degrees.
m2.drawmeridians(np.arange(-180, 180, 30))
m2.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = m2(lon, lat)
x4, y4 = m2(lon4, lat4)
# contour data over the map.
cs = m2.contourf(x, y, d_hat1)
cs = m2.scatter(x4, y4, color='red', s=10)
plt.title('Contours of directional KDE and mean shift: Step '+str(curr_step))


plt.subplot(223)
curr_step = BMS_path1.shape[2]-1
lon4, lat4, R = cart2sph(*BMS_path1[:,:,curr_step].T)
m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
# draw lat/lon grid lines every 30 degrees.
m2.drawmeridians(np.arange(-180, 180, 30))
m2.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = m2(lon, lat)
x4, y4 = m2(lon4, lat4)
# contour data over the map.
cs = m2.contourf(x, y, d_hat1)
cs = m2.scatter(x4, y4, color='red', s=10)
plt.title('Contours of directional KDE and mean shift: Step '+str(curr_step)+' (Converged)')

plt.subplot(224)
curr_step = 0
lon3, lat3, R = cart2sph(*BMS_path1[:,:,curr_step].T)
m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
# draw lat/lon grid lines every 30 degrees.
m2.drawmeridians(np.arange(-180, 180, 30))
m2.drawparallels(np.arange(-90, 90, 30))
# compute native map projection coordinates of lat/lon grid.
x, y = m2(lon, lat)
x3, y3 = m2(lon3, lat3)
# contour data over the map.
cs = m2.contourf(x, y, d_hat1)
cs = m2.scatter(x3, y3, c=uni_bms_m_lab1, s=10, cmap=cm.cool)
## cmap=cm.cool
plt.title('Contours of directional KDE and affiliations of initial points')
plt.show()
 ```
 
 <p align="center">
<img src="https://github.com/zhangyk8/DirMS/blob/main/Figures/BMS_output.png" style="zoom:60%" />
 <B>Fig 3. </B>An example output of directional KDE and blurring directional mean shift algorithm 
 </p>
 
 ### Additional Reference
- M. Á. Carreira-Perpiñán. Fast nonparametric clustering with gaussian blurring mean-shift. _Proceedings of the 23rd International Conference on Machine Learning_, 2006:153–160, 01.
 
