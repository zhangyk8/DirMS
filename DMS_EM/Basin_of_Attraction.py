#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: January 15, 2021

Description: This script generates the plots of basins of attraction for 
directional mean shift algorithm (Figure 1 in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from Utility_fun import sph2cart, cart2sph, Unique_Modes, vMF_samp_mix
from DirMS_fun_Fast import MS_DirKDE_Fs, MS_DirKDE_L, L1_D

from multiprocessing import Pool
from functools import partial

if __name__ == "__main__":
    ## Set up the query points to which the directional mean shift algorithm applied
    nrows_f, ncols_f = (180, 360)
    lon_f, lat_f = np.meshgrid(np.linspace(-180, 180, ncols_f), 
                               np.linspace(-90, 90, nrows_f))
    xg, yg, zg = sph2cart(lon_f, lat_f)
    query_points_f = np.concatenate((xg.reshape(nrows_f*ncols_f, 1), 
                                     yg.reshape(nrows_f*ncols_f, 1),
                                     zg.reshape(nrows_f*ncols_f, 1)), axis=1)
    
    ## Generate a vMF random sample with three local modes
    mode3 = np.array([[-120,-45], [0,60], [150,0]])   ## Local modes in degree
    x3, y3, z3 = sph2cart(*mode3.T)
    mu3 = np.concatenate((x3.reshape(3,1), 
                          y3.reshape(3,1), 
                          z3.reshape(3,1)), axis=1)
    
    np.random.seed(123)  ## Set a seed only for reproducibility
    kappa3 = [8.0, 8.0, 5.0]
    prob3 = [0.3, 0.3, 0.4]
    vMF_data3 = vMF_samp_mix(1000, mu=mu3, kappa=kappa3, prob=prob3)
    lon3, lat3, R = cart2sph(*vMF_data3.T)
    
    ## Apply the directional mean shift with the von Mises kernel and cluster
    ## the initial points based on the local modes to which they converge
    with Pool(processes=100) as pool:
        chunksize = 100
        num_p = query_points_f.shape[0]
        part_fun = partial(MS_DirKDE_Fs, data=vMF_data3, h=None, eps=1e-8, 
                           max_iter=3000)
        MS_vM = pool.map(part_fun, 
                         [query_points_f[i:(i+chunksize)] \
                                         for i in range(0, num_p, chunksize)])
        MS_vM = np.concatenate(MS_vM, axis=0)

    uni_m_vM, uni_m_lab_vM = Unique_Modes(can_modes=MS_vM, tol=1e-3)
    
    with Pool(processes=100) as pool:
        chunksize = 100
        num_p = query_points_f.shape[0]
        part_fun = partial(MS_DirKDE_L, data=vMF_data3, h=None, L_D=L1_D, p=2, 
                           eps=1e-8, max_iter=3000)
        MS_L = pool.map(part_fun, 
                        [query_points_f[i:(i+chunksize)] \
                                        for i in range(0, num_p, chunksize)])
        MS_L = np.concatenate(MS_L, axis=0)

    uni_m_L, uni_m_lab_L = Unique_Modes(can_modes=MS_L, tol=1e-3)
    
    print("Generating the plots for the basins of attraction of directional mean "\
          "shift algorithms with the von Mises and a truncated convex kernel.\n")
    
    fig=plt.figure(figsize=(14,8))
    lon3, lat3, R = cart2sph(*query_points_f.T)
    m1 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,\
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m1.drawmeridians(np.arange(-180, 180, 30))
    m1.drawparallels(np.arange(-90, 90, 30))
    # compute native map projection coordinates of lat/lon grid.
    x3, y3 = m1(lon3, lat3)
    # contour data over the map.
    cs = m1.scatter(x3, y3, c=uni_m_lab_L, s=20)
    fig.savefig('./Figures/BoA_threeM_L.png')
    fig.savefig('./Figures/BoA_threeM_L.pdf')
    
    fig=plt.figure(figsize=(14,8))
    lon3, lat3, R = cart2sph(*query_points_f.T)
    m1 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,\
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m1.drawmeridians(np.arange(-180, 180, 30))
    m1.drawparallels(np.arange(-90, 90, 30))
    # compute native map projection coordinates of lat/lon grid.
    x3, y3 = m1(lon3, lat3)
    # contour data over the map.
    cs = m1.scatter(x3, y3, c=uni_m_lab_vM, s=20)
    fig.savefig('./Figures/BoA_threeM_vM.png')
    fig.savefig('./Figures/BoA_threeM_vM.pdf')
    
    print("Save the plots as 'BoA_threeM_L.png', 'BoA_threeM_L.pdf',"\
          "'BoA_threeM_vM.png', and 'BoA_threeM_vM.pdf'.\n")