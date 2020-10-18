#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: October 13, 2020

Description: This script generates the contour plots of a 2-von Mises-Fisher 
density and a mixture of 2-vMF densities (Figure 2 in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from Utility_fun import sph2cart, vMF_density


if __name__ == "__main__":
    ## Set up the query points for computing the vMF densities
    nrows, ncols = (100, 180)
    lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), np.linspace(-90, 90, nrows))
    xg, yg, zg = sph2cart(lon, lat)

    query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                                   yg.reshape(nrows*ncols, 1),
                                   zg.reshape(nrows*ncols, 1)), axis=1)
    
    print("Generating the contour plot for a 2-vMF density with mean (0,0,1) and "\
          "concentration parameter 4.0. \n")
    mu1 = np.array([[0,0,1]])
    kappa1 = [4.0]
    d_pts = vMF_density(query_points, mu=mu1, kappa=kappa1).reshape(nrows, ncols)
    
    fig = plt.figure(figsize=(8,8))
    # set up map projection
    m1 = Basemap(projection='ortho', lat_0=30, lon_0=15, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m1.drawmeridians(np.arange(-180, 180, 30))
    m1.drawparallels(np.arange(-90, 90, 30))
    # compute native map projection coordinates of lat/lon grid.
    x, y = m1(lon, lat)
    # contour data over the map.
    cs = m1.contourf(x, y, d_pts)
    fig.savefig('./Figures/vMF_density1.pdf')
    print("Save the plot as 'vMF_density1.pdf'\n")
    
    
    print("Generating the contour plot for a mixture of 2-vMF densities with "\
          "mean (0,0,1) & (1,0,0), concentration parameters 5.0 & 5.0, and mixture "\
          "probabilities 0.4 & 0.6. \n")
    mu2 = np.array([[0,0,1], [1,0,0]])
    kappa2 = [5.0, 5.0]
    prob2 = [0.4, 0.6]
    d_pts2 = vMF_density(query_points, mu=mu2, kappa=kappa2, prob=prob2).reshape(nrows, ncols)

    fig = plt.figure(figsize=(8,8))
    # set up map projection
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=15)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    # compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    # contour data over the map.
    cs = m2.contourf(x, y, d_pts2)
    fig.savefig('./Figures/vMF_dst_Mix.pdf')
    print("Save the plot as 'vMF_dst_Mix.pdf'\n")
