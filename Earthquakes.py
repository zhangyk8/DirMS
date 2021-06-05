#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: October 14, 2020

Description: This script generates all the plots of real-world applications on 
the earthquake data (Figure 9 in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from Utility_fun import sph2cart, cart2sph, Unique_Modes
from DirMS_fun import DirKDE, MS_DirKDE


if __name__ == "__main__":
    # Load the earthquake data
    Earthquakes = pd.read_csv('./Data/Earthquakes_08210921.csv')
    print("The data set has the number of recorded earthquakes as " \
          + str(Earthquakes.shape[0]) + ".")
    
    # Convert the longitudes and latitudes into Cartesian coordinates 
    X, Y, Z = sph2cart(*Earthquakes[['longitude', 'latitude']].values.T)
    EQ_cart = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)], 
                              axis=1)
    ## Set up the query points for computing the estimated densities
    nrows, ncols = (90, 180)
    lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), np.linspace(-90, 90, nrows))
    xg, yg, zg = sph2cart(lon, lat)
    query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                                   yg.reshape(nrows*ncols, 1),
                                   zg.reshape(nrows*ncols, 1)), axis=1)
    
    ## Estimated directional density
    d_EQ = DirKDE(query_points, EQ_cart).reshape(nrows, ncols)
    ## Perform the directional mean shift algorithm on the earthquake data
    MS_path_EQ = MS_DirKDE(EQ_cart, EQ_cart, h=None, eps=1e-7, max_iter=1000)
    # Compute the mode affiliations
    num_ms_m1 = MS_path_EQ.shape[2]-1
    uni_ms_m1, uni_ms_m_lab1 = Unique_Modes(can_modes=MS_path_EQ[:,:,num_ms_m1], 
                                            tol=1e-2)
    
    plt.rcParams.update({'font.size': 20})  ## Change the font sizes of ouput figures
    print("Generating the plots for the spherical KDE on the earthquake data "\
          "and mean shift iteration points at step 0, 10, and 82 in an "\
          "orthographic view. \n")
    
    fig = plt.figure(figsize=(8,8))
    curr_step = 0
    lon4, lat4, R = cart2sph(*MS_path_EQ[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=-80)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_EQ)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/EQ_Step0_ortho.pdf')
    
    fig=plt.figure(figsize=(8,8))
    curr_step = (MS_path_EQ.shape[2]-1)//8
    lon4, lat4, R = cart2sph(*MS_path_EQ[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=-80)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_EQ)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/EQ_MidStep_ortho.pdf')
    
    fig=plt.figure(figsize=(8,8))
    curr_step = MS_path_EQ.shape[2]-1
    lon4, lat4, R = cart2sph(*MS_path_EQ[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=-80)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_EQ)
    cs = m2.scatter(x4, y4, color='red', s=60)
    fig.savefig('./Figures/EQ_conv_ortho.pdf')
    
    print("Save the plots as 'EQ_Step0_ortho.pdf', 'EQ_MidStep_ortho.pdf',"\
          "and 'EQ_conv_ortho.pdf'.\n")
    
    print("Generating the plots for the spherical KDE on the earthquake data "
          "and mean shift iteration points at step 0, 10, and 82 in a "\
          "cylindrical equidistant view. \n")
    
    fig=plt.figure(figsize=(14,8))
    curr_step = 0
    lon4, lat4, R = cart2sph(*MS_path_EQ[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_EQ)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/EQ_Step0_cyl.pdf')
    
    fig=plt.figure(figsize=(14,8))
    curr_step = (MS_path_EQ.shape[2]-1)//8
    lon4, lat4, R = cart2sph(*MS_path_EQ[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_EQ)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/EQ_MidStep_cyl.pdf')
    
    fig=plt.figure(figsize=(14,8))
    curr_step = MS_path_EQ.shape[2]-1
    lon4, lat4, R = cart2sph(*MS_path_EQ[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_EQ)
    cs = m2.scatter(x4, y4, color='red', s=60)
    fig.savefig('./Figures/EQ_conv_cyl.pdf')
    
    print("Save the plots as 'EQ_Step0_cyl.pdf', 'EQ_MidStep_cyl.pdf',"\
          "and 'EQ_conv_cyl.pdf'.\n")
    
    print('Generating the plot of contour lines of the estimated density and '\
          'modes after mode clustering on the world map.\n')
    
    fig=plt.figure(figsize=(14,8))
    curr_step = MS_path_EQ.shape[2]-1
    lon3, lat3, R = cart2sph(*MS_path_EQ[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw coastlines, country boundaries, fill continents.
    m2.drawcoastlines(linewidth=0.25)
    m2.drawcountries(linewidth=0.25)
    m2.etopo(scale=0.5, alpha=0.1)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x3, y3 = m2(lon3, lat3)
    x, y = m2(lon, lat)
    cs = m2.scatter(x3, y3, color='red', s=80, marker='D')
    cs = m2.contour(x, y, d_EQ, linewidths=3, cmap='hsv')
    fig.savefig('./Figures/EQ_mode_graph.pdf')
    
    print("Save the plot as 'EQ_mode_graph.pdf'.\n")
    
    print('Generating the plot of earthquake affiliations after mode clustering'\
          ' on the world map.\n')
    
    fig=plt.figure(figsize=(14,8))
    curr_step = 0
    lon3, lat3, R = cart2sph(*MS_path_EQ[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
    # draw coastlines, country boundaries, fill continents.
    m2.drawcoastlines(linewidth=0.25)
    m2.drawcountries(linewidth=0.25)
    m2.etopo(scale=0.5, alpha=0.07)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    # compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x3, y3 = m2(lon3, lat3)
    cs = m2.scatter(x3, y3, c=uni_ms_m_lab1, s=40, cmap='Set1', alpha=10)
    fig.savefig('./Figures/EQ_MS_affi.pdf')
    
    print("Save the plot as 'EQ_MS_affi.pdf'.\n")
    
    