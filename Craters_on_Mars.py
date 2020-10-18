#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: October 14, 2020

Description: This script generates all the plots of real-world applications on 
crater data on Mars (Figure 7 and 10 in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from Utility_fun import sph2cart, cart2sph, Unique_Modes
from DirMS_fun import DirKDE, MS_DirKDE


if __name__ == "__main__":
    # Load the crater data on Mars
    Mars = pd.read_csv('./Data/Mars_Craters.csv')
    # Filter out those craters with diameter less than or equal to 5 km
    Mars = Mars.loc[Mars['Diameter'] > 5,:]
    # Transform those craters whose areocentric longitudes are greater than 180 degree
    # back to the interval [-180, 0]
    Mars.loc[Mars['Center_Longitude'] >=180,'Center_Longitude'] = \
    Mars.loc[Mars['Center_Longitude'] >=180,'Center_Longitude'] -360
    print("The trimmed data set has the number of craters as " + str(Mars.shape[0]) + ".")
    
    # Convert the areocentric longitudes and latitudes into Cartesian coordinates 
    X, Y, Z = sph2cart(*Mars[['Center_Longitude', 'Center_Latitude']].values.T)
    Mars_cart = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)], 
                                axis=1)
    ## Set up the query points for computing the estimated densities
    nrows, ncols = (90, 180)
    lon, lat = np.meshgrid(np.linspace(-180, 180, ncols), np.linspace(-90, 90, nrows))
    xg, yg, zg = sph2cart(lon, lat)
    query_points = np.concatenate((xg.reshape(nrows*ncols, 1), 
                                   yg.reshape(nrows*ncols, 1),
                                   zg.reshape(nrows*ncols, 1)), axis=1)
    
    ## Estimated directional density
    d_Mars = DirKDE(query_points, Mars_cart).reshape(nrows, ncols)
    ## Perform the directional mean shift algorithm on the crater data on Mars
    MS_path_Mars = MS_DirKDE(Mars_cart, Mars_cart, h=None, eps=1e-7, max_iter=1000)
    # Compute the mode affiliations
    num_ms_m1 = MS_path_Mars.shape[2]-1
    uni_ms_m1, uni_ms_m_lab1 = Unique_Modes(can_modes=MS_path_Mars[:,:,num_ms_m1], 
                                            tol=1e-2)
    
    plt.rcParams.update({'font.size': 20})  ## Change the font sizes of ouput figures
    print("Generating the plots for the spherical KDE on the Martian crater data "\
          "and mean shift iteration points at step 0, 17, and 178 in an "\
          "orthographic view. \n")
    
    fig = plt.figure(figsize=(8,8))
    curr_step = 0
    lon4, lat4, R = cart2sph(*MS_path_Mars[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=-30)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_Mars)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/Mars_Step0_ortho.pdf')
    
    fig = plt.figure(figsize=(8,8))
    curr_step = (MS_path_Mars.shape[2]-1)//10
    lon4, lat4, R = cart2sph(*MS_path_Mars[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=-30)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_Mars)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/Mars_MidStep_ortho.pdf')
    
    fig = plt.figure(figsize=(8,8))
    curr_step = MS_path_Mars.shape[2]-1
    lon4, lat4, R = cart2sph(*MS_path_Mars[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=-30)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_Mars)
    cs = m2.scatter(x4, y4, color='red', s=60)
    fig.savefig('./Figures/Mars_conv_ortho.pdf')
    
    print("Save the plots as 'Mars_Step0_ortho.pdf', 'Mars_MidStep_ortho.pdf',"\
          "and 'Mars_conv_ortho.pdf'.\n")
    
    print("Generating the plots for the spherical KDE on the Martian crater data "
          "and mean shift iteration points at step 0, 17, and 178 in a "\
          "cylindrical equidistant view. \n")
    
    fig = plt.figure(figsize=(14,8))
    curr_step = 0
    lon4, lat4, R = cart2sph(*MS_path_Mars[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_Mars)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/Mars_Step0_cyl.pdf')
    
    fig = plt.figure(figsize=(14,8))
    curr_step = (MS_path_Mars.shape[2]-1)//10
    lon4, lat4, R = cart2sph(*MS_path_Mars[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_Mars)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/Mars_MidStep_cyl.pdf')
    
    fig = plt.figure(figsize=(14,8))
    curr_step = MS_path_Mars.shape[2]-1
    lon4, lat4, R = cart2sph(*MS_path_Mars[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_Mars)
    cs = m2.scatter(x4, y4, color='red', s=60)
    fig.savefig('./Figures/Mars_conv_cyl.pdf')
    
    print("Save the plots as 'Mars_Step0_cyl.pdf', 'Mars_MidStep_cyl.pdf',"\
          "and 'Mars_conv_cyl.pdf'.\n")
    
    print("Generating the plot for affiliations of craters on Mars after "\
          "mean shift clustering in a Hammer projection view.\n")
    fig = plt.figure(figsize=(14,8))
    curr_step = 0
    lon3, lat3, R = cart2sph(*MS_path_Mars[:,:,curr_step].T)
    m2 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0, lat_0=0)
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x3, y3 = m2(lon3, lat3)
    # contour data over the map.
    cs = m2.contourf(x, y, d_Mars)
    cs = m2.scatter(x3, y3, c=uni_ms_m_lab1, s=30, cmap='prism')
    fig.savefig('./Figures/Mars_MC_affi.pdf')
    print("Save the plot as 'Mars_MC_affi.pdf'.\n")
    
    ## Specify the bandwidth choices 
    curr_bw = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    cmap_color = ['prism', 'prism', 'nipy_spectral', 'hsv', 'hsv']
    
    print("Generating the plot for affiliations of craters on Mars after mode "\
          "clustering under various bandwidth choices in a Hammer projection view.\n")
    for i in range(len(curr_bw)):
        ## Estimated directional density
        d_Mars1 = DirKDE(query_points, Mars_cart, h=curr_bw[i]).reshape(nrows, ncols)
        ## Perform the directional mean shift algorithm 
        MS_path_Mars1 = MS_DirKDE(Mars_cart, Mars_cart, h=curr_bw[i], eps=1e-7, 
                                  max_iter=1000)
        # Compute the mode affiliations
        num_ms_m11 = MS_path_Mars1.shape[2]-1
        uni_ms_m11, uni_ms_m_lab11 = Unique_Modes(can_modes=MS_path_Mars1[:,:,num_ms_m11], 
                                                  tol=1e-2)
        print('The number of clusters under mode clustering with bandwidth h='\
              + str(curr_bw[i]) + ' is ' + str(uni_ms_m11.shape[0]) +'.\n')
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=(14,8))
        curr_step = 0
        lon3, lat3, R = cart2sph(*MS_path_Mars1[:,:,curr_step].T)
        m2 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,
                     llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0, lat_0=0)
        m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
        x, y = m2(lon, lat)
        x3, y3 = m2(lon3, lat3)
        # contour data over the map.
        cs = m2.contour(x, y, d_Mars1)
        if i < 5:
            cs = m2.scatter(x3, y3, c=uni_ms_m_lab11, s=30, cmap=cmap_color[i])
        else:
            cs = m2.scatter(x3, y3, c=uni_ms_m_lab11, s=30, cmap=cm.cool)
        fig.savefig('./Figures/Mars_MC_h0' + str(int(curr_bw[i]*100)) +'.pdf')
        print("Save the plot as 'Mars_MC_h0" + str(int(curr_bw[i]*100)) +".pdf'\t")
        