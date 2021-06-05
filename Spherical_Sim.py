#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: April 23, 2021

Description: This script generates all the plots of our simulation studies in the 
spherical cases (Figures 1, 6, and 10 in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from Utility_fun import sph2cart, cart2sph, vMF_samp, Unique_Modes, vMF_samp_mix
from DirMS_fun import DirKDE, MS_DirKDE
import numpy.linalg as LA
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
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
    num_iter1 = MS_path1.shape[2]-1
    uni_ms_m1, uni_ms_m_lab1 = Unique_Modes(can_modes=MS_path1[:,:,num_iter1], 
                                            tol=1e-2)
    print("The misclassification rate of the one-mode simulation on the sphere is "\
          +str(np.mean(uni_ms_m_lab1))+".\n")
    
    plt.rcParams.update({'font.size': 20})  ## Change the font sizes of ouput figures
    print("Generating the plots for the spherical KDE with one mode and mean "\
          "shift iteration points at step 0, 19, and 79 in an orthographic view. \n")
    
    ## Generating the figure (step 0)
    fig=plt.figure(figsize=(8,8))
    curr_step = 0
    lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat1)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/MS_OneMode_Step0_ortho.pdf')
    
    ## Generating the figure (mid step)
    fig=plt.figure(figsize=(8,8))
    curr_step = (MS_path1.shape[2]-1)//4
    lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat1)
    cs = m2.scatter(x4, y4, color='red', s=50)
    fig.savefig('./Figures/MS_OneMode_Step19_ortho.pdf')
    
    ## Generating the figure (final step)
    fig=plt.figure(figsize=(8,8))
    curr_step = MS_path1.shape[2]-1
    lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat1)
    cs = m2.scatter(x4, y4, color='red', s=50)
    fig.savefig('./Figures/MS_OneMode_Step_conv_ortho.pdf')
    
    print("Save the plots as 'MS_OneMode_Step0_ortho.pdf', 'MS_OneMode_Step19_ortho.pdf',"\
          "and 'MS_OneMode_Step_conv_ortho.pdf'.\n")
    
    print("Generating the plots for the spherical KDE with one mode and mean "\
          "shift iteration points at step 0, 19, and 79 in a cylindrical equidistant "\
          "view. \n")
    
    fig=plt.figure(figsize=(14,8))
    curr_step = 0
    lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-150, urcrnrlon=150, 
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat1)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/MS_OneMode_Step0_cyl.pdf')
    
    fig=plt.figure(figsize=(14,8))
    curr_step = (MS_path1.shape[2]-1)//4
    lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-150, urcrnrlon=150,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat1)
    cs = m2.scatter(x4, y4, color='red', s=50)
    fig.savefig('./Figures/MS_OneMode_Step19_cyl.pdf')
    
    fig=plt.figure(figsize=(14,8))
    curr_step = MS_path1.shape[2]-1
    lon4, lat4, R = cart2sph(*MS_path1[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-150, urcrnrlon=150,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    # compute native map projection coordinates of lat/lon grid.
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat1)
    cs = m2.scatter(x4, y4, color='red', s=50)
    fig.savefig('./Figures/MS_OneMode_Step_conv_cyl.pdf')
    
    print("Save the plots as 'MS_OneMode_Step0_cyl.pdf', 'MS_OneMode_Step19_cyl.pdf',"\
          "and 'MS_OneMode_Step_conv_cyl.pdf'.\n")
    
    print("Generating the plot for affiliations of data points (one mode) after "\
          "mean shift clustering in a cylindrical equidistant view.\n")
    
    fig=plt.figure(figsize=(14,8))
    curr_step = 0
    lon3, lat3, R = cart2sph(*MS_path1[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-150, urcrnrlon=150,\
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x3, y3 = m2(lon3, lat3)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat1)
    cs = m2.scatter(x3, y3, c=uni_ms_m_lab1, s=50, cmap=cm.cool)
    fig.savefig('./Figures/MS_OneMode_MC_affi.pdf')
    
    print("Save the plot as 'MS_OneMode_MC_affi.pdf'.\n")
    
    ## Generate a vMF random sample with three local modes
    mode3 = np.array([[-120,-45], [0,60], [150,0]])   ## Local modes in degree
    x3, y3, z3 = sph2cart(*mode3.T)
    mu3 = np.concatenate((x3.reshape(3,1), y3.reshape(3,1), z3.reshape(3,1)), axis=1)
    
    np.random.seed(123)  ## Set a seed only for reproducibility
    kappa3 = [8.0, 8.0, 5.0]
    prob3 = [0.3, 0.3, 0.4]
    vMF_data3, true_lab = vMF_samp_mix(1000, mu=mu3, kappa=kappa3, prob=prob3, 
                                       label=True)
    lon3, lat3, R = cart2sph(*vMF_data3.T)
    d_hat3 = DirKDE(query_points, vMF_data3).reshape(nrows, ncols)
    
    ## Perform directional mean shift algorithm on the generated random sample
    MS_path3 = MS_DirKDE(vMF_data3, vMF_data3, h=None, eps=1e-7, max_iter=1000)
    # Compute the mode affiliations
    num_iter3 = MS_path3.shape[2]-1
    uni_ms_m3, uni_ms_m_lab3 = Unique_Modes(can_modes=MS_path3[:,:,num_iter3], 
                                            tol=1e-2)
    # Rearrange the assignment of cluster groups
    clu_lab = []
    for i in range(uni_ms_m3.shape[0]):
        clu_lab.append(np.argmin(LA.norm(uni_ms_m3[i,:] - mu3, axis=1)))
    uni_lab_adj = np.zeros_like(uni_ms_m_lab3)
    for i in range(uni_lab_adj.shape[0]):
        uni_lab_adj[i] = clu_lab[uni_ms_m_lab3[i]]
    
    print("The confusion matrix of mode clustering is ")
    print(confusion_matrix(true_lab, uni_lab_adj))
    print("\n")
    
    plt.rcParams.update({'font.size': 20})  ## Change the font sizes of ouput figures
    print("Generating the plots for the spherical KDE with three modes and mean "\
          "shift iteration points at step 0, 5, and 28 in an orthographic view. \n")

    fig=plt.figure(figsize=(8,8))
    curr_step = 0
    lon4, lat4, R = cart2sph(*MS_path3[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat3)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/MS_TripMode_Step0_ortho.pdf')
    
    fig=plt.figure(figsize=(8,8))
    curr_step = (MS_path3.shape[2]-1)//5
    lon4, lat4, R = cart2sph(*MS_path3[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat3)
    cs = m2.scatter(x4, y4, color='red', s=40)
    fig.savefig('./Figures/MS_TripMode_MidStep_ortho.pdf')
    
    fig=plt.figure(figsize=(8,8))
    curr_step = MS_path3.shape[2]-1
    lon4, lat4, R = cart2sph(*MS_path3[:,:,curr_step].T)
    m2 = Basemap(projection='ortho', lat_0=30, lon_0=0)
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30))
    m2.drawparallels(np.arange(-90, 90, 30))
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat3)
    cs = m2.scatter(x4, y4, color='red', s=80)
    fig.savefig('./Figures/MS_TripMode_Step_conv_ortho.pdf')
    
    print("Save the plots as 'MS_TripMode_Step0_ortho.pdf', 'MS_TripMode_MidStep_ortho.pdf',"\
          "and 'MS_TripMode_Step_conv_ortho.pdf'.\n")
    
    print("Generating the plots for the spherical KDE with three modes and mean "\
          "shift iteration points at step 0, 5, and 28 in a cylindrical equidistant "\
          "view. \n")
    
    fig=plt.figure(figsize=(14,8))
    curr_step = 0
    lon4, lat4, R = cart2sph(*MS_path3[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat3)
    cs = m2.scatter(x4, y4, color='red', s=30)
    fig.savefig('./Figures/MS_TripMode_Step0_cyl.pdf')
    
    fig=plt.figure(figsize=(14,8))
    curr_step = (MS_path3.shape[2]-1)//5
    lon4, lat4, R = cart2sph(*MS_path3[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat3)
    cs = m2.scatter(x4, y4, color='red', s=40)
    fig.savefig('./Figures/MS_TripMode_MidStep_cyl.pdf')
    
    fig=plt.figure(figsize=(14,8))
    curr_step = MS_path3.shape[2]-1
    lon4, lat4, R = cart2sph(*MS_path3[:,:,curr_step].T)
    m2 = Basemap(projection='cyl', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c')
    # draw lat/lon grid lines every 30 degrees.
    m2.drawmeridians(np.arange(-180, 180, 30), labels=[1,1,0,1])
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x4, y4 = m2(lon4, lat4)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat3)
    cs = m2.scatter(x4, y4, color='red', s=80)
    fig.savefig('./Figures/MS_TripMode_Step_conv_cyl.pdf')
    
    print("Save the plots as 'MS_TripMode_Step0_cyl.pdf', 'MS_TripMode_MidStep_cyl.pdf',"\
          "and 'MS_TripMode_Step_conv_cyl.pdf'.\n")
    
    print("Generating the plot for affiliations of data points (three modes) "\
          "after mean shift clustering in a Hammer projection view.\n")
    
    fig=plt.figure(figsize=(14,8))
    curr_step = 0
    lon3, lat3, R = cart2sph(*MS_path3[:,:,curr_step].T)
    m2 = Basemap(projection='hammer', llcrnrlon=-180, urcrnrlon=180,
                 llcrnrlat=-90, urcrnrlat=90, resolution='c', lon_0=0, lat_0=0)
    m2.drawparallels(np.arange(-90, 90, 30), labels=[1,1,0,1])
    x, y = m2(lon, lat)
    x3, y3 = m2(lon3, lat3)
    # contour data over the map.
    cs = m2.contourf(x, y, d_hat3)
    cs = m2.scatter(x3, y3, c=uni_ms_m_lab3, s=30, cmap=cm.cool)
    fig.savefig('./Figures/MS_TripMode_MC_affi.pdf')
    
    print("Save the plot as 'MS_TripMode_MC_affi.pdf'.\n")