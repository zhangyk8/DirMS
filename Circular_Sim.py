#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: October 13, 2020

Description: This script generates all the plots of our simulation study in the 
circular case (Figure 5 in the paper).
"""

import matplotlib.pyplot as plt
import numpy as np
from Utility_fun import f1_samp, Unique_Modes
from DirMS_fun import DirKDE, MS_DirKDE


if __name__ == "__main__":
    np.random.seed(123)   ## Set a seed only for reproducibility
    kap = 6
    One_d_data = f1_samp(60, kappa=kap)
    One_d_data_ang = np.arctan2(One_d_data[:,1], One_d_data[:,0])
    
    ## Set up the query points for computing the estimated densities
    x = np.linspace(-np.pi, np.pi, 200)
    x_cart = np.concatenate((np.cos(x).reshape(len(x),1), 
                             np.sin(x).reshape(len(x),1)), axis=1)
    curr_bw = 0.3   ## Choose a smaller bandwidth parameter than the rule of thumb
    
     ## Perform the directional mean shift algorithm on the generated random sample
    MS_path1 = MS_DirKDE(One_d_data, One_d_data, h=curr_bw, eps=1e-7, max_iter=1000)
    # Compute the mode affiliations
    num_ms_m1 = MS_path1.shape[2]-1
    uni_ms_m1, uni_ms_m_lab1 = Unique_Modes(can_modes=MS_path1[:,:,num_ms_m1], 
                                            tol=1e-2)
    
    plt.rcParams.update({'font.size': 20})  ## Change the font sizes of ouput figures
    print("Generating the plots for the circular KDE on the 1-d circular random"\
          " sample and mean shift iteration points at step 0, 11, and 45. \n")
    
    fig=plt.figure(figsize=(8,6))
    plt.plot(x, DirKDE(x_cart, One_d_data, h=curr_bw), label="Circular KDE")
    plt.scatter(One_d_data_ang, DirKDE(One_d_data, One_d_data, h=curr_bw), 
                color="red", s=60)
    plt.legend()
    fig.savefig('./Figures/MS_1d_Step0.pdf')
    
    curr_s = (MS_path1.shape[2]-1)//4
    curr_path_ang = np.arctan2(MS_path1[:,1,curr_s], MS_path1[:,0,curr_s])
    fig=plt.figure(figsize=(8,6))
    plt.plot(x, DirKDE(x_cart, One_d_data, h=curr_bw), label="Circular KDE")
    plt.scatter(curr_path_ang, DirKDE(MS_path1[:,:,curr_s], One_d_data, h=curr_bw), 
                color="red", s=60)
    plt.legend()
    fig.savefig('./Figures/MS_1d_Step11.pdf')
    
    curr_s = (MS_path1.shape[2]-1)
    curr_path_ang = np.arctan2(MS_path1[:,1,curr_s], MS_path1[:,0,curr_s])
    fig=plt.figure(figsize=(8,6))
    plt.plot(x, DirKDE(x_cart, One_d_data, h=curr_bw), label="Circular KDE")
    plt.scatter(curr_path_ang, DirKDE(MS_path1[:,:,curr_s], One_d_data, h=curr_bw), 
                color="red", s=60)
    plt.legend()
    fig.savefig('./Figures/MS_1d_Step_conv.pdf')
    
    print("Save the plots as 'MS_1d_Step0.pdf', 'MS_1d_Step11.pdf', and "\
          "'MS_1d_Step_conv.pdf'.\n")
    
    print("Generating the plots for mean shift iteration points at step 0, 11, "\
          "and 45 on a unit circle. \n")
    
    t = np.linspace(0, 2*np.pi, 100)
    fig = plt.figure(figsize=(8,6))
    plt.plot(np.cos(t), np.sin(t), linewidth=1)
    curr_s = 0
    plt.scatter(MS_path1[:,0,curr_s], MS_path1[:,1,curr_s], color="red", s=60)
    plt.axis('equal')
    fig.savefig('./Figures/MS_1d_Step0_cir.pdf')
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(np.cos(t), np.sin(t), linewidth=1)
    curr_s = (MS_path1.shape[2]-1)//4
    plt.scatter(MS_path1[:,0,curr_s], MS_path1[:,1,curr_s], color="red", s=60)
    plt.axis('equal')
    fig.savefig('./Figures/MS_1d_Step11_cir.pdf')
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(np.cos(t), np.sin(t), linewidth=1)
    curr_s = (MS_path1.shape[2]-1)
    plt.scatter(MS_path1[:,0,curr_s], MS_path1[:,1,curr_s], color="red", s=60)
    plt.axis('equal')
    fig.savefig('./Figures/MS_1d_Step_conv_cir.pdf')
    
    print("Save the plots as 'MS_1d_Step0_cir.pdf', 'MS_1d_Step11_cir.pdf', "\
          "and 'MS_1d_Step_conv_cir.pdf'.\n")
    
    print("Generating the plots for affiliations of data points after mean "\
          "shift clustering.\n")
    
    fig=plt.figure(figsize=(8,6))
    plt.plot(x, DirKDE(x_cart, One_d_data, h=curr_bw), label="Circular KDE")
    plt.scatter(One_d_data_ang[uni_ms_m_lab1 == 0], 
                DirKDE(One_d_data[uni_ms_m_lab1 == 0,:], One_d_data, h=curr_bw), 
                c='red', marker='x', s=60)
    plt.scatter(One_d_data_ang[uni_ms_m_lab1 == 1], 
                DirKDE(One_d_data[uni_ms_m_lab1 == 1,:], One_d_data, h=curr_bw), 
                c='black', marker='o', s=60)
    plt.legend()
    fig.savefig('./Figures/MS_1d_MC.pdf')
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(np.cos(t), np.sin(t), linewidth=1)
    curr_s = 0
    plt.scatter(MS_path1[uni_ms_m_lab1 == 0, 0,curr_s], 
                MS_path1[uni_ms_m_lab1==0, 1,curr_s], 
                c='red', marker='x', s=60)
    plt.scatter(MS_path1[uni_ms_m_lab1 == 1, 0,curr_s], 
                MS_path1[uni_ms_m_lab1==1, 1,curr_s], 
                c='black', marker='o', s=60)
    plt.axis('equal')
    fig.savefig('./Figures/MS_1d_MC_cir.pdf')
    
    print("Save the plots as 'MS_1d_MC.pdf' and 'MS_1d_MC_cir.pdf'.\n")