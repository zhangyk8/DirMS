#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: October 14, 2020

Description: This script runs the repeated experiments of the directional mean
shift algorithm and its blurring version on simulated vMF distributed data sets
with one, two, and three modes, repectively (Table 1 in the paper).  
"""

import numpy as np
import pandas as pd
from Utility_fun import vMF_samp, vMF_samp_mix, Unique_Modes
from DirMS_fun import MS_DirKDE, MS_Blurring_DirKDE


if __name__ == "__main__":
    ## Repeated experiments on vMF densities with one mode
    np.random.seed(123)  ## Set a seed only for reproducibility
    K = 20  ## The number of repeated times
    # List of storing the number of modes recovered by the directional mean shift (DMS)
    MS_m1_Num = []
    # List of storing the number of steps for the convergence of DMS
    MS_m1_step = []
    # List of storing the number of modes recovered by the blurring directional mean shift (BDMS)
    BMS_m1_Num = []
    # List of storing the number of steps for the convergence of BDMS
    BMS_m1_step = []
    # List of storing the average distance between modes yielded by DMS and 
    # the nearest modes yielded by BDMS
    Corr_avg_dist_m1 = []
    print("Repeated experiments on vMF densities with one mode.\n")
    for i in range(K):
        vMF_data1 = vMF_samp(1000, mu=np.array([1,0,0]), kappa=5)
        MS_path1 = MS_DirKDE(vMF_data1, vMF_data1, h=None, eps=1e-7, 
                             max_iter=1000, diff_method='mean')
        # Records the number of steps for MS
        num_ms_m1 = MS_path1.shape[2]-1
        MS_m1_step.append(num_ms_m1)
        # Compute the mode affiliations
        uni_ms_m1, uni_ms_m_lab1 = Unique_Modes(can_modes=MS_path1[:,:,num_ms_m1], 
                                                tol=1e-2)
        MS_m1_Num.append(uni_ms_m1.shape[0])
    
        # Records the number of steps for BMS
        BMS_path1 = MS_Blurring_DirKDE(vMF_data1, vMF_data1, h=None, tol_1=1e-7, 
                                       tol_2=1e-8, bins=None, max_iter=1000)
        num_bms_m1 = BMS_path1.shape[2]-1
        BMS_m1_step.append(num_bms_m1)
        uni_bms_m1, uni_bms_m_lab1 = Unique_Modes(can_modes=BMS_path1[:,:,num_bms_m1], 
                                                  tol=1e-2)
        BMS_m1_Num.append(uni_bms_m1.shape[0])
    
        Est_mode_err_lst = []
        for j in range(uni_ms_m1.shape[0]):
            # Average distances between estimated local modes and the nearest ones in BMS sets
            Est_mode_err_lst.append(min(np.sqrt(np.sum((uni_bms_m1 - uni_ms_m1[j,:])**2, 
                                                       axis=1))))
        Corr_avg_dist_m1.append(np.mean(np.array(Est_mode_err_lst)))
    
    print("\n DMS and BDMS in one mode case have been ending.......\n\n")
    ## Synthesize the results into a data frame
    One_mode_re = pd.DataFrame()
    One_mode_re['MS_mode_Num'] = MS_m1_Num
    One_mode_re['BMS_mode_Num'] = BMS_m1_Num
    One_mode_re['MS_Num_Step'] = MS_m1_step
    One_mode_re['BMS_Num_Step'] = BMS_m1_step
    One_mode_re['Est_Mode_Error_Dist_avg'] = Corr_avg_dist_m1
    
    print("The mean values of interested statistics (One Mode):\n")
    print(np.mean(One_mode_re, axis=0))
    print("The standard deviations of interested statistics (One Mode):\n")
    print(np.std(One_mode_re, axis=0))
    
    
    ## Repeated experiments on vMF densities with two modes
    np.random.seed(123)  ## Set a seed only for reproducibility
    K = 20  ## The number of repeated times
    # List of storing the number of modes recovered by the directional mean shift (DMS)
    MS_m2_Num = []
    # List of storing the number of steps for the convergence of DMS
    MS_m2_step = []
    # List of storing the number of modes recovered by the blurring directional 
    # mean shift (BDMS)
    BMS_m2_Num = []
    # List of storing the number of steps for the convergence of BDMS
    BMS_m2_step = []
    # List of storing the average distance between modes yielded by DMS and 
    # the nearest modes yielded by BDMS
    Corr_avg_dist_m2 = []
    print("\n\n Repeated experiments on vMF densities with two modes.\n")
    
    for i in range(K):
        mu2 = np.array([[0,1,0], [0,0,1]])
        kappa2 = [8.0, 6.0]
        prob2 = [0.4, 0.6]
        vMF_data2 = vMF_samp_mix(1000, mu=mu2, kappa=kappa2, prob=prob2)
    
        MS_path2 = MS_DirKDE(vMF_data2, vMF_data2, h=None, eps=1e-7, 
                             max_iter=1000, diff_method='mean')
        # Records the number of steps for MS
        num_ms_m2 = MS_path2.shape[2]-1
        MS_m2_step.append(num_ms_m2)
        # Compute the mode affiliations
        uni_ms_m2, uni_ms_m_lab2 = Unique_Modes(can_modes=MS_path2[:,:,num_ms_m2], 
                                                tol=1e-2)
        MS_m2_Num.append(uni_ms_m2.shape[0])
    
        # Records the number of steps for BMS
        BMS_path2 = MS_Blurring_DirKDE(vMF_data2, vMF_data2, h=None, tol_1=1e-7, 
                                       tol_2=1e-8, bins=None, max_iter=1000)
        num_bms_m2 = BMS_path2.shape[2]-1
        BMS_m2_step.append(num_bms_m2)
        uni_bms_m2, uni_bms_m_lab2 = Unique_Modes(can_modes=BMS_path2[:,:,num_bms_m2], 
                                                  tol=1e-2)
        BMS_m2_Num.append(uni_bms_m2.shape[0])
    
        Est_mode_err_lst = []
        for j in range(uni_ms_m2.shape[0]):
            # Average distances between estimated local modes and the nearest ones in BMS sets
            Est_mode_err_lst.append(min(np.sqrt(np.sum((uni_bms_m2 - uni_ms_m2[j,:])**2, 
                                                       axis=1))))
        Corr_avg_dist_m2.append(np.mean(np.array(Est_mode_err_lst)))
        
    print("\n DMS and BDMS in two modes case have been ending.......\n\n")
    ## Synthesize the results into a data frame
    Two_mode_re = pd.DataFrame()
    Two_mode_re['MS_mode_Num'] = MS_m2_Num
    Two_mode_re['BMS_mode_Num'] = BMS_m2_Num
    Two_mode_re['MS_Num_Step'] = MS_m2_step
    Two_mode_re['BMS_Num_Step'] = BMS_m2_step
    Two_mode_re['Est_Mode_Error_Dist_avg'] = Corr_avg_dist_m2
    
    print("The mean values of interested statistics (Two Modes):\n")
    print(np.mean(Two_mode_re, axis=0))
    print("The standard deviations of interested statistics (Two Modes):\n")
    print(np.std(Two_mode_re, axis=0))
    
    
    ## Repeated experiments on vMF densities with three modes
    np.random.seed(123)  ## Set a seed only for reproducibility
    K = 20  ## The number of repeated times
    # List of storing the number of modes recovered by the directional mean shift (DMS)
    MS_m3_Num = []
    # List of storing the number of steps for the convergence of DMS
    MS_m3_step = []
    # List of storing the number of modes recovered by the blurring directional 
    # mean shift (BDMS)
    BMS_m3_Num = []
    # List of storing the number of steps for the convergence of BDMS
    BMS_m3_step = []
    # List of storing the average distance between modes yielded by DMS and 
    # the nearest modes yielded by BDMS
    Corr_avg_dist_m3 = []
    print("\n\n Repeated experiments on vMF densities with three modes.\n")
    
    for i in range(K):
        mu3 = np.array([[0,1,0], [1,0,0], [0,-1,0]])
        kappa3 = [8.0, 6.0, 5.0]
        prob3 = [0.3, 0.4, 0.3]
        vMF_data3 = vMF_samp_mix(1000, mu=mu3, kappa=kappa3, prob=prob3)
    
        MS_path3 = MS_DirKDE(vMF_data3, vMF_data3, h=None, eps=1e-7, 
                             max_iter=1000, diff_method='mean')
        # Records the number of steps for MS
        num_ms_m3 = MS_path3.shape[2]-1
        MS_m3_step.append(num_ms_m3)
        # Compute the mode affiliations
        uni_ms_m3, uni_ms_m_lab3 = Unique_Modes(can_modes=MS_path3[:,:,num_ms_m3], 
                                                tol=1e-2)
        MS_m3_Num.append(uni_ms_m3.shape[0])
    
        # Records the number of steps for BMS
        BMS_path3 = MS_Blurring_DirKDE(vMF_data3, vMF_data3, h=None, tol_1=1e-7, 
                                       tol_2=1e-8, bins=None, max_iter=1000)
        num_bms_m3 = BMS_path3.shape[2]-1
        BMS_m3_step.append(num_bms_m3)
        uni_bms_m3, uni_bms_m_lab3 = Unique_Modes(can_modes=BMS_path3[:,:,num_bms_m3], 
                                                  tol=1e-2)
        BMS_m3_Num.append(uni_bms_m3.shape[0])
    
        Est_mode_err_lst = []
        for j in range(uni_ms_m3.shape[0]):
            Est_mode_err_lst.append(min(np.sqrt(np.sum((uni_bms_m3 - uni_ms_m3[j,:])**2, 
                                                       axis=1))))
        Corr_avg_dist_m3.append(np.mean(np.array(Est_mode_err_lst)))
        
    print("\n DMS and BDMS in three modes case have been ending.......\n\n")
    ## Synthesize the results into a data frame
    Three_mode_re = pd.DataFrame()
    Three_mode_re['MS_mode_Num'] = MS_m3_Num
    Three_mode_re['BMS_mode_Num'] = BMS_m3_Num
    Three_mode_re['MS_Num_Step'] = MS_m3_step
    Three_mode_re['BMS_Num_Step'] = BMS_m3_step
    Three_mode_re['Est_Mode_Error_Dist_avg'] = Corr_avg_dist_m3
    
    print("The mean values of interested statistics (Three Modes):\n")
    print(np.mean(Three_mode_re, axis=0))
    print("The standard deviations of interested statistics (Three Modes):\n")
    print(np.std(Three_mode_re, axis=0))