#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: April 23, 2021

Description: This script contains the auxiliary code for computing 
misclassification rates of mode clustering with the directional mean shift 
algorithm on higher-dimensional spheres. 
"""

import ray
import numpy as np
from sklearn.metrics import confusion_matrix
from Utility_fun import Unique_Modes, vMF_samp_mix
from DirMS_fun import MS_DirKDE

@ray.remote
def DMS_Clu(D, B=50):
    mis_rate = np.zeros((B, 1))
    for j in range(B):
        kappa = [10.0, 10.0, 10.0, 10.0]
        prob = [0.25, 0.25, 0.25, 0.25]
        mu = np.eye(D+1)
        mu = mu[:4,:]
        vMF_data, true_lab = vMF_samp_mix(1000, mu=mu, kappa=kappa, 
                                          prob=prob, label=True)
        ## Perform directional mean shift algorithm on the generated random sample
        MS_path = MS_DirKDE(vMF_data, vMF_data, h=None, eps=1e-7, max_iter=5000)
        # Compute the mode affiliations
        num_ms_m = MS_path.shape[2]-1
        uni_ms_m, uni_ms_m_lab = Unique_Modes(can_modes=MS_path[:,:,num_ms_m], 
                                              tol=1e-2)
        conf_mat = confusion_matrix(true_lab, uni_ms_m_lab)
        mis_rate[j,0] = sum(np.max(conf_mat, axis=1))/1000
        
    return mis_rate