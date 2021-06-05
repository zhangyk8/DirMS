#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: April 23, 2021

Description: This script contains the code for computing misclassification rates 
of mode clustering with the directional mean shift algorithm on higher-dimensional 
spheres. (Figure 7 in the paper) (It may take more than an hour to execute, 
depending on the computing platform and CPU resources.)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ray
from DMS_Ray import DMS_Clu

if __name__ == "__main__":
    np.random.seed(123)  ## Set a seed only for reproducibility
    
    ray.init()
    result_ids = []
    Dim = [3,4,5,6,7,8,9,10,11,12]
    B = 20
    Dim_new = np.repeat(Dim, B)
    for i in range(len(Dim_new)):
        result_ids.append(DMS_Clu.remote(D=Dim_new[i], B=1))
    mis_rate_res = ray.get(result_ids)
    mis_rate_res = np.concatenate(mis_rate_res, axis=0)
    ray.shutdown()
    
    plt.rcParams.update({'font.size': 20})  ## Change the font sizes of ouput figures
    print("Generating the side-by-side boxplot of the misclassification rates "\
          "of mode clustering with the directional mean shift algorithm. \n")
    fig = plt.figure(figsize=(12,8))
    mis_rate_dict = {"Dimension q": Dim_new, 
                     "Misclassification Rate": mis_rate_res[:,0]}
    mis_rate_df = pd.DataFrame(mis_rate_dict)
    ax = sns.boxplot(x="Dimension q", y="Misclassification Rate", 
                     data=mis_rate_df)
    fig.savefig('./Figures/misrate_boxplot.pdf')