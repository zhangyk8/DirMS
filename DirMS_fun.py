#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: October 14, 2020

Description: This script implements the main functions for the directional KDE,
directional mean shift algorithm, and blurring directional mean shift algorithm.
"""

import numpy as np
import pandas as pd
import scipy.special as sp
from collections import Counter
from scipy.stats import entropy


def DirKDE(x, data, h=None):
    '''
    q-dim directional KDE with the von Mises Kernel
    
    Parameters:
        x: (m,d)-array
            The Eulidean coordinates of m query points on a unit hypersphere, 
            where d=q+1 is the Euclidean dimension of data
    
        data: (n,d)-array
            The Euclidean coordinates of n directional random sample points in 
            d-dimensional Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then a rule of thumb for 
            directional KDEs with the von Mises kernel in Garcia-Portugues (2013)
            is applied.)
    
    Return:
        f_hat: (m,)-array
            The corresponding directinal density estimates at m query points.
    '''
    n = data.shape[0]  ## Number of data points
    d = data.shape[1]  ## Euclidean Dimension of the data

    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (d - R_bar ** 2) / (1 - R_bar ** 2)
        if d == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - 2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(d / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (d / 2) * (2 * (d - 1) * sp.iv(d/2, 2*kap_hat) + \
                                  (d+1) * kap_hat * sp.iv(d/2+1, 2*kap_hat)))) ** (1/(d + 3))
        print("The current bandwidth is " + str(h) + ".\n")
    
    f_hat = np.mean(np.exp(np.dot(x, data.T)/(h**2))/((2*np.pi)**(d/2)*\
                           sp.iv(d/2-1, 1/(h**2))*h**(d-2)), axis=1)
    return f_hat


def MS_DirKDE(y_0, data, h=None, eps=1e-7, max_iter=1000, diff_method='all'):
    '''
    Directional mean shift algorithm with the von-Mises Kernel
    
    Parameters:
        y_0: (N,d)-array
            The Euclidean coordinates of N directional initial points in 
            d-dimensional Euclidean space.
    
        data: (n,d)-array
            The Euclidean coordinates of n directional random sample points in 
            d-dimensional Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then a rule of thumb for 
            directional KDEs with the von Mises kernel in Garcia-Portugues (2013)
            is applied.)
        
        eps: float
            The precision parameter for stopping the mean shift iteration.
            (Default: eps=1e-7)
        
        max_iter: int
            The maximum number of iterations for the mean shift iteration.
            (Default: max_iter=1000)
            
        diff_method: str ('all'/'mean')
            The method of computing the differences between two consecutive sets
            of iteration points when they are compared with the precision 
            parameter to stop the algorithm. (When diff_method='all', all the 
            differences between two consecutive sets of iteration points need 
            to be smaller than 'eps' for terminating the algorithm. When 
            diff_method='mean', only the mean difference is compared with 'eps'
            and stop the algorithm. Default: diff_method='all'.)
    
    Return:
        MS_path: (N,d,T)-array
            The whole iterative trajectory of every initial point yielded by 
            the mean shift algorithm.
    '''

    n = data.shape[0]  ## Number of data points
    d = data.shape[1]  ## Euclidean dimension of the data

    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (d - R_bar ** 2) / (1 - R_bar ** 2)
        if d == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - \
                  2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(d / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (d / 2) * (2 * (d - 1) * sp.iv(d/2, 2*kap_hat) + \
                                  (d+1) * kap_hat * sp.iv(d/2+1, 2*kap_hat)))) ** (1/(d + 3))
        print("The current bandwidth is " + str(h) + ".\n")

    MS_path = np.zeros((y_0.shape[0], d, max_iter))
    MS_path[:,:,0] = y_0
    for t in range(1, max_iter):
        # y_can = np.dot(np.exp(-(1-np.dot(MS_path[:,:,t-1], data.T))/(h**2)), data)
        y_can = np.dot(np.exp(np.dot(MS_path[:,:,t-1], data.T/(h**2))), data)
        y_dist = np.sqrt(np.sum(y_can ** 2, axis=1))
        MS_path[:,:,t] = y_can / y_dist.reshape(len(y_dist), 1)
        if diff_method == 'mean' and \
        np.mean(1- np.diagonal(np.dot(MS_path[:,:,t], MS_path[:,:,t-1].T))) <=eps:
            break
        else:
            if all(1 - np.diagonal(np.dot(MS_path[:,:,t], MS_path[:,:,t-1].T)) <= eps):
                break       

    if t < max_iter:
        print('The directional mean shift algorithm converges in ' + str(t) + 'steps!')
    else:
        print('The directional mean shift algorithm reaches the maximum number '\
              'of iterations,' + str(max_iter) + ' and has not yet converged.')
    return MS_path[:,:,:(t+1)]


def MS_Blurring_DirKDE(y_0, data, h=None, tol_1=1e-5, tol_2=1e-7, bins=None, 
                       max_iter=1000):
    '''
    Blurring directional mean shift algorithm with the von Mises kernel
    
    Parameters:
        y_0: (N,d)-array
            The Euclidean coordinates of N directional initial points in 
            d-dimensional Euclidean space.
    
        data: (n,d)-array
            The Euclidean coordinates of n directional random sample points in 
            d-dimensional Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then a rule of thumb for 
            directional KDEs with the von Mises kernel in Garcia-Portugues (2013)
            is applied.)
       
        tol_1: float
            The precision parameter for the mean location difference between 
            two consecutive iteration sets of points. (Default: tol_1=1e-5)
    
        tol_2: float
            The stopping criterion for the entropy differences between two 
            consecutive iteration sets of points. (Default: tol_2=1e-7)
       
        bins: int
            The number of bins for computing the entropy. (Default: bins=None. 
            Then 'bins=int(np.floor(0.9*n))', where n is the number of random 
            sample points.)
       
        max_iter: int
            The maximum number of iterations for the mean shift algorithm.
    
    Return:
        MS_path: (N,d,T)-array
            The whole iterative trajectory of every initial point yielded by 
            the mean shift algorithm.
    '''
    n = data.shape[0]  ## Number of data points
    d = data.shape[1]  ## Euclidean dimension of the data

    ## Rule of thumb for directional KDE
    if h is None:
        R_bar = np.sqrt(sum(np.mean(data, axis=0) ** 2))
        ## An approximation to kappa (Banerjee 2005 & Sra, 2011)
        kap_hat = R_bar * (d - R_bar ** 2) / (1 - R_bar ** 2)
        if d == 3:
            h = (8*np.sinh(kap_hat)**2/(n*kap_hat * \
                 ((1+4*kap_hat**2)*np.sinh(2*kap_hat) - \
                  2*kap_hat*np.cosh(2*kap_hat))))**(1/6)
        else:
            h = ((4 * np.sqrt(np.pi) * sp.iv(d / 2 - 1, kap_hat)**2) / \
                 (n * kap_hat ** (d / 2) * (2 * (d - 1) * sp.iv(d/2, 2*kap_hat) + \
                                  (d+1) * kap_hat * sp.iv(d/2+1, 2*kap_hat)))) ** (1/(d + 3))
        print("The current bandwidth is " + str(h) + ".\n")
        
    if bins is None:
        bins = int(np.floor(0.9*n))
    
    MS_path = np.zeros((y_0.shape[0], d, max_iter))
    MS_path[:,:,0] = y_0
    curr_data = np.copy(data)
    curr_entropy = 0
    for t in range(1, max_iter):
        y_can = np.dot(np.exp(np.dot(MS_path[:,:,t-1], curr_data.T)/(h**2)), curr_data)
        y_dist = np.sqrt(np.sum(y_can**2, axis=1))
        MS_path[:,:,t] = y_can/y_dist.reshape(len(y_dist), 1)
        iter_error = 1- np.diagonal(np.dot(MS_path[:,:,t], MS_path[:,:,t-1].T))
        ## Compute the histogram of the differences between two consecutive 
        ## sets of iteration points.
        clu_freq = np.array(list(Counter(pd.cut(iter_error, bins=bins, 
                                                labels=False)).values()))
        clu_entropy = entropy(clu_freq/sum(clu_freq))
        if np.mean(iter_error) <= tol_1 or abs(clu_entropy - curr_entropy) <= tol_2:
            break
        else:
            curr_entropy = clu_entropy
            curr_data = MS_path[:,:,t]
            
    if t<max_iter:
        print('The blurring Mean Shift algorithm converges in ' + str(t) + 'steps!')
    else:
        print('The blurring Mean Shift algorithm reaches the maximum number of '\
              'iterations,'+str(max_iter)+' and has not yet converged.')
    return MS_path[:,:,:(t+1)]
