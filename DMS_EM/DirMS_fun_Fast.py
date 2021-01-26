#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang

Last Editing: January 15, 2021

Description: This script implements the directional KDE and directional mean 
shift algorithm with the von Mises and other general kernels in order to 
accommodate multi-processing iterations.
"""

import numpy as np
import scipy.special as sp


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
    
    if d == 3:
        f_hat = np.mean(np.exp((np.dot(x, data.T)-1)/(h**2))/(2*np.pi\
                        *(1-np.exp(-2/h**2))*h**2), axis=1)
    else:
        f_hat = np.mean(np.exp(np.dot(x, data.T)/(h**2))/((2*np.pi)**(d/2)*\
                           sp.iv(d/2-1, 1/(h**2))*h**(d-2)), axis=1)
    return f_hat


def MS_DirKDE_Fs(y_0, data, h=None, eps=1e-7, max_iter=1000):
    '''
    Directional mean shift algorithm with the von-Mises Kernel (that is adaptive 
    to multi-processing iterations)
    
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
    
    Return:
        MS_new: (N,d)-array
            The Euclidean coordinates of the final iterative points yielded by 
            the directional mean shift algorithm.
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

    MS_new = np.copy(y_0)
    for t in range(1, max_iter):
        MS_old = np.copy(MS_new)
        y_can = np.dot(np.exp((np.dot(MS_old, data.T)-1)/(h**2)), data)
        y_dist = np.sqrt(np.sum(y_can ** 2, axis=1))
        MS_new = y_can / y_dist.reshape(len(y_dist), 1)
        if all(1 - np.diagonal(np.dot(MS_new, MS_old.T)) <= eps):
            break       

    if t < max_iter:
        print('The directional mean shift algorithm converges in ' + str(t) + ' steps!')
    else:
        print('The directional mean shift algorithm reaches the maximum number '\
              'of iterations,' + str(max_iter) + ' and has not yet converged.')
    return MS_new


'''
------------------
The following two functions are coded for a family of truncated convex kernels 
and their derivatives.
'''

def L1(r,p=1):
    return ((1-r)**p) * (r <= 1) * (r >= 0)

def L1_D(r,p=1):
    return (-p*(1-r)**(p-1)) * (r <= 1) * (r >= 0)

'''
-----------------
'''

def DirKDE_L_prop(x, data, L=L1, p=1, h=None):
    '''
    The q-dim directional KDE with a truncated convex kernel (computed up to a 
    constant)
    
    Parameters:
        x: (m,d)-array
            The Eulidean coordinates of m query points on a unit hypersphere, 
            where d=q+1 is the Euclidean dimension of data.
    
        data: (n,d)-array
            The Euclidean coordinates of n directional random sample points in 
            d-dimensional Euclidean space.
        
        L: Pre-defined Python function
           The directional kernel function.
       
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
        
    f_hat = np.mean(L1((1-np.dot(x, data.T))/(h**2), p=p), axis=1)
    
    return f_hat


def MS_DirKDE_L(y_0, data, L_D=L1_D, p=1, h=None, eps=1e-7, max_iter=1000):
    '''
    Directional mean shift algorithm with a truncated convex kernel (that is 
    adaptive to multi-processing iterations)
    
    Parameters:
        y_0: (N,d)-array
            The Euclidean coordinates of N directional initial points in 
            d-dimensional Euclidean space.
    
        data: (n,d)-array
            The Euclidean coordinates of n directional random sample points in 
            d-dimensional Euclidean space.
        
        L_D: Pre-defined Python function
           The first-order derivative of the directional kernel.
       
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
    
    Return:
        MS_new: (N,d)-array
            The Euclidean coordinates of the final iterative points yielded by 
            the directional mean shift algorithm.
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

    MS_new = np.copy(y_0)
    for t in range(1, max_iter):
        MS_old = np.copy(MS_new)
        y_can = -np.dot(L_D((1-np.dot(MS_old, data.T))/(h**2), p=p), data)
        y_dist = np.sqrt(np.sum(y_can ** 2, axis=1))
        MS_new = y_can / y_dist.reshape(len(y_dist), 1)
        if all(1 - np.diagonal(np.dot(MS_new, MS_old.T)) <= eps):
            break       

    if t < max_iter:
        print('The directional mean shift algorithm converges in ' + str(t) + ' steps!')
    else:
        print('The directional mean shift algorithm reaches the maximum number '\
              'of iterations,' + str(max_iter) + ' and has not yet converged.')
    return MS_new
