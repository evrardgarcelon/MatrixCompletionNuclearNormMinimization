#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:13:51 2018

@author: Gaercelon Evrard
"""

import numpy as np
from scipy.linalg import svd

def dr_nuclear_norm_minimization(x0, y, niter, Omega, mu = 1, gamma = 1,eps = 10**-10) :
    
    d1,d2 = x0.shape
    
    Phi  = lambda x: x[Omega//d2,Omega%d2]
    PhiS = lambda y: np.bincount(Omega,weights = y,minlength=d1*d2).reshape((d1,d2))
    
    SoftThresh = lambda x, gamma: np.maximum(0, 1-gamma/np.maximum(abs(x), 1e-10))*x
    
    def SoftThreshDiag(a, b, c, gamma): 
        m,n = a.shape[0],c.shape[0]
        Sigma = np.zeros((m,n))
        np.fill_diagonal(Sigma,SoftThresh(b,gamma))
        return np.dot(np.dot(a,Sigma),c)
    
    def ProxG(x, gamma) : 
        U,s,V = svd(x)
        return SoftThreshDiag(U,s,V,gamma)
    
    rProxG = lambda x, gamma: 2*ProxG(x, gamma)-x
    
    ProxF = lambda x, gamma: x + PhiS(y - Phi(x))
    rProxF = lambda x, gamma: 2*ProxF(x, gamma)-x
    
    k = 0
    
    iterates = [x0]
    
    x0_tilde = x0
    
    x_old = x0 + np.ones(x0.shape)
    
    while k < niter and np.abs(np.linalg.norm(x0,'nuc') - np.linalg.norm(x_old,'nuc')) > eps:
        
        x0_tilde = (1 - mu/2)*x0_tilde + (mu/2)*rProxG(rProxF(x0_tilde,gamma),gamma)
        x_old = x0
        x0 = ProxF(x0_tilde,gamma)
        iterates.append(x0)
        if k%20 == 0 :
            print('k = ', k)
        k+=1
        
    
    iterates = iterates + [x0]*(niter - k)
        
    return np.array(iterates)
        
        
        
        
