#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 21:57:40 2018

@author: Garcelon Evrard
"""

import numpy as np
from scipy.linalg import solve

def matrix_factorization(y, Omega, niter, H0, W0, d1, d2, r, la = 1, eps = 10**-10) :
    H,W = H0,W0
    iterates = [np.dot(H0,W0.T)]
    k = 0
    loss,old_loss = 0,1
    while k< niter and np.abs(loss-old_loss) > eps:
        old_loss = loss
        if k%2 == 0 :
            for i in range(d1) :
                temp1 = np.zeros(r)
                temp2 =  la*np.eye(r)
                for j in range(d2) :
                    if (d2*i + j in Omega) :
                        arg = np.argwhere(Omega == d2*i +j).squeeze()
                        temp1 += W[j]*y[arg]
                        temp2 += np.outer(W[j],W[j])
                H[i] = solve(temp2,temp1)
            
        else : 
            for j in range(d2) :
                temp1 = np.zeros(r)
                temp2 =  la*np.eye(r)
                for i in range(d1) :
                    if (d2*i + j in Omega) :
                        arg = np.argwhere(Omega == d2*i +j).squeeze()
                        temp1 += H[i]*y[arg]
                        temp2 += np.outer(H[i],H[i])
                W[j] = solve(temp2,temp1)
        k+=1
        loss = 0
        for j in range(d2) :
            loss += la*np.linalg.norm(W[j])**2 
            for i in range(d1) :
                loss += la*np.linalg.norm(H[i])**2 
                if (d2*i + j in Omega) :
                    arg = np.argwhere(Omega == d2*i +j).squeeze()
                    loss += ((y[arg] - np.dot(H[i],W[j]))**2)
        iterates.append(np.dot(H,W.T))
        
    return np.dot(H,W.T),np.array(iterates)
 
