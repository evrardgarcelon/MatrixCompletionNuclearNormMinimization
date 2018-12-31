#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 00:33:36 2018

@author: Garcelon Evrard
"""

import numpy as np
import numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.linalg import norm
from dr_nuclear_norm_minimization import dr_nuclear_norm_minimization
from matrix_factorization import matrix_factorization
from matrix_model import gaussian_model

from tqdm import tqdm

d1 = d2 = 50
d = max(d1,d2)

r = 4

P = int(2*d*np.log(d))
   
Omega = npr.permutation(d1*d2)[:P]    
    
batch_size = 25
niter = 300

quantile = 0.01

Phi  = lambda x: x[Omega//d2,Omega%d2]

error_batch_dr = np.zeros((batch_size,niter+1))
error_batch_mf = np.zeros((batch_size,niter+1))

for j in tqdm(range(batch_size)) :
    
    M = gaussian_model(d1,d2,r)
    y = Phi(M)
    x0 = np.zeros((d1,d2))
    _,x1 = matrix_factorization(y, Omega, niter, np.ones((d1,r)), np.ones((d2,r)), d1, d2, r, la = 10**-2,eps = -1)
    x2 = dr_nuclear_norm_minimization(x0, y, niter, Omega, mu = 1, gamma = 1,eps = -1)
    error_batch_mf[j,:] = norm(x1 - M,axis = (1,2))/norm(M)
    error_batch_dr[j,:] = norm(x2 - M,axis = (1,2))/norm(M)
#%%

mean_error_dr = np.mean(error_batch_dr,axis = 0)
mean_error_mf = np.mean(error_batch_mf,axis = 0)

q_quantile_dr = np.quantile(error_batch_dr, quantile, axis = 0)
Q_quantile_dr = np.quantile(error_batch_dr, 1 - quantile, axis = 0)

q_quantile_mf = np.quantile(error_batch_mf, quantile, axis = 0)
Q_quantile_mf = np.quantile(error_batch_mf, 1 - quantile, axis = 0)

   #%%
tsav_mf = np.linspace(0,niter+1,niter+1,dtype = 'int')
tsav_dr = np.linspace(0,niter+1,niter+1,dtype = 'int')
plt.figure(0)
plt.loglog(mean_error_dr,linewidth = 1.5,color = 'green',label = 'Nuclear Norm Minimization')
plt.fill_between(tsav_dr,q_quantile_dr,Q_quantile_dr,alpha = 0.15,color = 'green')

plt.figure(0)
plt.loglog(mean_error_mf,linewidth = 1.5,color = 'blue',label = 'Matrix Factorization')
plt.fill_between(tsav_mf,q_quantile_mf,Q_quantile_mf,alpha = 0.15,color = 'blue')

plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()

#%%

