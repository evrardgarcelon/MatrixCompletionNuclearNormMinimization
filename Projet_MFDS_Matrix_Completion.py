#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 11:52:28 2018

@author: Garcelon Evrard

"""
import numpy as np
import numpy.random as npr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.linalg import norm
from dr_nuclear_norm_minimization import dr_nuclear_norm_minimization
from matrix_factorization import matrix_factorization
from matrix_model import gaussian_model, SDP_model

from tqdm import tqdm


d1 = d2 = 50
d = max(d1,d2)

r = np.linspace(1,d-1,20,dtype = 'int')

P = np.linspace(1,d**2,20,dtype = 'int')

Omega = []

for p in P :
    
    I = npr.permutation(d1*d2)[:p]
    
    
    Omega.append(I[:p])


batch_size = 5
niter = 150

quantile = 0.25


mean_error = np.zeros((len(r),len(P)))



for i in tqdm(range(len(r))):
    
    for j in tqdm(range(len(P))) :
        
        omega = Omega[j]
        Phi  = lambda x: x[omega//d2,omega%d2]
        error_batch = []
        
        for _ in (range(batch_size)) :
            
            M = gaussian_model(d1,d2,r[i])
            y = Phi(M)
            x0 = np.zeros((d1,d2))
            x,_ = matrix_factorization(y, omega, niter, np.ones((d1,r[i])), np.ones((d2,r[i])), d1, d2, r[i], la = 10**-2)
            error_batch.append(norm(x-M)/norm(M))
            
        mean_error[i,j] = np.mean(np.array(error_batch))


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
r = r
P = P 
tsav = np.linspace(0,d-2,30,dtype = 'int')
tsav1,tsav2 = np.meshgrid(tsav,tsav)
r1,P1=r[tsav],P[tsav] 
rr,PP = np.meshgrid(r1,P1)
ax.scatter(rr,PP,mean_error[tsav1,tsav2],color = 'blue',marker = '^')
ax.set_xlabel('Rank')
ax.set_ylabel('Nb of observations')
ax.set_zlabel('Error') 
plt.show()


        

        
        
    

    

