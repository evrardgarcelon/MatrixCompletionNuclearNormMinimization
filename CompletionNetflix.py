#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 01:11:10 2018

@author: Garcelon Evrard
"""

import numpy as np
import pandas as pd
import pylab as plt
from dr_nuclear_norm_minimization import dr_nuclear_norm_minimization
from preprocessing_data import preprocess_netflix

df_p,M,df_filterd,y_train,y_test,Omega_train,Omega_test,d1,d2= preprocess_netflix()


# Douglas-Rachford matrix completion
niter = 1
x0 = np.ones((d1,d2))
x2 = dr_nuclear_norm_minimization(x0, y_train, niter, Omega_train, mu = 1, gamma = 1)
predicted_M = np.around(x2[-1])
test_prediction = predicted_M[Omega_test//d2,Omega_test%d2]
rmse_dr = np.sqrt(np.mean((test_prediction - y_test)**2))
df_predicted = pd.DataFrame(data= predicted_M ,index=df_p.index, columns=df_p.columns)

y_pred = x2[:,Omega_test//d2,Omega_test%d2]
error_dr = np.sqrt(np.mean(y_pred - y_test,axis = 1)**2)

plt.loglog(error_dr,linewidth = 1,color = 'orange')
plt.xlabel('Iterations')
plt.ylabel('RMSE on the test set')


# Mean-rating prediction

mean_rating = (df_p.mean(axis = 0)).values
y_pred = mean_rating[Omega_test%d2]
rmse_mr = np.sqrt(np.mean((y_pred - y_test)**2))

print('RMSE for mean_rating :',rmse_mr, '\n RMSE for the Douglas-Rachford algorithm : ',rmse_dr)



#%% Top 10 movies 

