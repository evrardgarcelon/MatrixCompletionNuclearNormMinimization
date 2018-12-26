#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:44:32 2018

@author: Garcelon Evrard
"""

import numpy as np
import numpy.random as npr

def gaussian_model(d1,d2,r) :   
    ''' Creates a matrix M of rank r as the product of two random gaussian matrix of size (d1,r) 
        and (r,d2)
            - d1 : the first dimension of M
            - d2 : the second dimension of M
            - r  : rank of M
            
    '''
    return np.dot(npr.randn(d1,r),npr.randn(r,d2))

def SDP_model(d1,r) :
    ''' Creates a matrix M of rank r as positive semi-definite matrix sampled from a gaussian matrix
        and (r,d2)
        
            - d1 : dimension of M
            - r  : rank of M  
    '''
    X = npr.randn(d1,r)
    return np.dot(X,X.T)

