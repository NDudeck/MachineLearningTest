#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:28:40 2019

@author: ndudeck
"""
import numpy as np;
import Function as F;
import scipy.optimize as sciop

#Define functions as Function object
fn1 = F.Function(lambda x: np.exp(x));
fn2 = F.Function(lambda x: np.log(x+1));
                 
fn1.fish2class();
fn2.fish2class();

Sw = fn1.fish2_Sw + fn2.fish2_Sw;
Sb = np.matmul(fn2.fish2_m - fn1.fish2_m,\
     np.transpose(fn2.fish2_m - fn1.fish2_m))

def fish2_J(w):
    return -np.matmul(np.matmul(np.transpose(w),Sb),w)/ \
            (np.matmul(np.matmul(np.transpose(w),Sw),w))

res = sciop.minimize(fish2_J,np.ones((100,1)),method = 'BFGS', options={'disp':True,'maxiter':25000}, tol=1e-100)