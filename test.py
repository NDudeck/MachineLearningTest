#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:28:40 2019

@author: ndudeck
"""
import numpy as np;
import Function as F;
import scipy.optimize as sciop
#import matplotlib.pyplot as plt #This plots stuff

#Define functions as Function object
fn1 = F.Function(lambda x: np.exp(x));
fn2 = F.Function(lambda x: np.log(x+1));

def fish_classify_2():  
    
    # Generate needed vars for 1of2 classify               
    fn1.fish2class();
    fn2.fish2class();
    
    Sw = fn1.fish2_Sw + fn2.fish2_Sw;
    Sb = np.matmul(fn2.fish2_m - fn1.fish2_m,\
         np.transpose(fn2.fish2_m - fn1.fish2_m))
    
    # J(w) to optimize
    def fish2_J(w):
        return -np.matmul(np.matmul(np.transpose(w),Sb),w)/ \
                (np.matmul(np.matmul(np.transpose(w),Sw),w))
    
    grad = np.gradient(fish2_J)
    res = sciop.minimize(fish2_J,np.ones((100,1)),jac = grad, \
                         method = 'BFGS', \
                         options={'disp':True,'maxiter':25000}, tol=1e-10)
    
    # Normalize W
    w = res.x;
    w = w.reshape((fn1.res,1))
    w_norm = w/np.linalg.norm(w)
    
    #w_prop = np.matmul(np.linalg.inv(Sw),(fn2.fish2_m - fn1.fish2_m))
    #w_prop_norm = w_prop/np.linalg.norm(w_prop);
    #print(w_prop_norm - w_norm)
    
    a = np.matmul(np.transpose(w_norm),fn2.trainY[:,0].reshape(100,1));
    print(a)
    
fish_classify_2()




