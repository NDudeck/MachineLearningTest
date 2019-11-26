# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:08:15 2019

@author: bjt32
"""
#%%
import numpy as np
import math
import matplotlib.pyplot as plt

#%%

tol = 10**-6
max_iterations = 100000
x = np.empty((2,1))
alpha = 0 #initial alpha value
alpha_max = 1 #max step length
c1 = 10**-4
c2 = .8
x = np.random.rand(2,1)

#%%
def rosen(x):
    
    value = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    
    return value

def rosen_grad(x):
    
    grad = 400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2
    
    return grad

#%%
"""Setting up linesearch"""

def zoom(alpha,beta):
    global alpha_min
    grad = rosen_grad(x) 
    value = rosen(x)
    p = -grad/np.linalg.norm(grad) #normalized direction of -gradient
    value_step = rosen(x + beta*p)

    alpha_new = -grad*beta**2/(2*(value_step-value-grad*beta)) #quadratic interpolation to find new alpha
    
    grad_step = rosen_grad(x + alpha_new*p)
    value_step = rosen(x + alpha_new*p)
    
    if value_step > value + c1*alpha_new*grad or value_step >= rosen(x + alpha*p):
        beta = alpha_new
        
    else:
        if np.linalg.norm(grad_step) <= -c2*grad:
            alpha_min = alpha_new
        if grad_step*(beta-alpha) >= 0:
            beta = alpha
        alpha = alpha_new

    return alpha_min

def linesearch(alpha):
    beta = np.random.rand(1)/10000
    alpha_min = 0
    global old_beta
    while beta <= alpha_max:
        grad = rosen_grad(x)
        value = rosen(x)
        p = -grad/np.linalg.norm(grad) #normalized direction of -gradient
        grad_step = rosen_grad(x + beta*p)
        value_step = rosen(x + beta*p)

        i = 1
        if value_step > rosen(x) + c1*beta*grad\
        or (value_step >= rosen(x + beta*p) and i >1):
            alpha_min = zoom(alpha,beta) #alpha_min is used for next iteration of eval
        elif np.linalg.norm(grad_step) <= np.linalg.norm(-c2*grad):
            alpha_min = beta
        elif np.all(grad_step >= 0):
            alpha_min = zoom(beta,alpha)
        old_beta = beta
        beta = 5*beta
        i += 1
        
    return alpha_min

#%%
"""Linesearch method"""

grad = rosen_grad(x)
value = rosen(x)
initial_grad = grad
initial_value = value
iterations = 0

#while np.linalg.norm(grad) > tol and iterations <= max_iterations:
for i in range(0,1):
    x[i] = x[i] - linesearch(alpha)*grad[i]
        
    
    value = rosen(x)
    grad = rosen_grad(x)
    iterations += 1


