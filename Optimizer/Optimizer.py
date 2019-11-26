# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:49:31 2019

@author: bjt32
"""
#%%
import numpy as np
import math
import matplotlib.pyplot as plt

#%%
# global variables to customize

n = 9 #number of points sampled
m = 5 #degree of polynomial used -1
alpha = 10**-2 #stepsize for gradient
tol = 10**-3 #tolerance for gradient
max_iterations = 100000 #max number of iterations before termination
#if using linesearch, use these alphas
#alpha = np.empty((2,1))
#alpha[0,1] = 0 #initial alpha value
#alpha_max = 10**-2 #max step length
#p = -grad/np.linalg.norm(grad) #normalized direction of -gradient

# initializing variables used
t = np.empty((n,1))
w = np.reshape(np.random.normal(size = m),(m,1))
x = np.empty((n,1))
x0 = np.arange(n)
y_prime = np.empty((m,1))
z = np.empty((m,1))

#creating x, y, and t to be used
for i in range(0,n):
    t[i] = math.cos(math.pi*2*i/n)
    
for i in range(0,n):
    x[i] = t[i] + np.random.normal()/2


plt.plot(x0,t)
plt.plot(x0,x)
#%%
"""Setting up function and gradient"""

def error_fun(x,w,t):
    y = np.zeros((n,1))
    error = 0
    for i in range(0,n):
        for j in range(0,m):
            y[i] += w[j]*x[i]**j
    
    for i in range(0,n):
        error += (t[i]-y[i])**2
        
    return .5*error

def error_grad(x,w,t):
    error_grad = np.zeros((m,1))
    y = np.zeros((n,1))
    for i in range(0,n):
        for j in range(0,m):
            y[i] += w[j]*x[i]**j
    
    for i in range(0,m):
        for j in range(0,n):
            error_grad[i] += (y[j]-t[j])*x[j]**i
    
    return error_grad
#%%
"""Solving using steepest descent"""
grad = error_grad(x,w,t)
value = error_fun(x,w,t)
initial_grad = grad
initial_value = value
iterations = 0

while np.linalg.norm(grad) > tol and iterations <= max_iterations:
    for i in range(0,m):
        w[i] = w[i] - alpha*grad[i]
        
    value = error_fun(x,w,t)
    grad = error_grad(x,w,t)
    iterations += 1

y = np.zeros((n,1))
for i in range(0,n):
    for j in range(0,m):
        y[i] += w[j]*x[i]**j
            
plt.plot(x0,t)
plt.plot(x0,y)

#%%
"""Setting up linesearch"""
alpha = np.empty((2,1))
alpha[0] = 0 #initial alpha value
alpha[1] = np.random.rand(1)/10000
alpha_max = 1 #max step length
c1 = 10**-4
c2 = .8

def zoom(alpha):
    alpha_new = -grad*alpha[1]**2/(2(-value-grad*alpha[1])) #quadratic interpolation to find new alpha
    
    grad = error_grad(x,w,t)
    grad_step = error_grad(x,w + alpha_new*p,t)
    value = error_fun(x,w,t)
    value_step = error_fun(x,w + alpha_new*p,t)
    
    if value_step > value + c1*alpha_new*grad or value_step >= error_fun(x,w + alpha[0]*p,t):
        alpha[1] = alpha_new
        
    else:
        if np.linalg.norm(grad_step) <= -c2*grad:
            alpha_min = alpha_new
        if grad_step*(alpha[1]-alpha[0]) >= 0:
            alpha[1] = alpha[0]
        alpha[0] = alpha_new

def linesearch(alpha):
    
    while alpha[1] <= alpha_max:
        grad = error_grad(x,w,t)
        grad_step = error_grad(x,w + alpha[1]*p,t)
        value = error_fun(x,w,t)
        value_step = error_fun(x,w + alpha[1]*p,t)
        
        i = 1
        if np.all(value_step > error_fun(x,w,t) + c1*alpha[1]*grad)\
        or (np.all(value_step >= error_fun(x,w + alpha[0]*p,t)) and i >1):
            alpha_min = zoom(alpha[0],alpha[1]) #alpha_min is used for next iteration of eval
        elif np.linalg.norm(grad_step) <= np.linalg.norm(-c2*grad):
            alpha_min = alpha[1]
        elif np.all(grad_step >= 0):
            alpha_min = zoom(alpha[1],alpha[0])
        alpha[0] = alpha[1]
        alpha[1] = 5*alpha[1]
        i += 1
        
    return alpha_min

linesearch(alpha)
    
    










