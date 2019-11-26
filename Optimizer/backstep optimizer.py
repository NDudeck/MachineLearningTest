# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:38:42 2019

@author: bjt32
"""

#%%
"""import packages"""
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
"""initialize and change variables"""
max_iters = 10000
tol = 10**-6
alpha = np.zeros((max_iters,1))
value = np.zeros((max_iters,1))
gradient = np.zeros((max_iters,2))
x = np.zeros((max_iters,2))
alpha[0] = 1
x[0] = np.random.rand(2,1).T
its = 0
#%%
"""define functions used"""
def fun(y):
    value = math.sin(.5*y[0]*y[1])+math.cos(.25*y[0]*y[1]**2)
    return value

def grad(y):
    grad = np.empty((2,1))
    grad[0] = .5*y[1]*math.cos(.5*y[0]*y[1])-.25*y[1]**2*math.sin(.25*y[0]*y[1]**2)
    grad[1] = .5*y[0]*math.cos(.5*y[0]*y[1])-.5*y[0]*y[1]*math.sin(.25*y[0]*y[1]**2)
    return grad

#%%
"""mini"""
value[0] = fun(x[its,:])
gradient[0,:] = grad(x[its,:]).T
initial_value = value[0]
initial_grad = gradient[0,:]

while np.linalg.norm(gradient[its,:]) > tol and its < max_iters:
    for i in range(0,2):
        x[its+1,i] = x[its,i] - alpha[its]*gradient[its,i]
        
    its += 1    
    value[its] = fun(x[its,:])
    gradient[its] = grad(x[its,:]).T
    if value[its] >= value[its-1]:
        alpha[its] = alpha[its-1]/2
    else:
        alpha[its] = alpha[its-1]

final_value = value[its]
final_grad = gradient[its]
#%%
"""plot data"""
n = 50
x0 = np.zeros((n,1))
X = Y = np.empty((n,n))
z0 = np.zeros((n,1))
Z = np.empty((n**2,1))
x0[:,0] = np.linspace(-5,2,n)
X, Y = np.meshgrid(x0,x0)
R = np.append(X.reshape(n**2,1),Y.reshape(n**2,1),axis=1)

for i in range(0,n**2):
    Z[i] = fun(R[i,:])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,Z.reshape(n,n), alpha = .8)
ax = fig.gca(projection='3d')
ax.scatter(x[:,0],x[:,1],value,c = 'Green', s = 50)

plt.show()



        