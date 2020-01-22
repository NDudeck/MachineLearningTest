# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:59:05 2020

@author: bjt32
"""

import nummpy as np
import math

c1 = 
c2 = 
#%%
def rang(lamb, a, b):
    if lamb < a:
        return a
    elif lamb > b:
        return b
    else:
        return lamb

def cubic():
    
#%%
    
def refine():
    
    
    
    
    
    
    
    
    
    
def zoom():
    grad_flamb = gradf(x + lamb*p)*p
    if grad_flamb >= c2*grad_f0:
        return lamb
    elif lamb = 1:
        while lamb <= lamb_max:
            lamb_p, f_p = lamb, f_lamb
            lamb = 2*lamb
            f_lamb = f(x+lamb*p)
            if f_lamb > f0 + lamb*c1*grad_f0:
                lamb_p, f_p = lamb, f_lamb #MUST SWAP, NOT SET EQUAL
                refine()
            grad_flamb = gradf(x+lamb*p)*p
            if grad_flamb >= c2*grad_f0:
                return lamb
        return lamb_max
    
def linesearch(f,x,p,lamb_min):
    f0 = f(x)
    grad_f0 = gradf(x)*p
    lamb = 1
    while lamb >= lamb_min
        f_lamb = f(x + lamb*p)
        if f_lamb <= f0 + lamb*c1*grad_f0:
            zoom()
        else:
            if lamb = 1:
                lamb_temp = grad_f0/(2*(f0+grad_f0-f_lamb))
            else:
                cubic()
            lamb_p = lamb
            f_p = f_lamb
            lamb = range(lamb_temp, lamb/10,lamb/2)
    return lamb_min

