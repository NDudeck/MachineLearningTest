#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:26:16 2019

@author: ndudeck

This script defines a function object. To construct a function object, it 
needs a lambda function to create.

"""
import numpy as np; #This does some math stuff  easily

class Function:
    
    #Any variable defined here will be shared by all Function objects
    interval = [0,3]  # 2x1 array 
    res = 100         # resolution of samples
    
    trainX = np.linspace(interval[0],interval[1],res)
    
    def __init__(self,equation):
        
        #Any variable defined here will only be with one Function object
        self.equation = equation; #lambda function
        
        #This function generates n evenly spaced points on the interval and 
        #adds noise. The result is an nx10 matrix with each column as a set of
        #data with increasing noise.
    
        self.trainY = np.zeros((self.res,10)).reshape(100,10); #Allocate
        self.trainY[:,0] = self.equation(self.trainX);
        for i in range(1,10):
            self.trainY[:,i] = self.trainY[:,i-1] + \
            np.random.normal(0,0.25,(self.res,1)).reshape(self.res,);
        
    
    def fish2class(self):
        
        #builds the things we need to classify for fisher 2 class
        
        self.fish2_m = np.zeros((self.res,1));
        for i in range(3,8):
            self.fish2_m[:,0] = self.fish2_m[:,0] + self.trainY[:,i];
            self.fish2_m = self.fish2_m/5
        
        self.fish2_Sw = np.zeros((self.res,self.res));
        for i in range(3,8):
            self.fish2_Sw = self.fish2_Sw + \
            np.matmul(self.trainY[:,i] - self.fish2_m[:,0], \
            np.transpose(self.trainY[:,i] - self.fish2_m[:,0]))
        