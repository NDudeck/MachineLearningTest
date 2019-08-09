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
        
    
    def fish_class(self):
        
        #builds the things we need to classify for fisher 2 class
        
        self.fish_m = np.zeros((self.res,1));
        self.fish_m[:,0] = np.mean(self.trainY[:,0:5], axis=1);
            
        self.fish_Sw = np.zeros((self.res,self.res));
        for i in range(0,5):
            self.fish_Sw = self.fish_Sw + \
            np.matmul(self.trainY[:,i].reshape(self.res,1) - self.fish_m[:,0], \
            (self.trainY[:,i].reshape(self.res,1) - self.fish_m[:,0]).T)
        
    def fish_sb(self, m):
        return 5*np.matmul((self.fish_m - m),(self.fish_m - m).T)
    
    def fish5class(self):
        
        self.fish_K = 5     #number of classes
        self.fish_D = self.res   #number of dimensions
        self.fish_Dprime = 3 #number of features
        