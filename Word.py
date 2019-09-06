#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:26:16 2019

@author: ndudeck

This script defines a function object. To construct a function object, it 
needs a lambda function to create.

"""
import numpy as np; #This does some math stuff  easily

class Word:
    
    #Any variable defined here will be shared by all Function objects
    rate = 4000;        # resolution of samples
    sets = 1000;
    
    def __init__(self,data):
        
        #This function takes the input data and places it into the training matrix
        self.trainY = data;
        
    
    def fish_class(self):
        
        #builds the things we need to classify for fisher 2 class
        
        self.fish_m = np.zeros((self.rate,1));
        self.fish_m[:,0] = np.mean(self.trainY[:,0:self.sets-1], axis=1);
            
        self.fish_Sw = np.zeros((self.rate,self.rate));
        for i in range(0,self.sets):
            self.fish_Sw = self.fish_Sw + \
            np.matmul(self.trainY[:,i].reshape(self.rate,1) - self.fish_m[:,0], \
            (self.trainY[:,i].reshape(self.rate,1) - self.fish_m[:,0]).T)
        
    def fish_sb(self, m):
        return self.sets*np.matmul((self.fish_m - m),(self.fish_m - m).T)
    
#    def fish5class(self):
#        
#        self.fish_K = 5     #number of classes
#        self.fish_D = self.rate   #number of dimensions
#        self.fish_Dprime = 3 #number of features
        