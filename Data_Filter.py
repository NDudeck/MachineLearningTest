# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:08:56 2019

@author: bjt32
"""
#%%
import os
from __future__ import print_function, division

import array
import copy
import math

import numpy as np
import random
import scipy
import scipy.stats
import scipy.fftpack
import struct
import subprocess
import thinkplot
import warnings
import glob
import thinkdsp

#%%
files = sorted(glob.glob('Right_unedited*.wav'))

n = len(files)

traindata = np.empty((n,1501))

for i in range(0, n):
    wave = thinkdsp.read_wave(files[i])
    spectrum = wave.make_spectrum(files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    if len(b) == 16000:
        traindata[i] = b[8000:9501]
    else:
        os.remove(files[i])
    
for i in range(0,len(traindata)):
    if max(traindata[i]) < 1:
        os.remove(files[i])
        