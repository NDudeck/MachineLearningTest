 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:28:40 2019

@author: ndudeck
"""
#%%
import numpy as np;
import Function as F;
import Word as W;
import scipy.optimize as sciop
import matplotlib.pyplot as plt #This plots stuff
import thinkdsp
import glob
import scipy
import scipy.stats
import time


#%%
#Define functions as Function object
fn1 = F.Function(lambda x: np.exp(x));
fn2 = F.Function(lambda x: np.log(x+1));
fn3 = F.Function(lambda x: np.sin(x)*10);
fn4 = F.Function(lambda x: x**3);
fn5 = F.Function(lambda x: 10*x**.5);

fn1.fish_class();
fn2.fish_class();
fn3.fish_class();
fn4.fish_class();
fn5.fish_class();

def fish_classify_2_fn(data):  
    
    global res
    global fish2_Sw
    global fish2_Sb
    # Generate needed vars for 1of2 classify               
    fn1.fish_class();
    fn2.fish_class();
    
    fish2_Sw = fn1.fish_Sw + fn2.fish_Sw;
    fish2_Sb = np.matmul(fn2.fish_m - fn1.fish_m,\
         (fn2.fish_m - fn1.fish_m).T)
    
    # J(w) to optimize
    def fish2_J(w):
        return -np.matmul(np.matmul(w.T,fish2_Sb),w)/ \
                (np.matmul(np.matmul(w.T,fish2_Sw),w))
    
    def fish2_jac(w):
        A = 1/(np.matmul(np.matmul(w.T,fish2_Sw),w));
        B = np.matmul(np.matmul(w.T,fish2_Sb),w)
        return B*-1*A*2*np.matmul(fish2_Sw,w)*A + A*2*np.matmul(fish2_Sb,w);
        
    res = sciop.minimize(fish2_J,55*np.ones((100,1)),jac = fish2_jac, \
                         method = 'BFGS', \
                         options={'disp':True,'maxiter':25000}, tol=1e-10)
    
    # Normalize W
    w = res.x;
    w = w.reshape((fn1.res,1))
    w_norm = w/np.linalg.norm(w)
    
#    w_prop = np.matmul(np.linalg.inv(Sw),(fn2.fish2_m - fn1.fish2_m))
#    w_prop_norm = w_prop/np.linalg.norm(w_prop);
#    print(w_prop_norm - w_norm)
    
    a = np.matmul(w_norm.T,data.reshape(100,1));
    return a;

#data = fn1.trainY[:,1];
#print(fish_classify_2_fn(data));


def fish_classify_5(data):
    
    #Build Sw (4.40)
    fish5_Sw = fn1.fish_Sw + fn2.fish_Sw + fn3.fish_Sw + fn4.fish_Sw + fn5.fish_Sw;
    
    fish5_m =  (fn1.fish_m + fn2.fish_m + fn3.fish_m + fn4.fish_m + fn5.fish_m)/500;
    
    fish5_Sb = fn1.fish_sb(fish5_m) + fn2.fish_sb(fish5_m) + fn3.fish_sb(fish5_m) \
               + fn4.fish_sb(fish5_m) + fn5.fish_sb(fish5_m);
               
    def fish5_J(W):
        W = W.reshape(100,5);
        return -np.trace(np.matmul(np.linalg.inv(np.matmul(np.matmul(W.T,fish5_Sw),W)), np.matmul(np.matmul(W.T,fish5_Sb),W)))
  
    w0 = np.linspace(1,100,500);

    res = sciop.minimize(fish5_J,w0,method = 'BFGS', options={'disp':True,'maxiter':250000}, tol=1e-100);
    print(res.x)
    W = res.x.reshape(100,5)
    return np.matmul(W.T,data);
 
#%%
start = time.time()
### 5 CLASS DATA TESTS ###
Go_files = sorted(glob.glob('Go**/*.wav'))
On_files = sorted(glob.glob('On**/*.wav'))
Right_files = sorted(glob.glob('Right**/*.wav'))
Six_files = sorted(glob.glob('Six**/*.wav'))
Stop_files = sorted(glob.glob('Stop**/*.wav'))

n = 3200

traindataGo = np.empty((n,1501))
traindataOn = np.empty((n,1501))
traindataRight = np.empty((n,1501))
traindataSix = np.empty((n,1501))
traindataStop = np.empty((n,1501))

Go_Test = np.empty((100,1501))
On_Test = np.empty((100,1501))
Right_Test = np.empty((100,1501))
Six_Test = np.empty((100,1501))
Stop_Test = np.empty((100,1501))

for i in range(0, n):
    wave = thinkdsp.read_wave(Go_files[i])
    spectrum = wave.make_spectrum(Go_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataGo[i] = b[8000:9501]
    
for i in range(0, n):
    wave = thinkdsp.read_wave(On_files[i])
    spectrum = wave.make_spectrum(On_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataOn[i] = b[8000:9501]

for i in range(0, n):
    wave = thinkdsp.read_wave(Right_files[i])
    spectrum = wave.make_spectrum(Right_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataRight[i] = b[8000:9501]
    
for i in range(0, n):
    wave = thinkdsp.read_wave(Six_files[i])
    spectrum = wave.make_spectrum(Six_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataSix[i] = b[8000:9501]
    
for i in range(0, n):
    wave = thinkdsp.read_wave(Stop_files[i])
    spectrum = wave.make_spectrum(Stop_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataStop[i] = b[8000:9501]
    
for i in range(3300, 3400):
    wave = thinkdsp.read_wave(Go_files[i])
    spectrum = wave.make_spectrum(Go_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Go_Test[i-3300] = b[8000:9501]
    
for i in range(3300, 3400):
    wave = thinkdsp.read_wave(On_files[i])
    spectrum = wave.make_spectrum(On_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    On_Test[i-3300] = b[8000:9501]
    
for i in range(3300, 3400):
    wave = thinkdsp.read_wave(Right_files[i])
    spectrum = wave.make_spectrum(Right_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Right_Test[i-3300] = b[8000:9501]
    
for i in range(3300, 3400):
    wave = thinkdsp.read_wave(Six_files[i])
    spectrum = wave.make_spectrum(Six_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Six_Test[i-3300] = b[8000:9501]
    
for i in range(3300, 3400):
    wave = thinkdsp.read_wave(Stop_files[i])
    spectrum = wave.make_spectrum(Stop_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Stop_Test[i-3300] = b[8000:9501]

while len(traindataGo) > 3000:
    traindataGo = np.delete(traindataGo, [0], 0)
while len(traindataOn) > 3000:        
    traindataOn = np.delete(traindataOn, [0], 0)
while len(traindataRight) > 3000:
    traindataRight = np.delete(traindataRight, [0], 0)
while len(traindataSix) > 3000:
    traindataSix = np.delete(traindataSix, [0], 0)
while len(traindataStop) > 3000:
    traindataStop = np.delete(traindataStop, [0], 0)
    
for i in range(0,3000):
    for j in range(0,1501):
        if traindataGo[i,j] > 1000:
            traindataGo[i,j] = 0 
    
for i in range(0,3000):
    for j in range(0,1501):
        if traindataOn[i,j] > 1000:
            traindataOn[i,j] = 0 
            
for i in range(0,3000):
    for j in range(0,1501):
        if traindataRight[i,j] > 1000:
            traindataRight[i,j] = 0 
            
for i in range(0,3000):
    for j in range(0,1501):
        if traindataSix[i,j] > 1000:
            traindataSix[i,j] = 0 
            
for i in range(0,3000):
    for j in range(0,1501):
        if traindataStop[i,j] > 1000:
            traindataStop[i,j] = 0 
    
elapsed_time_DataMaking = (time.time() - start)

w1 = W.Word(traindataGo.T)
w2 = W.Word(traindataOn.T)
w3 = W.Word(traindataRight.T)
w4 = W.Word(traindataSix.T)
w5 = W.Word(traindataStop.T)
    
w1.fish_class()
w2.fish_class()
w3.fish_class()
w4.fish_class()
w5.fish_class()
    
elapsed_time_Classes = (time.time() - start) - elapsed_time_DataMaking
            
#%%
    
start = time.time()

w1 = W.Word(traindataGo.T);
w2 = W.Word(traindataOn.T);
    
Y = np.zeros((5,100))

X = np.zeros((500,1501))

def fish_classify_2_w():  
    
    # Generate needed vars for 1of2 classify               
    w1.fish_class();
    w2.fish_class();
    
    fish2_Sw = w1.fish_Sw + w2.fish_Sw;
    fish2_Sb = np.matmul(w2.fish_m - w1.fish_m,\
         (w2.fish_m - w1.fish_m).T)
    #filter Sb and Sw
    
    #may need to make each w[i] independent
#    w = {}
#    for i  in range(0,1500):
#        w["w" + str(i) ] = i
#    w = np.asarray(w)
    # J(w) to optimize
    def fish2_J(w):
        return -1*np.matmul(np.matmul(w.T,fish2_Sb),w)/ \
                (np.matmul(np.matmul(w.T,fish2_Sw),w))
    
    def fish2_jac(w):
        A = np.matmul(np.matmul(w.T,fish2_Sw),w);
        B = np.matmul(np.matmul(w.T,fish2_Sb),w)
        return -(A*2*np.matmul(fish2_Sb,w) - B*2*np.matmul(fish2_Sw,w))/(A*A)
    
    res = sciop.minimize(fish2_J,np.random.rand(1501,1), jac = fish2_jac, \
                         options={'disp':True,'maxiter':25000}, tol=1e-10000)
    print(res)
    # Normalize W
    w = res.x;
    w = w.reshape((w1.rate,1))
    w_norm = w/np.linalg.norm(w)
    
    w_prop = np.matmul(np.linalg.inv(fish2_Sw),(w2.fish_m - w1.fish_m))
    w_prop_norm = w_prop/np.linalg.norm(w_prop);
    print(w_prop_norm - w_norm)
    
    return w_norm;


w = fish_classify_2_w();

for i in range(0, 100):
    X[i] = Go_Test[i]
    X[i+100] = On_Test[i]
    X[i+200] = Right_Test[i]
    X[i+300] = Six_Test[i]
    X[i+400] = Stop_Test[i]
    
Y = np.matmul(w.T,X.T)

elapsed_time_fish2 = (time.time() - start)

#%%

start = time.time()

Y = np.zeros((5,100))

X = np.zeros((500,1501))

# 2 Class #

def fish_classify_5_2w():

    fish_mtotal = np.zeros((1501,1))
    fish_mtotal = w1.fish_m + w2.fish_m 
    fish_mtotal = fish_mtotal/(2)
    
    fish5_Sw = np.zeros((1501,1501))
    fish5_Sw = w1.fish_Sw + w2.fish_Sw
        
    fish5_Sb = np.zeros((1501,1501))
    fish5_Sb = np.matmul((w1.fish_m - fish_mtotal),(w1.fish_m - fish_mtotal).T) + \
               np.matmul((w2.fish_m - fish_mtotal),(w2.fish_m - fish_mtotal).T)
        
    def fish5_J(w):
        w = w.reshape(1501,2);
        return -np.trace(np.matmul(np.linalg.inv(np.matmul(\
                np.matmul(w.T,fish5_Sw),w)), np.matmul(np.matmul(w.T,fish5_Sb),w)))
  
    def fish5_jac(w):
        w = w.reshape(1501,2)
        return np.squeeze(((2*np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(fish5_Sw,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))),w.T),fish5_Sb),w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))) \
                - 2*np.matmul(np.matmul(fish5_Sb,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))))).reshape(1,2*1501))
        
    res = sciop.minimize(fish5_J,np.random.rand(2*1501,1),method = 'BFGS', jac = fish5_jac,options={'disp':True,'maxiter':250000}, tol=1e-100);
    print(res)
    w = res.x.reshape(1501,2)
    # Normalize W
    w = res.x;
    w = w.reshape((w1.rate,2))
    w_norm = w/np.linalg.norm(w)
    return w_norm;

# 3 Class #
    
def fish_classify_5_3w():

    fish_mtotal = np.zeros((1501,1))
    fish_mtotal = w1.fish_m + w2.fish_m + w3.fish_m 
    fish_mtotal = fish_mtotal/(3)
    
    fish5_Sw = np.zeros((1501,1501))
    fish5_Sw = w1.fish_Sw + w2.fish_Sw + w3.fish_Sw
        
    fish5_Sb = np.zeros((1501,1501))
    fish5_Sb = np.matmul((w1.fish_m - fish_mtotal),(w1.fish_m - fish_mtotal).T) + \
               np.matmul((w2.fish_m - fish_mtotal),(w2.fish_m - fish_mtotal).T) + \
               np.matmul((w3.fish_m - fish_mtotal),(w3.fish_m - fish_mtotal).T)
        
    def fish5_J(w):
        w = w.reshape(1501,3);
        return -np.trace(np.matmul(np.linalg.inv(np.matmul(\
                np.matmul(w.T,fish5_Sw),w)), np.matmul(np.matmul(w.T,fish5_Sb),w)))
  
    def fish5_jac(w):
        w = w.reshape(1501,3)
        return np.squeeze(((2*np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(fish5_Sw,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))),w.T),fish5_Sb),w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))) \
                - 2*np.matmul(np.matmul(fish5_Sb,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))))).reshape(1,3*1501))
        
    res = sciop.minimize(fish5_J,np.random.rand(3*1501,1),method = 'BFGS', jac = fish5_jac,options={'disp':True,'maxiter':250000}, tol=1e-100);
    print(res)
    w = res.x.reshape(1501,3)
    # Normalize W
    w = res.x;
    w = w.reshape((w1.rate,3))
    w_norm = w/np.linalg.norm(w)
    return w_norm;

# 4 Class #
    
def fish_classify_5_4w():
    
    fish_mtotal = np.zeros((1501,1))
    fish_mtotal = w1.fish_m + w2.fish_m + w3.fish_m + w4.fish_m
    fish_mtotal = fish_mtotal/(4)
    
    fish5_Sw = np.zeros((1501,1501))
    fish5_Sw = w1.fish_Sw + w2.fish_Sw + w3.fish_Sw + w4.fish_Sw
        
    fish5_Sb = np.zeros((1501,1501))
    fish5_Sb = np.matmul((w1.fish_m - fish_mtotal),(w1.fish_m - fish_mtotal).T) + \
               np.matmul((w2.fish_m - fish_mtotal),(w2.fish_m - fish_mtotal).T) + \
               np.matmul((w3.fish_m - fish_mtotal),(w3.fish_m - fish_mtotal).T) + \
               np.matmul((w4.fish_m - fish_mtotal),(w4.fish_m - fish_mtotal).T)
        
    def fish5_J(w):
        w = w.reshape(1501,4);
        return -np.trace(np.matmul(np.linalg.inv(np.matmul(\
                np.matmul(w.T,fish5_Sw),w)), np.matmul(np.matmul(w.T,fish5_Sb),w)))
  
    def fish5_jac(w):
        w = w.reshape(1501,4)
        return np.squeeze(((2*np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(fish5_Sw,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))),w.T),fish5_Sb),w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))) \
                - 2*np.matmul(np.matmul(fish5_Sb,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))))).reshape(1,4*1501))
        
    res = sciop.minimize(fish5_J,np.random.rand(4*1501,1),method = 'BFGS', jac = fish5_jac,options={'disp':True,'maxiter':250000}, tol=1e-100);
    print(res)
    w = res.x.reshape(1501,4)
    # Normalize W
    w = res.x;
    w = w.reshape((w1.rate,4))
    w_norm = w/np.linalg.norm(w)
    return w_norm;

# 5 Class #
    
def fish_classify_5_5w():

    fish_mtotal = np.zeros((1501,1))
    fish_mtotal = w1.fish_m + w2.fish_m + w3.fish_m + w4.fish_m + w5.fish_m
    fish_mtotal = fish_mtotal/(5)
    
    fish5_Sw = np.zeros((1501,1501))
    fish5_Sw = w1.fish_Sw + w2.fish_Sw + w3.fish_Sw + w4.fish_Sw + w5.fish_Sw
        
    fish5_Sb = np.zeros((1501,1501))
    fish5_Sb = np.matmul((w1.fish_m - fish_mtotal),(w1.fish_m - fish_mtotal).T) + \
               np.matmul((w2.fish_m - fish_mtotal),(w2.fish_m - fish_mtotal).T) + \
               np.matmul((w3.fish_m - fish_mtotal),(w3.fish_m - fish_mtotal).T) + \
               np.matmul((w4.fish_m - fish_mtotal),(w4.fish_m - fish_mtotal).T) + \
               np.matmul((w5.fish_m - fish_mtotal),(w5.fish_m - fish_mtotal).T)
        
    def fish5_J(w):
        w = w.reshape(1501,5);
        return -np.trace(np.matmul(np.linalg.inv(np.matmul(\
                np.matmul(w.T,fish5_Sw),w)), np.matmul(np.matmul(w.T,fish5_Sb),w)))
  
    def fish5_jac(w):
        w = w.reshape(1501,5)
        return np.squeeze(((2*np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(fish5_Sw,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))),w.T),fish5_Sb),w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))) \
                - 2*np.matmul(np.matmul(fish5_Sb,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))))).reshape(1,5*1501))
        
    res = sciop.minimize(fish5_J,np.random.rand(5*1501,1),method = 'BFGS', jac = fish5_jac,options={'disp':True,'maxiter':250000}, tol=1e-100);
    print(res)
    w = res.x.reshape(1501,5)
    # Normalize W
    w = res.x;
    w = w.reshape((w1.rate,5))
    w_norm = w/np.linalg.norm(w)
    return w_norm;
    
fish21 = fish_classify_5_2w();
#fish3 = fish_classify_5_3w();
#fish4 = fish_classify_5_4w();
#fish5 = fish_classify_5_5w();

for i in range(0, 100):
    X[i] = Go_Test[i]
    X[i+100] = On_Test[i]
    X[i+200] = Right_Test[i]
    X[i+300] = Six_Test[i]
    X[i+400] = Stop_Test[i]
    
Y21 = np.matmul(fish21.T,X.T)
#Y3 = np.matmul(fish3.T,X.T)
#Y4 = np.matmul(fish4.T,X.T)
#Y5 = np.matmul(fish5.T,X.T)

        
elapsed_time_fish5 = (time.time() - start)
    
#%%
Z2 = np.zeros((2,500))
Z3 = np.zeros((3,500))
Z4 = np.zeros((4,500))
Z5 = np.zeros((5,500))

Z2.T[np.arange(len(Y2.T)),Y2.T.argmax(1)] = 1
Z3.T[np.arange(len(Y3.T)),Y3.T.argmax(1)] = 1
Z4.T[np.arange(len(Y4.T)),Y4.T.argmax(1)] = 1
Z5.T[np.arange(len(Y5.T)),Y5.T.argmax(1)] = 1

    
    
    
    
    
    
    