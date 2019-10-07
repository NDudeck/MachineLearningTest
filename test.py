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
#data = fn1.trainY[:,6]
#print(fish_classify_5(data));
    
def sample(wave, factor):
    """Simulates sampling of a wave.
    
    wave: Wave object
    factor: ratio of the new framerate to the original
    """
    ys = np.zeros(len(wave))
    ys[::factor] = wave.ys[::factor]
    return thinkdsp.Wave(ys, framerate=wave.framerate) 

Cat_files = sorted(glob.glob('Cat_unedited*.wav'))
Dog_files = sorted(glob.glob('Dog_unedited*.wav'))
Cat_Test_files = sorted(glob.glob('Cat_Test*.wav'))
Dog_Test_files = sorted(glob.glob('Dog_Test*.wav'))
#TAKE OUT CAT ((19), (190), (195), (257), (268), (430), (471), (488), (792), (798), (799), (896)
#(1078) (1497), (1808), (1856), (1867), (1900), (1901)

#Cat_remove = [19,190,195,257,268,430,471,488,792,798,799,896, 1078, 1497, 1808, 1856, 1867, 1900, 1901]
#for i in range(0, len(Cat_remove)):
    #Cat_files = Cat_files.remove("Cat_unedited ("Cat_remove[i]").wav")
#TAKE OUT DOG (133), (235), (306), (457), (458), (466), (512), (513), (522), (632), (706)
#(811), (839), (1097), (1098), (1203), (1269), (1539), (1574)
#Dog_remove = [133, 235, 306, 457, 458, 512, 522, 632, 706, 811, 839, 1097, 1098, 1203, 1269, 1539, 1574]
j = np.empty((5, 4000))
traindataC = np.empty((2001,1501))
traindataD = np.empty((2001,1501))
Cat_Test = np.empty((len(Cat_Test_files), 1501))
Dog_Test = np.empty((8, 1501))
y1 = np.empty((8,1))
y2 = np.empty((len(Cat_Test_files),1))

#####SPECTRUM ANALYSIS###

for i in range(0, len(traindataC)):
    wave = thinkdsp.read_wave(Cat_files[i])
    spectrum = wave.make_spectrum(Cat_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataC[i] = b[8000:9501]
    
for i in range(0, len(traindataD)):
    wave = thinkdsp.read_wave(Dog_files[i])
    spectrum = wave.make_spectrum(Dog_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataD[i] = b[8000:9501]


for i in range(0, 8):
    wave = thinkdsp.read_wave(Dog_Test_files[i])
    spectrum = wave.make_spectrum(Dog_Test_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Dog_Test[i] = b[8000:9501]
    
for i in range(0, len(Cat_Test_files)):
    wave = thinkdsp.read_wave(Cat_Test_files[i])
    spectrum = wave.make_spectrum(Cat_Test_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Cat_Test[i] = b[8000:9501]
    

for i in range(0,len(traindataC)):
        if max(traindataC[i]) < 1:
            traindataC = np.delete(traindataC, i, 0)
            
for i in range(0,len(traindataD)):
        if max(traindataD[i]) < 1:
            traindataD = np.delete(traindataD, i, 0)

traindataD = np.delete(traindataD, [0], 0)
            
w1 = W.Word(traindataC.T);
w2 = W.Word(traindataD.T);
    


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
        return (2*B*np.matmul(fish2_Sb,w) - 2*B*np.matmul(fish2_Sw,w))/(A*A);
    
    res = sciop.minimize(fish2_J,np.ones((w1.rate,1)), jac = fish2_jac, \
                         options={'disp':True,'maxiter':25000}, tol=1e-10000)
    
    # Normalize W
    w = res.x;
    w = w.reshape((w1.rate,1))
    w_norm = w/np.linalg.norm(w)
    
    w_prop = np.matmul(np.linalg.inv(fish2_Sw),(w2.fish_m - w1.fish_m))
    w_prop_norm = w_prop/np.linalg.norm(w_prop);
    print(w_prop_norm - w_norm)
    
    return w_norm;


w = fish_classify_2_w();
for i in range(0, 8):
    y1[i] = np.matmul(w.T,Dog_Test[i]);
    y2[i] = np.matmul(w.T,Cat_Test[i]);

print(y1)
print(y2)

#%%

### 5 CLASS DATA TESTS ###
Cat_files = sorted(glob.glob('Cat_unedited*.wav'))
Dog_files = sorted(glob.glob('Dog_unedited*.wav'))
Stop_files = sorted(glob.glob('Stop_unedited*.wav'))
Go_files = sorted(glob.glob('Go_unedited*.wav'))
On_files = sorted(glob.glob('On_unedited*.wav'))

Cat_Test_files = sorted(glob.glob('Cat_Test*.wav'))
Dog_Test_files = sorted(glob.glob('Dog_Test*.wav'))
Stop_Test_files = sorted(glob.glob('Stop_Test*.wav'))
Go_Test_files = sorted(glob.glob('Go_Test*.wav'))
On_Test_files = sorted(glob.glob('On_Test*.wav'))

n = 2000

traindataC = np.empty((n,1501))
traindataD = np.empty((n,1501))
traindataS = np.empty((n,1501))
traindataG = np.empty((n,1501))
traindataO = np.empty((n,1501))

Cat_Test = np.empty((8,1501))
Dog_Test = np.empty((8,1501))
Stop_Test = np.empty((8,1501))
Go_Test = np.empty((8,1501))
On_Test = np.empty((8,1501))

for i in range(0, n):
    wave = thinkdsp.read_wave(Cat_files[i])
    spectrum = wave.make_spectrum(Cat_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataC[i] = b[8000:9501]
    
for i in range(0, n):
    wave = thinkdsp.read_wave(Dog_files[i])
    spectrum = wave.make_spectrum(Dog_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataD[i] = b[8000:9501]

for i in range(0, n):
    wave = thinkdsp.read_wave(Stop_files[i])
    spectrum = wave.make_spectrum(Stop_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataS[i] = b[8000:9501]
    
for i in range(0, n):
    wave = thinkdsp.read_wave(Go_files[i])
    spectrum = wave.make_spectrum(Go_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataG[i] = b[8000:9501]
    
for i in range(0, n):
    wave = thinkdsp.read_wave(On_files[i])
    spectrum = wave.make_spectrum(On_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    traindataO[i] = b[8000:9501]
    
for i in range(0, 8):
    wave = thinkdsp.read_wave(Dog_Test_files[i])
    spectrum = wave.make_spectrum(Dog_Test_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Dog_Test[i] = b[8000:9501]
    
for i in range(0, 8):
    wave = thinkdsp.read_wave(Cat_Test_files[i])
    spectrum = wave.make_spectrum(Cat_Test_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Cat_Test[i] = b[8000:9501]
    
for i in range(0, 8):
    wave = thinkdsp.read_wave(Stop_Test_files[i])
    spectrum = wave.make_spectrum(Stop_Test_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Stop_Test[i] = b[8000:9501]
    
for i in range(0, 8):
    wave = thinkdsp.read_wave(Go_Test_files[i])
    spectrum = wave.make_spectrum(Go_Test_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    Go_Test[i] = b[8000:9501]
    
for i in range(0, 8):
    wave = thinkdsp.read_wave(On_Test_files[i])
    spectrum = wave.make_spectrum(On_Test_files[i])
    spectrum.low_pass(1500)
    a = spectrum.render_full()
    b = a[1]
    On_Test[i] = b[8000:9501]
    
for i in range(0,len(traindataC)):
    if max(traindataC[i]) < 1:
        traindataC = np.delete(traindataC, i, 0)
for i in range(0,len(traindataD)):
    if max(traindataD[i]) < 1:
        traindataD = np.delete(traindataD, i, 0)
for i in range(0,len(traindataS)):
    if max(traindataS[i]) < 1:
        traindataS = np.delete(traindataS, i, 0)
for i in range(0,len(traindataG)):    
    if max(traindataG[i]) < 1:
        traindataG = np.delete(traindataG, i, 0)
for i in range(0,len(traindataO)):    
    if max(traindataO[i]) < 1:
        traindataO = np.delete(traindataO, i, 0)

while len(traindataC) > 2000:
    traindataC = np.delete(traindataC, [0], 0)
while len(traindataD) > 2000:        
    traindataD = np.delete(traindataD, [0], 0)
while len(traindataG) > 2000:
    traindataG = np.delete(traindataG, [0], 0)
while len(traindataO) > 2000:
    traindataO = np.delete(traindataO, [0], 0)
while len(traindataS) > 2000:
    traindataS = np.delete(traindataS, [0], 0)
#%%
w1 = W.Word(traindataC.T)
w2 = W.Word(traindataD.T)
w3 = W.Word(traindataS.T)
w4 = W.Word(traindataG.T)
w5 = W.Word(traindataO.T)

Y = np.zeros((5,40))

X = np.zeros((45,1501))

def fish_classify_5_w():
    
    w1.fish_class()
    w2.fish_class()
    w3.fish_class()
    w4.fish_class()
    w5.fish_class()

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
        return np.squeeze(((2*np.matmul(np.matmul(np.matmul(np.matmul(fish5_Sw,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))),np.matmul(np.matmul(w.T,fish5_Sb),w)),np.matmul(np.matmul(w.T,fish5_Sw),w)) \
                - 2*np.matmul(np.matmul(fish5_Sb,w),np.linalg.inv(np.matmul(np.matmul(w.T,fish5_Sw),w))))).reshape(1,7505))
    
    w0 = np.linspace(1,100,7505)
    
    res = sciop.minimize(fish5_J,w0,method = 'BFGS', jac = fish5_jac,options={'disp':True,'maxiter':250000}, tol=1e-100);
    print(res.x)
    w = res.x.reshape(1501,5)
    # Normalize W
    w = res.x;
    w = w.reshape((w1.rate,5))
    w_norm = w/np.linalg.norm(w)
    return w_norm;
    
w = fish_classify_5_w();

for i in range(0, 8):
    X[i] = Cat_Test[i]
    X[i+9] = Dog_Test[i]
    X[i+18] = Go_Test[i]
    X[i+27] = On_Test[i]
    X[i+36] = Stop_Test[i]
    
Y = np.matmul(w.T,X.T)
        

    
    
    
    
    
    
    
    
    
    
    
    
    