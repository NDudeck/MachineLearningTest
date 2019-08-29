#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:28:40 2019

@author: ndudeck
"""
import numpy as np;
import Function as F;
import Word as W;
import scipy.optimize as sciop
import matplotlib.pyplot as plt #This plots stuff

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
        
    res = sciop.minimize(fish2_J,np.ones((100,1)),jac = fish2_jac, \
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

#data = fn2.trainY[:,1];
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

wave = thinkdsp.read_wave('Cat1_unedited.wav')
spectrum = wave.make_spectrum('Cat1_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Cat1_edited.wav')
wave = thinkdsp.read_wave('Cat1_edited.wav')
sampled = sample(wave, 4)
Cat1_sampled = sampled.ys[0::4]
len(Cat1_sampled)

wave = thinkdsp.read_wave('Cat2_unedited.wav')
spectrum = wave.make_spectrum('Cat2_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Cat2_edited.wav')
wave = thinkdsp.read_wave('Cat2_edited.wav')
sampled = sample(wave, 4)
Cat2_sampled = sampled.ys[0::4]
len(Cat2_sampled)

wave = thinkdsp.read_wave('Cat3_unedited.wav')
spectrum = wave.make_spectrum('Cat3_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Cat3_edited.wav')
wave = thinkdsp.read_wave('Cat3_edited.wav')
sampled = sample(wave, 4)
Cat3_sampled = sampled.ys[0::4]
len(Cat3_sampled)

wave = thinkdsp.read_wave('Cat4_unedited.wav')
spectrum = wave.make_spectrum('Cat4_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Cat4_edited.wav')
wave = thinkdsp.read_wave('Cat4_edited.wav')
sampled = sample(wave, 4)
Cat4_sampled = sampled.ys[0::4]
len(Cat4_sampled)

wave = thinkdsp.read_wave('Cat5_unedited.wav')
spectrum = wave.make_spectrum('Cat5_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Cat5_edited.wav')
wave = thinkdsp.read_wave('Cat5_edited.wav')
sampled = sample(wave, 4)
Cat5_sampled = sampled.ys[0::4]
len(Cat5_sampled)

wave = thinkdsp.read_wave('Dog1_unedited.wav')
spectrum = wave.make_spectrum('Dog1_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Dog1_edited.wav')
wave = thinkdsp.read_wave('Dog1_edited.wav')
sampled = sample(wave, 4)
Dog1_sampled = sampled.ys[0::4]
len(Dog1_sampled)

wave = thinkdsp.read_wave('Dog2_unedited.wav')
spectrum = wave.make_spectrum('Dog2_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Dog2_edited.wav')
wave = thinkdsp.read_wave('Dog2_edited.wav')
sampled = sample(wave, 4)
Dog2_sampled = sampled.ys[0::4]
len(Dog2_sampled)

wave = thinkdsp.read_wave('Dog3_unedited.wav')
spectrum = wave.make_spectrum('Dog3_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Dog3_edited.wav')
wave = thinkdsp.read_wave('Dog3_edited.wav')
sampled = sample(wave, 4)
Dog3_sampled = sampled.ys[0::4]
len(Dog3_sampled)

wave = thinkdsp.read_wave('Dog4_unedited.wav')
spectrum = wave.make_spectrum('Dog4_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Dog4_edited.wav')
wave = thinkdsp.read_wave('Dog4_edited.wav')
sampled = sample(wave, 4)
Dog4_sampled = sampled.ys[0::4]
len(Dog4_sampled)

wave = thinkdsp.read_wave('Dog5_unedited.wav')
spectrum = wave.make_spectrum('Dog5_unedited.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Dog5_edited.wav')
wave = thinkdsp.read_wave('Dog5_edited.wav')
sampled = sample(wave, 4)
Dog5_sampled = sampled.ys[0::4]
len(Dog5_sampled)

wave = thinkdsp.read_wave('Dog_Test.wav')
spectrum = wave.make_spectrum('Dog_Test.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Dog_Test_edited.wav')
wave = thinkdsp.read_wave('Dog_Test_edited.wav')
sampled = sample(wave, 4)
Dog_Test_sampled = sampled.ys[0::4]
len(Dog_Test_sampled)

wave = thinkdsp.read_wave('Cat_Test.wav')
spectrum = wave.make_spectrum('Cat_Test.wav')
spectrum.low_pass(1500)
wave = spectrum.make_wave()
wave.play('Cat_Test_edited.wav')
wave = thinkdsp.read_wave('Cat_Test_edited.wav')
sampled = sample(wave, 4)
Cat_Test_sampled = sampled.ys[0::4]
len(Cat_Test_sampled)

traindata1 = 10000*np.vstack((Cat1_sampled,Cat2_sampled,Cat3_sampled,Cat4_sampled,Cat5_sampled))
traindata2 = 10000*np.vstack((Dog1_sampled,Dog2_sampled,Dog3_sampled,Dog4_sampled,Dog5_sampled))

w1 = W.Word(traindata1.T);
w2 = W.Word(traindata2.T);
    


def fish_classify_2_w():  
    
    # Generate needed vars for 1of2 classify               
    w1.fish_class();
    w2.fish_class();
    
    fish2_Sw = w1.fish_Sw + w2.fish_Sw;
    fish2_Sb = np.matmul(w2.fish_m - w1.fish_m,\
         (w2.fish_m - w1.fish_m).T)
    
    
    # J(w) to optimize
    def fish2_J(w):
        return -np.matmul(np.matmul(w.T,fish2_Sb),w)/ \
                (np.matmul(np.matmul(w.T,fish2_Sw),w))
    
    def fish2_jac(w):
        A = 1/(np.matmul(np.matmul(w.T,fish2_Sw),w));
        B = np.matmul(np.matmul(w.T,fish2_Sb),w)
        return B*-1*A*2*np.matmul(fish2_Sw,w)*A + A*2*np.matmul(fish2_Sb,w);
    
    res = sciop.minimize(fish2_J,np.ones((w1.rate,1)), jac = fish2_jac, \
                         options={'disp':True,'maxiter':25000}, tol=1e-10000)
    
    # Normalize W
    w = res.x;
    w = w.reshape((w1.rate,1))
    w_norm = w/np.linalg.norm(w)
    
#    w_prop = np.matmul(np.linalg.inv(Sw),(fn2.fish2_m - fn1.fish2_m))
#    w_prop_norm = w_prop/np.linalg.norm(w_prop);
#    print(w_prop_norm - w_norm)
    
    return w_norm;


data1 = 10000*Dog_Test_sampled
data2 = 10000*Cat_Test_sampled
w = fish_classify_2_w();
y1 = np.matmul(w.T,data1);
y2 = np.matmul(w.T,data2);

print(y1)
print(y2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    