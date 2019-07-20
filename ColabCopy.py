import scipy.optimize as sciop
import numpy as np #This does some math stuff  easily
import matplotlib.pyplot as plt #This plots stuff


class func:
  
  #This is called on creation
  def __init__(self,x_train, y_train):
    self.x_train = x_train
    self.y_train = y_train
    
    
  #Finds a polynomial regression of order k
  def lsq_fit(self,k):
    k = k+1
    self.lsq_A = np.zeros((k,k))
    self.lsq_b = np.zeros((k,1))
    
    #Populate A
    for i in range(k):
      for j in range(k):
        self.lsq_A[i][j] = np.sum((self.x_train)**(i+j))
    
    #Populate b
    for i in range(k):
      self.lsq_b[i][0] = np.sum(((self.y_train)*((self.x_train)**i)))
      
    #Solve Ax=b for x   
    self.lsq_x = np.linalg.solve(self.lsq_A,self.lsq_b)
    
    #Create array of powers of x
    self.lsq_p = np.zeros((k,1))
    self.lsq_y = np.linspace(0,3,5000)
    for i in range(self.x.size):
      for j in range(k):
        self.lsq_y[i] = self.lsq_y[i] + (self.lsq_x[j]*(self.x[i]**j))
   
  #TODO add functions that create more noise and put into array   


#The number of sample points
n = 100 #@param {type:"slider", min:0, max:5000, step:1}

#sample function here
#Define functions
fn1 = lambda x: np.exp(x)
fn2 = lambda x: np.sin(x)*10
fn3 = lambda x: np.log(x+1)
fn4 = lambda x: x**3
fn5 = lambda x: 10*x**.5

#Create training set off fn's

fn1_trainX = np.linspace(0,3,n)
fn1_trainY = np.zeros((10,n))
fn1_trainY[0] = fn1(fn1_trainX)
for i in range(1,10):
  for j in range(n):
    fn1_trainY[i][j] = fn1_trainY[i-1][j] + np.random.normal()
for i in range(3):
  plt.scatter(fn1_trainX,fn1_trainY[i])
#plt.show()


fn2_trainX = np.linspace(0,3,n)
fn2_trainY = np.zeros((10,n))
fn2_trainY[0] = fn2(fn2_trainX)
for i in range(1,10):
  for j in range(n):
    fn2_trainY[i][j] = fn2_trainY[i-1][j] + np.random.normal()
for i in range(3):
  plt.scatter(fn2_trainX,fn2_trainY[i])
#plt.show()


fn3_trainX = np.linspace(0,3,n)
fn3_trainY = np.zeros((10,n))
fn3_trainY[0] = fn3(fn3_trainX)
for i in range(1,10):
  for j in range(n):
    fn3_trainY[i][j] = fn3_trainY[i-1][j] + np.random.normal()
for i in range(3):
  plt.scatter(fn3_trainX,fn3_trainY[i])
#plt.show()


fn4_trainX = np.linspace(0,3,n)
fn4_trainY = np.zeros((10,n))
fn4_trainY[0] = fn4(fn4_trainX)
for i in range(1,10):
  for j in range(n):
    fn4_trainY[i][j] = fn4_trainY[i-1][j] + np.random.normal()
for i in range(3):
  plt.scatter(fn4_trainX,fn4_trainY[i])
#plt.show()

fn5_trainX = np.linspace(0,3,n)
fn5_trainY = np.zeros((10,n))
fn5_trainY[0] = fn5(fn5_trainX)
for i in range(1,10):
  for j in range(n):
    fn5_trainY[i][j] = fn5_trainY[i-1][j] + np.random.normal()
for i in range(3):
  plt.scatter(fn5_trainX,fn5_trainY[i])
plt.show()
    
#Pass training set to func
func1 = func(fn1_trainX,fn1_trainY)
#func2 = func(fn2_trainX,fn2_trainY)
func3 = func(fn3_trainX,fn3_trainY)
#func4 = func(fn4_trainX,fn4_trainY)
#func5 = func(fn5_trainX,fn5_trainY)

#Classify the data
def lsq_classify():
    
    #Build R,T
    lsq_n = 100; #The number of noisy data sets
    lsq_D = n;   #The number of points
    lsq_R = np.zeros((lsq_n,lsq_D));
    lsq_T = np.zeros((lsq_n,5))
    
    for i in range(0,lsq_n//5):
      lsq_R[i] = fn1_trainY[9] + np.random.normal(size = lsq_D);
      lsq_T[i][0] = 1;
    for i in range(lsq_n//5, 2*lsq_n//5):
      lsq_R[i] = fn2_trainY[9] + np.random.normal(size = lsq_D);
      lsq_T[i][1] = 1;
    for i in range(2*lsq_n//5,3*lsq_n//5):
      lsq_R[i] = fn3_trainY[9] + np.random.normal(size = lsq_D);
      lsq_T[i][2] = 1;
    for i in range(3*lsq_n//5, 4*lsq_n//5):
      lsq_R[i] = fn4_trainY[9] + np.random.normal(size = lsq_D);
      lsq_T[i][3] = 1;
    for i in range(4*lsq_n//5,lsq_n):
      lsq_R[i] = fn5_trainY[9] + np.random.normal(size = lsq_D);
      lsq_T[i][4] = 1;
      
      #Find W
      lsq_W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(lsq_R),lsq_R)),np.transpose(lsq_R)),lsq_T);
      #print(lsq_W)
      return lsq_W;

#give function and percentages
t=150
count1 = 0 
count2 = 0 
count3 = 0 
count4 = 0 
count5 = 0
for _ in range(0,t):
    Warr = np.transpose(lsq_classify());
    r_in = fn3_trainY[9];
    y_c = np.matmul(Warr,r_in);
    if np.argmax(y_c) == 0:
        count1 = count1 + 1
    elif np.argmax(y_c) == 1:
        count2 = count2 + 1
    elif np.argmax(y_c) == 2:
        count3 = count3 + 1
    elif np.argmax(y_c) == 3:
        count4 = count4 + 1
    elif np.argmax(y_c) == 4:
        count5 = count5 + 1

fnc = np.argmax([count1,count2,count3,count4,count5])+1
print("you're function is ", fnc)

print("Function 1: ", count1/t*100, "%")
print("Function 2: ", count2/t*100, "%")
print("Function 3: ", count3/t*100, "%")
print("Function 4: ", count4/t*100, "%")
print("Function 5: ", count5/t*100, "%")

#TODO
##Run multiple times and pick one that's highest % (output percents)


def fish_classify():
  #Define variables
  fish_N1 = n; #number of points in C1
  fish_N2 = n;
  
  fish_n = 50; #number of xn vectors
  
  fish_X1 = np.zeros((fish_N1,fish_n))
  fish_X2 = np.zeros((fish_N2,fish_n))
  
  fish_m1 = np.zeros((fish_N1,1))
  fish_m2 = np.zeros((fish_N2,1))
  fish_m = np.zeros((fish_N1,1))
  fish_mdiff = np.zeros((fish_N1,1))
  
  fish_Sw1 = np.zeros((fish_N1,fish_N1))
  fish_Sw2 = np.zeros((fish_N2,fish_N2))
  fish_Sw = np.zeros((fish_N1,fish_N1))
  fish_Sb = np.zeros((fish_N1,fish_N1))
    
  #Create noisy data (similar to lsq)
  fish_X1 = np.transpose(fish_X1);
  for i in range(0,fish_n):
    fish_X1[i] = fn1_trainY[9] + np.random.normal(size = fish_N1);
  fish_X1 = np.transpose(fish_X1);
  
  fish_X2 = np.transpose(fish_X2);
  for i in range(0,fish_n):
    fish_X2[i] = fn2_trainY[9] + np.random.normal(size = fish_N2);
  fish_X2 = np.transpose(fish_X2);
  
  #Find m1, m2
  for i in range(0,fish_N1):
    fish_m1[i] = np.average(fish_X1[i]);
    
  for i in range(0,fish_N2):
    fish_m2[i] = np.average(fish_X2[i]);
 
  #Build Sw
  for i in range(0,fish_n):
    fish_Sw1_diff = fish_X1[:,i] - fish_m1
    fish_Sw1 = fish_Sw1 + np.matmul(fish_Sw1_diff,np.transpose(fish_Sw1_diff))
  
  for i in range(0,fish_n):
    fish_Sw2_diff = fish_X2[:,i] - fish_m2
    fish_Sw2 = fish_Sw2 + np.matmul(fish_Sw2_diff,np.transpose(fish_Sw2_diff))
  
  fish_Sw = fish_Sw1 + fish_Sw2
  
  fish_mdiff = fish_m2 - fish_m1
  fish_Sb = np.matmul(fish_mdiff,np.transpose(fish_mdiff))
  
  fish_W = np.matmul(np.linalg.inv(fish_Sw),(fish_m2-fish_m1))
  
  fish_m = ((fish_N1*fish_m1)+(fish_N2*fish_m2))//(fish_N1+fish_N2)
  fish_Y = np.transpose(fish_W)
  fish_Z = fn1_trainY[9].reshape((n,1)) - fish_m
  res = np.matmul(fish_Y,fish_Z)
  
  #print(fish_Sb)
  #print(fish_Sw)
 
  def fish_J(w):
    return -np.matmul(np.matmul(np.transpose(w),fish_Sb),w)/(np.matmul(np.matmul(np.transpose(w),fish_Sw),w))
  
  #grad = np.gradient(fish_J, np.ones((n,1)))
  
  #res = sciop.minimize(fish_J,np.ones((n,1))/n,method = 'BFGS', options={'disp':True,'maxiter':25000}, tol=1e-100, jac = grad)
  
  #print(res)
  #w = res.x;
  #w = w.reshape((n,1))
  #w = w/np.linalg.norm(w)
  #print(w_prop_norm - w_norm)
  #print(grad)
  #print(fish_X1[:,1].size)
  
  
        
fish_classify();
#BFGS or SQSLP
#optimize J(w)
#normalize W
#normalize w and w_prop and take difference (should be 0s)
#Flatten/Unflatten functions
#Unflatten -> Matrix algebra -> return -> optimize -> unflatten
#Pass gradient as input for 2-case
#and 5-case and flatten
#input sound (from keyboard) and fft to visualize


#Define variables
fish_N = 100 #number of points in C1

fish_n = 50 #number of xn vectors

fish_X1 = np.zeros((fish_N,fish_n))
fish_X2 = np.zeros((fish_N,fish_n))
fish_X3 = np.zeros((fish_N,fish_n))
fish_X4 = np.zeros((fish_N,fish_n))
fish_X5 = np.zeros((fish_N,fish_n))

fish_m1 = np.zeros((fish_N,1))
fish_m2 = np.zeros((fish_N,1))
fish_m3 = np.zeros((fish_N,1))
fish_m4 = np.zeros((fish_N,1))
fish_m5 = np.zeros((fish_N,1))
fish_m = np.zeros((fish_N,1))

#SW matrix
fish_Sw1 = np.zeros((fish_N,fish_N))
fish_Sw2 = np.zeros((fish_N,fish_N))
fish_Sw3 = np.zeros((fish_N,fish_N))
fish_Sw4 = np.zeros((fish_N,fish_N))
fish_Sw5 = np.zeros((fish_N,fish_N))
fish_Sw = np.zeros((fish_N,fish_N))

#SB matrix
fish_Sb1 = np.zeros((fish_N,fish_N))
fish_Sb2 = np.zeros((fish_N,fish_N))
fish_Sb3 = np.zeros((fish_N,fish_N))
fish_Sb4 = np.zeros((fish_N,fish_N))
fish_Sb5 = np.zeros((fish_N,fish_N))
fish_Sb = np.zeros((fish_N,fish_N))

#W = np.ones((5, fish_N))

#Create noisy data (similar to lsq)
fish_X1 = np.transpose(fish_X1);
for i in range(0,fish_n):
  fish_X1[i] = fn1_trainY[9] + np.random.normal(size = fish_N);
fish_X1 = np.transpose(fish_X1);
  
fish_X2 = np.transpose(fish_X2);
for i in range(0,fish_n):
  fish_X2[i] = fn2_trainY[9] + np.random.normal(size = fish_N);
fish_X2 = np.transpose(fish_X2);

fish_X3 = np.transpose(fish_X3);
for i in range(0,fish_n):
  fish_X3[i] = fn3_trainY[9] + np.random.normal(size = fish_N);
fish_X3 = np.transpose(fish_X3);
  
fish_X4 = np.transpose(fish_X4);
for i in range(0,fish_n):
  fish_X4[i] = fn4_trainY[9] + np.random.normal(size = fish_N);
fish_X4 = np.transpose(fish_X4);

fish_X5 = np.transpose(fish_X5);
for i in range(0,fish_n):
  fish_X5[i] = fn5_trainY[9] + np.random.normal(size = fish_N);
fish_X5 = np.transpose(fish_X5);
                                             
#Find m's
                                                  
for i in range(0,fish_N):
  fish_m1[i] = np.average(fish_X1[i]);
    
for i in range(0,fish_N):
  fish_m2[:,0] = np.average(fish_X2[i]);
   
for i in range(0,fish_N):
  fish_m3[:,0] = np.average(fish_X3[i]);
                                                  
for i in range(0,fish_N):
  fish_m4[:,0] = np.average(fish_X4[i]);
                                                  
for i in range(0,fish_N):
  fish_m5[:,0] = np.average(fish_X5[i]);
                                                  
fish_m = (fish_m1 + fish_m2+ fish_m3 + fish_m4 + fish_m5)/5    
  
#Build Sw
for i in range(0,fish_n):
  fish_Sw1 = fish_X1[:,i] - fish_m1
  fish_Sw1 = fish_Sw1 + np.matmul(fish_Sw1,np.transpose(fish_Sw1))
  
for i in range(0,fish_n):
  fish_Sw2 = fish_X2[:,i] - fish_m2
  fish_Sw2 = fish_Sw2 + np.matmul(fish_Sw2,np.transpose(fish_Sw2))
    
for i in range(0,fish_n):
  fish_Sw3 = fish_X3[:,i] - fish_m3
  fish_Sw3 = fish_Sw3 + np.matmul(fish_Sw3,np.transpose(fish_Sw3))
    
for i in range(0,fish_n):
  fish_Sw4 = fish_X4[:,i] - fish_m4
  fish_Sw4 = fish_Sw4 + np.matmul(fish_Sw4,np.transpose(fish_Sw4))
  
for i in range(0,fish_n):
  fish_Sw5 = fish_X5[:,i] - fish_m5
  fish_Sw5 = fish_Sw5 + np.matmul(fish_Sw5,np.transpose(fish_Sw5))
                                                  
fish_Sw = fish_Sw1 + fish_Sw2 + fish_Sw3 + fish_Sw4 + fish_Sw5

#Build SB
fish_Sb1 = np.matmul((fish_m1 - fish_m),np.transpose(fish_m1 - fish_m)) * fish_N
fish_Sb2 = np.matmul((fish_m2 - fish_m),np.transpose(fish_m2 - fish_m)) * fish_N
fish_Sb3 = np.matmul((fish_m3 - fish_m),np.transpose(fish_m3 - fish_m)) * fish_N
fish_Sb4 = np.matmul((fish_m4 - fish_m),np.transpose(fish_m4 - fish_m)) * fish_N
fish_Sb5 = np.matmul((fish_m5 - fish_m),np.transpose(fish_m5 - fish_m)) * fish_N
fish_Sb = fish_Sb1 + fish_Sb2 + fish_Sb3 + fish_Sb4 + fish_Sb5
  
#print(fish_Sw.size)
#print(fish_Sb)
  
#print(np.matmul(W,fish_Sw))
def fish_multiple_J(W):
    W = W.reshape(100,5).T;
    return np.trace(np.matmul(np.linalg.inv(np.matmul(np.matmul(W,fish_Sw),np.transpose(W))), np.matmul(np.matmul(W,fish_Sb),np.transpose(W))))
  
w0 = np.linspace(1,5*fish_n*2,500);

res = sciop.minimize(fish_multiple_J,w0,method = 'BFGS', options={'disp':True,'maxiter':25000}, tol=1e-100)
print(res.x)