# CS 3190 - HW3 - Question 3 
# Jessika Jimenez

import numpy as np 
from scipy import linalg as LA
import pandas as pd
import random 
import math
import matplotlib as mpl
import matplotlib.pyplot as plt 
%matplotlib inline
from google.colab import drive
drive.mount("/content/gdrive")

# import & save values from x.csv
def read_file(file_x):
  x_data = []
  with open(file_x, "r") as f:
    for line in f:
      item = line.strip().split(",")
      x_data.append(np.array(item))
  return x_data
x_data = read_file('/content/gdrive/My Drive/University of Utah/2022 Fall/CS 3190/x.csv')
x = np.zeros(len(x_data))
for i in range(len(x_data)):
  x[i] = float(x_data[i])
#print(x, '\n')

# import & save values from y.csv
def read_file(file_y):
  y_data = []
  with open(file_y, "r") as f:
    for line in f:
      item = line.strip().split(",")
      y_data.append(np.array(item))
  return y_data
y_data = read_file('/content/gdrive/My Drive/University of Utah/2022 Fall/CS 3190/y.csv')
y = np.zeros(len(y_data))
for i in range(len(y_data)):
  y[i] = float(y_data[i])
#print(y, '\n')
print('\n\n')



#-------PROBLEM 3-------
# 3.(a) - Run gradient descent on f1 with starting point (x, y) = (0, 0), 
#         T = 30 steps and γ = .01
# expanded f1 function - f1(x, y) = (x − 3)^2 + 3(y + 1)^2
def func_f1(x, y):
  return ( (x-3)**2 + 3*(y+1)**2 )

# gradient of function f1
def func_grad(x, y):
  dfdx = 2*x - 6
  dfdy = 6*y + 6
  return np.array([dfdx, dfdy])

# prep for plot
xlist = np.linspace(-0.5, 1.5, 26)
ylist = np.linspace(-1, 1.5, 26)
x_, y_ = np.meshgrid(xlist, ylist)
z_ = func_f1(x_, y_)
lev = np.linspace(0, 20, 21)

# initialize gradient descent at (0,0), 30 iterations, 𝛾=0.01
v_init = np.array([0, 0])
num_iter = 30
values = np.zeros([num_iter, 2]) 
values[0,:] = v_init
v = v_init
gamma = 0.01

# gradient descent algorithm
for i in range(1,num_iter):
  _func = func_f1(v[0], v[1])
  _grad = func_grad(v[0], v[1])
  grad_norm = LA.norm(func_grad(v[0], v[1]))
  v = v - gamma * func_grad(v[0], v[1])
  values[i,:] = v
  print("Function value: {} | Gradient: {} | Gradient norm: {}".format(_func, _grad, grad_norm))

# plot
plt.contour(x_, y_, z_, levels = lev)
plt.plot(values[:,0], values[:,1], 'r-')
plt.plot(values[:,0], values[:,1], 'bo')
title = "Part (a) | gamma %0.02f" % (gamma)
plt.title(title)
plt.show()
print('\n\n')




# 3.(b) - Run gradient descent on f1 with starting point (x, y) = (10, 10), 
#         T = 100 steps and γ = .03
# prep for plot
xxlist = np.linspace(2, 10, 26)
yylist = np.linspace(-2, 10, 26)
xx_, yy_ = np.meshgrid(xxlist, yylist)
zz_ = func_f1(xx_, yy_)
lev_ = np.linspace(0, 20, 21)

# initialize gradient descent at (10,10), 100 iterations, 𝛾=0.03
vv_init = np.array([10, 10])  
nums_iter = 100
valuess = np.zeros([nums_iter, 2])
valuess[0,:] = vv_init
vv = vv_init
gammaa = 0.03

# gradient descent algorithm
for i in range(1,nums_iter):
  _func_f1 = func_f1(vv[0], vv[1])
  _grad_f1 = func_grad(vv[0], vv[1])
  grad_normm = LA.norm(func_grad(vv[0], vv[1]))
  vv = vv - gammaa * func_grad(vv[0], vv[1])
  valuess[i,:] = vv
  print("Function value: {} | Gradient: {} | Gradient norm: {}".format(_func_f1, _grad_f1, grad_normm))

# plot
plt.contour(xx_, yy_, zz_, levels = lev_)
plt.plot(valuess[:,0], valuess[:,1], 'r-')
plt.plot(valuess[:,0], valuess[:,1], 'bo')
title = "Part (b) | gamma %0.02f" % (gammaa)
plt.title(title)
plt.show()
print('\n\n')





# 3.(c) - Run any variant gradient descent on f2 with starting point 
#         (x, y) = (0, 2), T = 100 steps
# expanded f2 function - f2(x, y) = (1 − (y − 3))^2 + 10((x + 4) − (y − 3)^2)^2
def func_f2(x, y):
  return ( (1 - (y-3))**2 + 10 * ((x+4) - (y-3)**2)**2 )

# gradient of function f2
def func_grad_f2(x, y):
  ddx = 20 * (x - (y-3)**2 + 4)
  ddy = 2 * (-20 * (y-3) * (x - (y-3)**2 + 4) + y - 4)
  return np.array([ddx, ddy])

# prep for plot
_xlist = np.linspace(-0.2, 0.2, 26)
_ylist = np.linspace(0.5, 1.5, 26)
_x, _y = np.meshgrid(_xlist, _ylist)
_z = func_f2(_x, _y)
_lev = np.linspace(0, 50, 21)

# initialize gradient descent at (0,0), 100 iterations, & trying 𝛾=0.01
v_initial = np.array([0, 0])
num_it = 100
_values = np.zeros([num_it, 2]) 
_values[0,:] = v_initial
_v = v_initial
_gamma = 0.001

# gradient descent algorithm
for i in range(1,num_it):
  _func_f2 = func_f2(_v[0], _v[1])
  _grad_f2 = func_grad_f2(_v[0], _v[1])
  _grad_norm = LA.norm(func_grad_f2(_v[0], _v[1]))
  _v = _v - _gamma * func_grad_f2(_v[0], _v[1])
  _values[i,:] = _v
  print("Function value: {} | Gradient: {} | Gradient norm: {}".format(_func_f2, _grad_f2, _grad_norm))

# plot
plt.contour(_x, _y, _z, levels = _lev)
plt.plot(_values[:,0], _values[:,1], 'r-')
plt.plot(_values[:,0], _values[:,1], 'bo')
title = "Part (c) | gamma %0.03f" % (_gamma)
plt.title(title)
plt.show()
print('\n\n')


