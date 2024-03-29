# CS 3190 - HW3 - Question 1 
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




#-------PROBLEM 1-------
# 1.(a) - Predict the value of y for the new x values of 1 and 10.
x_bar = np.average(x)
y_bar = np.average(y)
X_bar = x - x_bar
Y_bar = y - y_bar
a = X_bar.dot(Y_bar)/X_bar.dot(X_bar) 
b = y_bar - a * x_bar
#print(x_bar, y_bar)
#print(a, b)
print('\n')
 
def lin_reg(x):
  return a * x + b

print("Solution for 1.(a):")
h = np.linspace(0,25,50)
plt.scatter(x, y, c="blue")
plt.scatter(x_bar, y_bar, c="red")
plt.plot(h, lin_reg(h), 'g', linewidth=2)
plt.show()

print('Predicted y for x=1:', lin_reg(1))
print('Predicted y for x=10:', lin_reg(10))
print('\n')




# 1.(b) - Split the data into a training set (the first 100 values) and test set
#         (the last 20 values); predict the y value at x values of 1 and 10.
x_train = x[:100]
x_test = x[-20:]
y_train = y[:100]
y_test = y[-20:]
#print(y_train, '\n')
#print(y_test, '\n')

xx_bar = np.average(x_train)
yy_bar = np.average(y_train)
XX_bar = x_train - xx_bar
YY_bar = y_train - yy_bar
aa = XX_bar.dot(YY_bar)/XX_bar.dot(XX_bar) 
bb = yy_bar - aa * xx_bar

def lin_regr(x):
  return aa * x + bb

print("Solution for 1.(b):")
h = np.linspace(0,25,50)
plt.scatter(x_train, y_train, c="blue")
plt.scatter(x_test, y_test, c="orange")
plt.scatter(xx_bar, yy_bar, c="red")
plt.scatter(1, lin_regr(1), c="black")
plt.scatter(10, lin_regr(10), c="black")
plt.plot(h, lin_regr(h), 'g', linewidth=2)
plt.show()

print('Predicted y for x=1:', lin_regr(1))
print('Predicted y for x=10:', lin_regr(10))
print('\n')




# 1.(c) - Build 2 models. One on just the training data (100 data points), and 
#         one on all data (120 data points). For each model report the RMSE 
#         error on 2 data sets: one on just testing data (20 data points)
#         one training data (100 data points).  

print("Solution for 1.(c):")
# residual vector model for full data (120 points) using data from part 1.(a)
y_hat_full = np.zeros(120)
for i in range(120):
  y_hat_full[i] = a * x[i] + b
r_full = y - y_hat_full
print('Residual vector of full data:')
print(r_full, '\n')

# residual vector model for training data (100 points) using data from part 1.(b)
y_hat_train = np.zeros(100)
for i in range(100):
  y_hat_train[i] = aa * x_train[i] + bb
r_train = y_train - y_hat_train
print('Residual vector of training data:')
print(r_train, '\n')

# RMSE for training data (n=100 points) using data from part 1.(b)
RMSE_train = np.zeros(100)
for i in range(100):
  RMSE_train[i] = math.sqrt(1/100 * r_train[i] ** 2)
print('RMSE of training data:')
print(RMSE_train, '\n')

# residual vector model for testing data (20 points) using data from part 1.(b)
y_hat_test = np.zeros(20)
for i in range(20):
  y_hat_test[i] = aa * x_test[i] + bb
r_test = y_test - y_hat_test
print('Residual vector of testing data:')
print(r_test, '\n')

# RMSE for testing data (n=20 points) using data from part 1.(b)
RMSE_test = np.zeros(20)
for i in range(20):
  RMSE_test[i] = math.sqrt(1/20 * r_test[i] ** 2)
print('RMSE of testing data:')
print(RMSE_test, '\n\n')



# 1.(d) - Expand data set x into n × (p + 1) matrix X˜p using standard polynomial
#         expansion for p = 3. Report the first 3 rows of this matrix. 
#         Report the degree-3 model using this matrix on the training data.
#         Report RMSE of the residual vector for the testing data
#         (20-dimensional vector) and training data (100-dimensional vector).
print("Solution for 1.(d):")
Xp = [[0] * 4 for i in range(120)]  # columns = 4, rows = 120
for i in range(120):
  for j in range(4):
    if j == 0:
      Xp[i][j] = 1
    if j == 1:
      Xp[i][j] = x[i] ** 1
    if j == 2:
      Xp[i][j] = x[i] ** 2
    if j == 3:
      Xp[i][j] = x[i] ** 3

print('First 3 rows of expanded full data:')
for i in range(3):
  print(Xp[i][0:4])
print('\n')

# plot degree-3 polynomial model using this matrix 
def plot_poly(x, y, p):
  plt.scatter(x, y, s=80, c="blue")
  #plt.axis([60,76,100,240])
  s = np.linspace(-10,25,101)

  coefs = np.polyfit(x, y, p)
  print("Coefficients: \n", coefs)
  ffit = np.poly1d(coefs)
  plt.plot(s, ffit(s), 'g-', linewidth=2.0)

  title = "degree %s fit" % p
  plt.title(title)
  plt.show()
plot_poly(x_train, y_train, 3)
print('\n')


# get residual vector & RMSE for training data (100-dimensional vector)
# polynomial expansion of training data from part 1.(b)
Xp_train = [[0] * 4 for i in range(100)]  # columns = 4, rows = 100
for i in range(100):
  for j in range(4):
    if j == 0:
      Xp_train[i][j] = 1
    if j == 1:
      Xp_train[i][j] = x_train[i] ** 1
    if j == 2:
      Xp_train[i][j] = x_train[i] ** 2
    if j == 3:
      Xp_train[i][j] = x_train[i] ** 3
#print('Polynomial expansion of training data:')
#for i in range(100):
#  print(Xp_train[i][0:4])
#print('\n')

# residual vector of expanded training data
coefs_train = np.polyfit(x_train, y_train, 3)
ffit_train = np.poly1d(coefs_train)
residual_vector_train = ffit_train(x_train) - y_train

# RMSE of expanded training data
RMSE_xp_train = np.zeros(100)
for i in range(100):
  RMSE_xp_train[i] = math.sqrt(1/100 * residual_vector_train[i] ** 2)
print('RMSE of expanded training data:')
print(RMSE_xp_train, '\n')


# polynomial expansion of testing data from part 1.(b)
Xp_test = [[0] * 4 for i in range(20)]  # columns = 4, rows = 20
for i in range(20):
  for j in range(4):
    if j == 0:
      Xp_test[i][j] = 1
    if j == 1:
      Xp_test[i][j] = x_test[i] ** 1
    if j == 2:
      Xp_test[i][j] = x_test[i] ** 2
    if j == 3:
      Xp_test[i][j] = x_test[i] ** 3
#print('Polynomial expansion of testing data:')
#for i in range(20):
#  print(Xp_test[i][0:4])
#print('\n')

# residual vector of testing data
coefs_test = np.polyfit(x_test, y_test, 3)
ffit_test = np.poly1d(coefs_test)
residual_vector_test = ffit_test(x_test) - y_test

# RMSE of testing data
RMSE_xp_test = np.zeros(20)
for i in range(20):
  RMSE_xp_test[i] = math.sqrt(1/20 * residual_vector_test[i] ** 2)
print('RMSE of expanded testing data:')
print(RMSE_xp_test, '\n')


