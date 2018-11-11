from numpy import *
import matplotlib.pyplot as plt
import time
import numpy as np
import Logistic_Regression as LR
 
def loadData():
	train_x = []
	train_y = []
	fileIn = open('Testdata.txt', 'r', encoding="utf-8-sig").readlines()
	for line in fileIn:
		lineArr = line.strip().split()
		train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
		train_y.append(float(lineArr[2]))
	print(len(train_x))
	return np.mat(train_x), np.mat(train_y).transpose()
 
 
## step 1: load data
print("step 1: load data...")
train_x, train_y = loadData()
test_x = train_x; test_y = train_y
 
## step 2: training...
print("step 2: training...")
opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
optimalWeights = LR.trainLogRegres(train_x, train_y, opts)
 
## step 3: testing
print("step 3: testing...")
accuracy = LR.testLogRegres(optimalWeights, test_x, test_y)
 
## step 4: show the result
print("step 4: show the result...")	
print("The classify accuracy is: %.3f%%" % (accuracy * 100))
LR.showLogRegres(optimalWeights, train_x, train_y) 