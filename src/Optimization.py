#! /usr/bin/pythonw
#This file contains the optimization algorithm

import matplotlib.pyplot as plt
from numpy import *
import time

class Optimization:

    def __init__(self):
        pass

    #gradient ascent optional optimize algorithm
    def gradAscent(self, dataMatrix, classLabels, numIteration, function):
        startTime = time.time() #get start time
        m, n = shape(dataMatrix)
        weights = ones((n,1))
        for i in range(numIteration):
            dataIndex = range(m)
            for j in range(m):
                alpha = 4/(1.0+j+i) + 0.01      #alpha changes on each iteration
                randIndex = int(random.uniform(0, len(dataIndex)))  #randomly selecting each instance to use in updating the weights
                h = function.sigmoid(sum(dataMatrix[randIndex]*weights))
                error = (classLabels[randIndex] - h)
                weights = weights + alpha * dataMatrix[randIndex].transpose() * error
                del (dataIndex[randIndex])
        print "\nOptimization finish within %fs!\n" %(time.time() - startTime)
        return weights