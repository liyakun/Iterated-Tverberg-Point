#! /usr/bin/pythonw
#This file contains the optimization algorithm

import matplotlib.pyplot as plt
import numpy as np
import time

class Optimization:

    error = []
    weights_all = []

    def __init__(self):
        pass

    #gradient ascent optional optimize algorithm
    def gradAscent(self, trainMatrix, classLabels, dataIndexList, numIteration, function):
        startTime = time.time() #get start time
        m, n = np.shape(trainMatrix)
        weights_tmp = np.ones((n,1))
        for i in range(numIteration):
            dataIndex = range(len(dataIndexList))
            for j in range(len(dataIndexList)):
                alpha = 4/(1.0+j+i) + 0.01      #alpha changes on each iteration
                randIndex = int(np.random.uniform(0, len(dataIndex)))  #randomly selecting each index from dataIndexList, then from dataMatrix to use in updating the weights
                h = function.sigmoid(np.sum(trainMatrix[dataIndexList[randIndex]]*weights_tmp))
                error_tmp = (classLabels[dataIndexList[randIndex]] - h)
                self.error.append(error_tmp)
                self.weights_all.append(np.linalg.norm(np.mat(weights_tmp)))
                weights_tmp = weights_tmp + alpha * trainMatrix[randIndex].transpose() * error_tmp
                del (dataIndex[randIndex])
        print "\nTraining finished within %fs!\n" % (time.time() - startTime)
        return weights_tmp

