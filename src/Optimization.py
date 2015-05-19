#! /usr/bin/pythonw
#This file contains the optimization algorithm
import numpy as np
import time

class Optimization:

    weights_all = []

    def __init__(self):
        pass

    #sigmoid function
    def sigmoid(self, inX):
        return 1.0/(1+np.exp(-inX))

    #gradient ascent optional optimize algorithm
    def gradAscent(self, trainMatrix, classLabels, dataIndexList, numIteration, function):
        self.weights_all = []
        startTime = time.time() #get start time
        m, n = np.shape(trainMatrix)
        weights_tmp = np.ones((n, 1))
        for i in range(numIteration):
            dataIndex = range(len(dataIndexList))
            for j in range(len(dataIndexList)):
                #alpha changes on each iteration
                alpha = 4/(1.0+j+i) + 0.001
                #randomly selecting each index from dataIndexList, then from dataMatrix to use in updating the weights
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                #get the sigmoid value
                h = function.sigmoid(np.sum(trainMatrix[dataIndexList[randIndex]]*weights_tmp))
                #store all the weights
                self.weights_all.append(np.linalg.norm(np.mat(weights_tmp)))
                #update weights in current training example
                weights_tmp = weights_tmp + alpha * trainMatrix[randIndex].transpose() * error_tmp
                del (dataIndex[randIndex])
        print "\nTraining finished within %fs!\n" % (time.time() - startTime)
        return weights_tmp

    def sigTest(self, instance, weights):
        sig_value = self.sigmoid(np.sum(instance*weights))
        if sig_value > 0.5:
            return 1.0
        else:
            return 0.0
